"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
Content is stored in a vector database for RAG queries.
"""

from mcp.server.fastmcp import FastMCP, Context
from sentence_transformers import CrossEncoder
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from supabase import Client
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import concurrent.futures


from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    MemoryAdaptiveDispatcher,
)  # noqa: E402
from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy  # noqa: E402
from utils import (  # noqa: E402
    get_supabase_client,
    add_documents_to_supabase,
    search_documents,
    extract_code_blocks,
    generate_code_example_summary,
    add_code_examples_to_supabase,
    update_source_info,
    extract_source_summary,
    analyze_content_distribution,
    create_combined_code_block,
    # GitHub utilities (API-based)
    parse_github_url,
    get_repository_tree,
    get_file_content,
    filter_crawlable_files,
    extract_language_metadata,
    process_notebook_as_script,
    SKIP_EXTENSIONS,
    # Local git operations
    clone_github_repository,
    get_local_file_tree,
    read_local_file_content,
    cleanup_temp_directory,
    create_temp_directory,
    # LLM error handling
    LLMProcessingError,
)


# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)


# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""

    crawler: AsyncWebCrawler
    supabase_client: Client
    reranking_model: Optional[CrossEncoder] = None


@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.

    Args:
        server: The FastMCP server instance

    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Supabase client
    """
    # Create browser configuration
    browser_config = BrowserConfig(headless=True, verbose=False)

    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()

    # Initialize Supabase client
    supabase_client = get_supabase_client()

    # Initialize cross-encoder model for reranking if enabled
    reranking_model = None
    if os.getenv("USE_RERANKING", "false") == "true":
        try:
            reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Failed to load reranking model: {e}")
            reranking_model = None

    try:
        yield Crawl4AIContext(
            crawler=crawler,
            supabase_client=supabase_client,
            reranking_model=reranking_model,
        )
    finally:
        # Clean up all components
        await crawler.__aexit__(None, None, None)


# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051"),
)


def rerank_results(
    model: CrossEncoder,
    query: str,
    results: List[Dict[str, Any]],
    content_key: str = "content",
) -> List[Dict[str, Any]]:
    """
    Rerank search results using a cross-encoder model.

    Args:
        model: The cross-encoder model to use for reranking
        query: The search query
        results: List of search results
        content_key: The key in each result dict that contains the text content

    Returns:
        Reranked list of results
    """
    if not model or not results:
        return results

    try:
        # Extract content from results
        texts = [result.get(content_key, "") for result in results]

        # Create pairs of [query, document] for the cross-encoder
        pairs = [[query, text] for text in texts]

        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)

        # Add scores to results and sort by score (descending)
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        return reranked
    except Exception as e:
        print(f"Error during reranking: {e}")
        return results


def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.

    Args:
        url: URL to check

    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path


def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.

    Args:
        url: URL to check

    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith(".txt")


def is_pdf(url: str) -> bool:
    """
    Check if a URL is a PDF file.

    Args:
        url: URL to check

    Returns:
        True if the URL is a PDF file, False otherwise
    """
    return url.lower().endswith(".pdf")


def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.

    Args:
        sitemap_url: URL of the sitemap

    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall(".//{*}loc")]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls


def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif "\n\n" in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind("\n\n")
            if (
                last_break > chunk_size * 0.3
            ):  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif ". " in chunk:
            # Find the last sentence break
            last_period = chunk.rfind(". ")
            if (
                last_period > chunk_size * 0.3
            ):  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks


def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.

    Args:
        chunk: Markdown chunk

    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r"^(#+)\s+(.+)$", chunk, re.MULTILINE)
    header_str = "; ".join([f"{h[0]} {h[1]}" for h in headers]) if headers else ""

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split()),
    }


def process_code_example(args):
    """
    Process a single code example to generate its summary.
    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (code, context_before, context_after, is_code_dominated)

    Returns:
        The generated summary
    """
    if len(args) == 4:
        code, context_before, context_after, is_code_dominated = args
        return generate_code_example_summary(
            code, context_before, context_after, is_code_dominated
        )
    else:
        # Backward compatibility
        code, context_before, context_after = args
        return generate_code_example_summary(code, context_before, context_after)


@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page or PDF and store its content in Supabase.

    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Supabase for later retrieval and querying. Automatically detects
    and processes PDF files using appropriate PDF extraction strategies.

    Args:
        ctx: The MCP server provided context
        url: URL of the web page or PDF to crawl

    Returns:
        Summary of the crawling operation and storage in Supabase
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Check if URL is a PDF and configure accordingly
        if is_pdf(url):
            # PDF-specific configuration
            pdf_crawler_strategy = PDFCrawlerStrategy()
            pdf_scraping_strategy = PDFContentScrapingStrategy()
            
            # Use PDF-specific crawler and scraping strategy
            async with AsyncWebCrawler(crawler_strategy=pdf_crawler_strategy) as pdf_crawler:
                run_config = CrawlerRunConfig(
                    cache_mode=CacheMode.BYPASS, 
                    stream=False,
                    scraping_strategy=pdf_scraping_strategy
                )
                result = await pdf_crawler.arun(url=url, config=run_config)
        else:
            # Regular web page configuration
            run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
            result = await crawler.arun(url=url, config=run_config)

        if result.markdown:
            # Extract source_id
            parsed_url = urlparse(url)
            source_id = parsed_url.netloc or parsed_url.path

            # Chunk the content
            chunks = smart_chunk_markdown(result.markdown)

            # Prepare data for Supabase
            urls = []
            chunk_numbers = []
            contents = []
            metadatas = []
            total_word_count = 0

            for i, chunk in enumerate(chunks):
                urls.append(url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = url
                meta["source"] = source_id
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

                # Accumulate word count
                total_word_count += meta.get("word_count", 0)

            # Create url_to_full_document mapping
            url_to_full_document = {url: result.markdown}

            # Update source information FIRST (before inserting documents)
            source_summary = extract_source_summary(
                source_id, result.markdown[:5000]
            )  # Use first 5000 chars for summary
            update_source_info(
                supabase_client, source_id, source_summary, total_word_count
            )

            # Add documentation chunks to Supabase (AFTER source exists)
            add_documents_to_supabase(
                supabase_client,
                urls,
                chunk_numbers,
                contents,
                metadatas,
                url_to_full_document,
            )

            # Extract and process code examples only if enabled
            extract_code_examples = os.getenv("USE_AGENTIC_RAG", "false") == "true"
            code_examples_combined = False
            if extract_code_examples:
                # Analyze content distribution to determine processing approach
                code_dominance_threshold = float(
                    os.getenv("CODE_DOMINANCE_THRESHOLD", "0.4")
                )
                content_analysis = analyze_content_distribution(result.markdown)
                is_code_dominated = (
                    content_analysis["code_percentage"] >= code_dominance_threshold
                )

                # Extract code blocks with appropriate min_length based on content type
                min_length = 0 if is_code_dominated else 1000
                code_blocks = extract_code_blocks(
                    result.markdown, min_length=min_length
                )
                if code_blocks:
                    code_urls = []
                    code_chunk_numbers = []
                    code_examples = []
                    code_summaries = []
                    code_metadatas = []

                    if is_code_dominated:
                        # Code-dominated content: combine all code blocks into one
                        code_examples_combined = True
                        combined_block = create_combined_code_block(
                            result.markdown, code_blocks
                        )

                        # Generate summary for combined block
                        summary = generate_code_example_summary(
                            combined_block["code"],
                            combined_block["context_before"],
                            combined_block["context_after"],
                            True,
                        )

                        # Prepare single combined entry
                        code_urls.append(url)
                        code_chunk_numbers.append(0)
                        code_examples.append(combined_block["code"])
                        code_summaries.append(summary)

                        # Create metadata for combined block
                        code_meta = {
                            "chunk_index": 0,
                            "url": url,
                            "source": source_id,
                            "char_count": len(combined_block["code"]),
                            "word_count": len(combined_block["code"].split()),
                            "language": combined_block["language"],
                        }
                        code_metadatas.append(code_meta)

                    else:
                        # Regular content: process individual code blocks
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=10
                        ) as executor:
                            # Prepare arguments for parallel processing
                            summary_args = [
                                (
                                    block["code"],
                                    block["context_before"],
                                    block["context_after"],
                                    False,  # is_code_dominated
                                )
                                for block in code_blocks
                            ]

                            # Generate summaries in parallel
                            summaries = list(
                                executor.map(process_code_example, summary_args)
                            )

                        # Prepare individual code block data
                        for i, (block, summary) in enumerate(
                            zip(code_blocks, summaries)
                        ):
                            code_urls.append(url)
                            code_chunk_numbers.append(i)
                            code_examples.append(block["code"])
                            code_summaries.append(summary)

                            # Create metadata for individual block
                            code_meta = {
                                "chunk_index": i,
                                "url": url,
                                "source": source_id,
                                "char_count": len(block["code"]),
                                "word_count": len(block["code"].split()),
                                "language": block.get("language", ""),
                            }
                            code_metadatas.append(code_meta)

                    # Add code examples to Supabase
                    add_code_examples_to_supabase(
                        supabase_client,
                        code_urls,
                        code_chunk_numbers,
                        code_examples,
                        code_summaries,
                        code_metadatas,
                    )

            # Format code examples message
            if extract_code_examples and code_blocks:
                code_examples_msg = f"{len(code_blocks)} code examples stored"
                if code_examples_combined:
                    code_examples_msg += " (combined)"
            else:
                code_examples_msg = "0 code examples stored"

            return json.dumps(
                {
                    "success": True,
                    "url": url,
                    "chunks_stored": len(chunks),
                    "code_examples_stored": len(code_blocks) if code_blocks else 0,
                    "code_examples_message": code_examples_msg,
                    "content_length": len(result.markdown),
                    "total_word_count": total_word_count,
                    "source_id": source_id,
                    "links_count": {
                        "internal": len(result.links.get("internal", [])),
                        "external": len(result.links.get("external", [])),
                    },
                },
                indent=2,
            )
        else:
            return json.dumps(
                {"success": False, "url": url, "error": result.error_message}, indent=2
            )
    except Exception as e:
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)


@mcp.tool()
async def smart_crawl_url(
    ctx: Context,
    url: str,
    max_depth: int = 3,
    max_concurrent: int = 10,
    chunk_size: int = 5000,
) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Supabase.

    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth

    All crawled content is chunked and stored in Supabase for later retrieval and querying.

    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters (default: 1000)

    Returns:
        JSON string with crawl summary and storage information
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Determine the crawl strategy
        crawl_results = []
        crawl_type = None

        if is_pdf(url):
            # For PDF files, use PDF-specific crawl
            crawl_results = await crawl_pdf_file(url)
            crawl_type = "pdf_file"
        elif is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps(
                    {"success": False, "url": url, "error": "No URLs found in sitemap"},
                    indent=2,
                )
            crawl_results = await crawl_batch(
                crawler, sitemap_urls, max_concurrent=max_concurrent
            )
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(
                crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent
            )
            crawl_type = "webpage"

        if not crawl_results:
            return json.dumps(
                {"success": False, "url": url, "error": "No content found"}, indent=2
            )

        # Process results and store in Supabase
        urls = []
        chunk_numbers = []
        contents = []
        metadatas = []
        chunk_count = 0

        # Track sources and their content
        source_content_map = {}
        source_word_counts = {}

        # Process documentation chunks
        for doc in crawl_results:
            source_url = doc["url"]
            md = doc["markdown"]
            chunks = smart_chunk_markdown(md, chunk_size=chunk_size)

            # Extract source_id
            parsed_url = urlparse(source_url)
            source_id = parsed_url.netloc or parsed_url.path

            # Store content for source summary generation
            if source_id not in source_content_map:
                source_content_map[source_id] = md[:5000]  # Store first 5000 chars
                source_word_counts[source_id] = 0

            for i, chunk in enumerate(chunks):
                urls.append(source_url)
                chunk_numbers.append(i)
                contents.append(chunk)

                # Extract metadata
                meta = extract_section_info(chunk)
                meta["chunk_index"] = i
                meta["url"] = source_url
                meta["source"] = source_id
                meta["crawl_type"] = crawl_type
                meta["crawl_time"] = str(asyncio.current_task().get_coro().__name__)
                metadatas.append(meta)

                # Accumulate word count
                source_word_counts[source_id] += meta.get("word_count", 0)

                chunk_count += 1

        # Create url_to_full_document mapping
        url_to_full_document = {}
        for doc in crawl_results:
            url_to_full_document[doc["url"]] = doc["markdown"]

        # Update source information for each unique source FIRST (before inserting documents)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            source_summary_args = [
                (source_id, content)
                for source_id, content in source_content_map.items()
            ]
            source_summaries = list(
                executor.map(
                    lambda args: extract_source_summary(args[0], args[1]),
                    source_summary_args,
                )
            )

        for (source_id, _), summary in zip(source_summary_args, source_summaries):
            word_count = source_word_counts.get(source_id, 0)
            update_source_info(supabase_client, source_id, summary, word_count)

        # Add documentation chunks to Supabase (AFTER sources exist)
        batch_size = 10
        await add_documents_to_supabase(
            supabase_client,
            urls,
            chunk_numbers,
            contents,
            metadatas,
            url_to_full_document,
            batch_size=batch_size,
        )

        # Extract and process code examples from all documents only if enabled
        extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
        total_original_code_blocks = 0
        pages_with_combined_code = 0
        if extract_code_examples_enabled:
            code_urls = []
            code_chunk_numbers = []
            code_examples = []
            code_summaries = []
            code_metadatas = []

            # Get threshold for educational content detection
            code_dominance_threshold = float(
                os.getenv("CODE_DOMINANCE_THRESHOLD", "0.4")
            )

            # Extract code blocks from all documents
            for doc in crawl_results:
                source_url = doc["url"]
                md = doc["markdown"]

                # Analyze content distribution to determine processing approach
                content_analysis = analyze_content_distribution(md)
                is_code_dominated = (
                    content_analysis["code_percentage"] >= code_dominance_threshold
                )

                # Extract code blocks with appropriate min_length based on content type
                min_length = 0 if is_code_dominated else 1000
                code_blocks = extract_code_blocks(md, min_length=min_length)

                if code_blocks:
                    total_original_code_blocks += len(code_blocks)
                    parsed_url = urlparse(source_url)
                    source_id = parsed_url.netloc or parsed_url.path

                    if is_code_dominated:
                        # Code-dominated content: combine all code blocks into one
                        pages_with_combined_code += 1
                        combined_block = create_combined_code_block(md, code_blocks)

                        # Generate summary for combined block
                        summary = generate_code_example_summary(
                            combined_block["code"],
                            combined_block["context_before"],
                            combined_block["context_after"],
                            True,
                        )

                        # Prepare single combined entry
                        code_urls.append(source_url)
                        code_chunk_numbers.append(len(code_examples))
                        code_examples.append(combined_block["code"])
                        code_summaries.append(summary)

                        # Create metadata for combined block
                        code_meta = {
                            "chunk_index": len(code_examples) - 1,
                            "url": source_url,
                            "source": source_id,
                            "char_count": len(combined_block["code"]),
                            "word_count": len(combined_block["code"].split()),
                            "language": combined_block["language"],
                        }
                        code_metadatas.append(code_meta)

                    else:
                        # Regular content: process individual code blocks
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=10
                        ) as executor:
                            # Prepare arguments for parallel processing
                            summary_args = [
                                (
                                    block["code"],
                                    block["context_before"],
                                    block["context_after"],
                                    False,  # is_code_dominated
                                )
                                for block in code_blocks
                            ]

                            # Generate summaries in parallel
                            summaries = list(
                                executor.map(process_code_example, summary_args)
                            )

                        # Prepare individual code block data
                        for i, (block, summary) in enumerate(
                            zip(code_blocks, summaries)
                        ):
                            code_urls.append(source_url)
                            code_chunk_numbers.append(len(code_examples))
                            code_examples.append(block["code"])
                            code_summaries.append(summary)

                            # Create metadata for individual block
                            code_meta = {
                                "chunk_index": len(code_examples) - 1,
                                "url": source_url,
                                "source": source_id,
                                "char_count": len(block["code"]),
                                "word_count": len(block["code"].split()),
                                "language": block.get("language", ""),
                            }
                            code_metadatas.append(code_meta)

            # Add all code examples to Supabase
            if code_examples:
                await add_code_examples_to_supabase(
                    supabase_client,
                    code_urls,
                    code_chunk_numbers,
                    code_examples,
                    code_summaries,
                    code_metadatas,
                    batch_size=batch_size,
                )
        else:
            code_examples = []

        # Format code examples message
        if extract_code_examples_enabled and total_original_code_blocks > 0:
            code_examples_msg = f"{total_original_code_blocks} code examples stored"
            if pages_with_combined_code > 0:
                code_examples_msg += f" ({pages_with_combined_code} pages combined)"
        else:
            code_examples_msg = "0 code examples stored"

        return json.dumps(
            {
                "success": True,
                "url": url,
                "crawl_type": crawl_type,
                "pages_crawled": len(crawl_results),
                "chunks_stored": chunk_count,
                "code_examples_stored": len(code_examples) if extract_code_examples_enabled else 0,
                "code_examples_message": code_examples_msg,
                "sources_updated": len(source_content_map),
                "urls_crawled": [doc["url"] for doc in crawl_results][:5]
                + (["..."] if len(crawl_results) > 5 else []),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)


@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources from the sources table.

    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database, along with their summaries and statistics. This is useful for discovering
    what content is available for querying.

    Always use this tool before calling the RAG query or code example query tool
    with a specific source filter!

    Args:
        ctx: The MCP server provided context

    Returns:
        JSON string with the list of available sources and their details
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Query the sources table directly
        result = (
            supabase_client.from_("sources").select("*").order("source_id").execute()
        )

        # Format the sources with their details
        sources = []
        if result.data:
            for source in result.data:
                sources.append(
                    {
                        "source_id": source.get("source_id"),
                        "summary": source.get("summary"),
                        "total_words": source.get("total_words"),
                        "created_at": source.get("created_at"),
                        "updated_at": source.get("updated_at"),
                    }
                )

        return json.dumps(
            {"success": True, "sources": sources, "count": len(sources)}, indent=2
        )
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


@mcp.tool()
async def perform_rag_query(
    ctx: Context, query: str, source: str = None, match_count: int = 5
) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.

    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.
    Get the source by using the get_available_sources tool before calling this search!

    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source and source.strip():
            filter_metadata = {"source": source}

        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search

            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata,
            )

            # 2. Get keyword search results using ILIKE
            keyword_query = (
                supabase_client.from_("crawled_pages")
                .select("id, url, chunk_number, content, metadata, source_id")
                .ilike("content", f"%{query}%")
            )

            # Apply source filter if provided
            if source and source.strip():
                keyword_query = keyword_query.eq("source_id", source)

            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []

            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []

            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get("id") for r in vector_results if r.get("id")}
            for kr in keyword_results:
                if kr["id"] in vector_ids and kr["id"] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get("id") == kr["id"]:
                            # Boost similarity score for items in both results
                            vr["similarity"] = min(1.0, vr.get("similarity", 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr["id"])
                            break

            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if (
                    vr.get("id")
                    and vr["id"] not in seen_ids
                    and len(combined_results) < match_count
                ):
                    combined_results.append(vr)
                    seen_ids.add(vr["id"])

            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr["id"] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append(
                        {
                            "id": kr["id"],
                            "url": kr["url"],
                            "chunk_number": kr["chunk_number"],
                            "content": kr["content"],
                            "metadata": kr["metadata"],
                            "source_id": kr["source_id"],
                            "similarity": 0.5,  # Default similarity for keyword-only matches
                        }
                    )
                    seen_ids.add(kr["id"])

            # Use combined results
            results = combined_results[:match_count]

        else:
            # Standard vector search only
            results = search_documents(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata,
            )

        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(
                ctx.request_context.lifespan_context.reranking_model,
                query,
                results,
                content_key="content",
            )

        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "content": result.get("content"),
                "metadata": result.get("metadata"),
                "similarity": result.get("similarity"),
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)

        return json.dumps(
            {
                "success": True,
                "query": query,
                "source_filter": source,
                "search_mode": "hybrid" if use_hybrid_search else "vector",
                "reranking_applied": use_reranking
                and ctx.request_context.lifespan_context.reranking_model is not None,
                "results": formatted_results,
                "count": len(formatted_results),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)


@mcp.tool()
async def search_code_examples(
    ctx: Context, query: str, source_id: str = None, match_count: int = 5
) -> str:
    """
    Search for code examples relevant to the query.

    This tool searches the vector database for code examples relevant to the query and returns
    the matching examples with their summaries. Optionally filter by source_id.
    Get the source_id by using the get_available_sources tool before calling this search!

    Use the get_available_sources tool first to see what sources are available for filtering.

    Args:
        ctx: The MCP server provided context
        query: The search query
        source_id: Optional source ID to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)

    Returns:
        JSON string with the search results
    """
    # Check if code example extraction is enabled
    extract_code_examples_enabled = os.getenv("USE_AGENTIC_RAG", "false") == "true"
    if not extract_code_examples_enabled:
        return json.dumps(
            {
                "success": False,
                "error": "Code example extraction is disabled. Perform a normal RAG search.",
            },
            indent=2,
        )

    try:
        # Get the Supabase client from the context
        supabase_client = ctx.request_context.lifespan_context.supabase_client

        # Check if hybrid search is enabled
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "false") == "true"

        # Prepare filter if source is provided and not empty
        filter_metadata = None
        if source_id and source_id.strip():
            filter_metadata = {"source": source_id}

        if use_hybrid_search:
            # Hybrid search: combine vector and keyword search

            # Import the search function from utils
            from utils import search_code_examples as search_code_examples_impl

            # 1. Get vector search results (get more to account for filtering)
            vector_results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count * 2,  # Get double to have room for filtering
                filter_metadata=filter_metadata,
            )

            # 2. Get keyword search results using ILIKE on both content and summary
            keyword_query = (
                supabase_client.from_("code_examples")
                .select("id, url, chunk_number, content, summary, metadata, source_id")
                .or_(f"content.ilike.%{query}%,summary.ilike.%{query}%")
            )

            # Apply source filter if provided
            if source_id and source_id.strip():
                keyword_query = keyword_query.eq("source_id", source_id)

            # Execute keyword search
            keyword_response = keyword_query.limit(match_count * 2).execute()
            keyword_results = keyword_response.data if keyword_response.data else []

            # 3. Combine results with preference for items appearing in both
            seen_ids = set()
            combined_results = []

            # First, add items that appear in both searches (these are the best matches)
            vector_ids = {r.get("id") for r in vector_results if r.get("id")}
            for kr in keyword_results:
                if kr["id"] in vector_ids and kr["id"] not in seen_ids:
                    # Find the vector result to get similarity score
                    for vr in vector_results:
                        if vr.get("id") == kr["id"]:
                            # Boost similarity score for items in both results
                            vr["similarity"] = min(1.0, vr.get("similarity", 0) * 1.2)
                            combined_results.append(vr)
                            seen_ids.add(kr["id"])
                            break

            # Then add remaining vector results (semantic matches without exact keyword)
            for vr in vector_results:
                if (
                    vr.get("id")
                    and vr["id"] not in seen_ids
                    and len(combined_results) < match_count
                ):
                    combined_results.append(vr)
                    seen_ids.add(vr["id"])

            # Finally, add pure keyword matches if we still need more results
            for kr in keyword_results:
                if kr["id"] not in seen_ids and len(combined_results) < match_count:
                    # Convert keyword result to match vector result format
                    combined_results.append(
                        {
                            "id": kr["id"],
                            "url": kr["url"],
                            "chunk_number": kr["chunk_number"],
                            "content": kr["content"],
                            "summary": kr["summary"],
                            "metadata": kr["metadata"],
                            "source_id": kr["source_id"],
                            "similarity": 0.5,  # Default similarity for keyword-only matches
                        }
                    )
                    seen_ids.add(kr["id"])

            # Use combined results
            results = combined_results[:match_count]

        else:
            # Standard vector search only
            from utils import search_code_examples as search_code_examples_impl

            results = search_code_examples_impl(
                client=supabase_client,
                query=query,
                match_count=match_count,
                filter_metadata=filter_metadata,
            )

        # Apply reranking if enabled
        use_reranking = os.getenv("USE_RERANKING", "false") == "true"
        if use_reranking and ctx.request_context.lifespan_context.reranking_model:
            results = rerank_results(
                ctx.request_context.lifespan_context.reranking_model,
                query,
                results,
                content_key="content",
            )

        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = {
                "url": result.get("url"),
                "code": result.get("content"),
                "summary": result.get("summary"),
                "metadata": result.get("metadata"),
                "source_id": result.get("source_id"),
                "similarity": result.get("similarity"),
            }
            # Include rerank score if available
            if "rerank_score" in result:
                formatted_result["rerank_score"] = result["rerank_score"]
            formatted_results.append(formatted_result)

        return json.dumps(
            {
                "success": True,
                "query": query,
                "source_filter": source_id,
                "search_mode": "hybrid" if use_hybrid_search else "vector",
                "reranking_applied": use_reranking
                and ctx.request_context.lifespan_context.reranking_model is not None,
                "results": formatted_results,
                "count": len(formatted_results),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)


async def crawl_markdown_file(
    crawler: AsyncWebCrawler, url: str
) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.

    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{"url": url, "markdown": result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []


async def crawl_pdf_file(url: str) -> List[Dict[str, Any]]:
    """
    Crawl a PDF file.

    Args:
        url: URL of the PDF file

    Returns:
        List of dictionaries with URL and markdown content
    """
    pdf_crawler_strategy = PDFCrawlerStrategy()
    pdf_scraping_strategy = PDFContentScrapingStrategy()
    
    async with AsyncWebCrawler(crawler_strategy=pdf_crawler_strategy) as pdf_crawler:
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            stream=False,
            scraping_strategy=pdf_scraping_strategy
        )
        
        result = await pdf_crawler.arun(url=url, config=crawl_config)
        if result.markdown:  # Check for markdown content regardless of success flag
            return [{"url": url, "markdown": result.markdown}]
        else:
            print(f"Failed to crawl PDF {url}: {result.error_message if result.error_message else 'No content extracted'}")
            return []


async def crawl_batch(
    crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel, handling both web pages and PDFs.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    # Separate PDFs from regular URLs
    pdf_urls = [url for url in urls if is_pdf(url)]
    web_urls = [url for url in urls if not is_pdf(url)]
    
    results = []
    
    # Handle regular web URLs
    if web_urls:
        crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            check_interval=1.0,
            max_session_permit=max_concurrent,
        )

        web_results = await crawler.arun_many(
            urls=web_urls, config=crawl_config, dispatcher=dispatcher
        )
        results.extend([
            {"url": r.url, "markdown": r.markdown}
            for r in web_results
            if r.success and r.markdown
        ])
    
    # Handle PDF URLs
    for pdf_url in pdf_urls:
        pdf_results = await crawl_pdf_file(pdf_url)
        results.extend(pdf_results)
    
    return results


async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler,
    start_urls: List[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [
            normalize_url(url)
            for url in current_urls
            if normalize_url(url) not in visited
        ]
        if not urls_to_crawl:
            break

        # Separate PDFs from regular URLs for this batch
        pdf_urls = [url for url in urls_to_crawl if is_pdf(url)]
        web_urls = [url for url in urls_to_crawl if not is_pdf(url)]
        
        # Process web URLs
        if web_urls:
            results = await crawler.arun_many(
                urls=web_urls, config=run_config, dispatcher=dispatcher
            )
        else:
            results = []
            
        # Process PDF URLs
        for pdf_url in pdf_urls:
            pdf_results = await crawl_pdf_file(pdf_url)
            # Convert PDF results to match arun_many format
            for pdf_result in pdf_results:
                # Create a mock result object with necessary attributes
                class MockResult:
                    def __init__(self, url, markdown, success=True):
                        self.url = url
                        self.markdown = markdown
                        self.success = success
                        self.links = {"internal": [], "external": []}  # PDFs don't have links to follow
                
                results.append(MockResult(pdf_result["url"], pdf_result["markdown"]))
        
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({"url": result.url, "markdown": result.markdown})
                # Only follow links for non-PDF results
                if hasattr(result, 'links') and result.links and not is_pdf(result.url):
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited:
                            next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all


# ============================================================================
# GitHub Repository Crawling MCP Tools
# ============================================================================

@mcp.tool()
async def analyze_github_repo(ctx: Context, github_url: str) -> str:
    """
    Analyze a GitHub repository structure using local cloning and return crawlable files information.
    
    This tool clones the repository locally, analyzes the file structure, filters out binary/data files,
    and provides a summary of crawlable content organized by file type and directory.
    
    Args:
        ctx: The MCP server provided context
        github_url: GitHub repository URL (e.g., "https://github.com/pulseq/pulseq")
        
    Returns:
        JSON string with repository analysis and crawlable files summary
    """
    temp_dir = None
    
    try:
        # Parse GitHub URL
        owner, repo = parse_github_url(github_url)
        
        # Create temporary directory and clone repository
        temp_dir = create_temp_directory()
        repo_path = clone_github_repository(owner, repo, temp_dir)
        
        # Get local file tree and skip report
        crawlable_files, skip_report = get_local_file_tree(repo_path)
        
        # Organize files by extension and directory
        files_by_extension = {}
        files_by_directory = {}
        
        for file_info in crawlable_files:
            ext = file_info["extension"]
            directory = file_info["directory"]
            
            # Group by extension
            if ext not in files_by_extension:
                files_by_extension[ext] = []
            files_by_extension[ext].append(file_info)
            
            # Group by directory
            if directory not in files_by_directory:
                files_by_directory[directory] = []
            files_by_directory[directory].append(file_info)
        
        # Create summary
        summary = {
            "repository": f"{owner}/{repo}",
            "total_crawlable_files": len(crawlable_files),
            "files_by_extension": {
                ext: {
                    "count": len(files),
                    "total_size": sum(f["size"] for f in files),
                    "examples": [f["path"] for f in files[:3]]  # Show first 3 as examples
                }
                for ext, files in files_by_extension.items()
            },
            "files_by_directory": {
                directory: {
                    "count": len(files),
                    "extensions": list(set(f["extension"] for f in files))
                }
                for directory, files in files_by_directory.items()
                if directory  # Skip root directory
            },
            "skipped_files": {
                "extensions": skip_report,
                "total_skipped": sum(skip_report.values()),
                "skipped_extensions": list(skip_report.keys())
            },
            "analysis_method": "local_clone"
        }
        
        return json.dumps(summary, indent=2)
        
    except Exception as e:
        return json.dumps({
            "error": f"Failed to analyze repository: {str(e)}",
            "repository": github_url,
            "analysis_method": "local_clone"
        })
    finally:
        # Always clean up temporary directory
        if temp_dir:
            cleanup_temp_directory(temp_dir)


@mcp.tool()
async def crawl_github_repo(
    ctx: Context, 
    github_url: str, 
    file_extensions: str = "all",
    max_files: int = 100
) -> str:
    """
    Crawl a GitHub repository using local cloning and store selected files in Supabase.
    
    This tool clones the repository locally, processes files with appropriate metadata, 
    and stores them in both crawled_pages and code_examples tables. This approach 
    handles large files that may exceed GitHub API limits.
    
    Args:
        ctx: The MCP server provided context
        github_url: GitHub repository URL (e.g., "https://github.com/pulseq/pulseq")
        file_extensions: Comma-separated list of extensions to crawl (e.g., ".m,.py,.md") or "all"
        max_files: Maximum number of files to crawl (default: 100)
        
    Returns:
        JSON string with crawling results and statistics
    """
    supabase_client: Client = ctx.request_context.lifespan_context.supabase_client
    temp_dir = None
    
    try:
        # Parse GitHub URL
        owner, repo = parse_github_url(github_url)
        
        # Create temporary directory and clone repository
        temp_dir = create_temp_directory()
        repo_path = clone_github_repository(owner, repo, temp_dir)
        
        # Get local file tree and filter crawlable files
        crawlable_files, skip_report = get_local_file_tree(repo_path)
        
        # Filter by requested extensions if not "all"
        if file_extensions.lower() != "all":
            requested_extensions = [ext.strip().lower() for ext in file_extensions.split(",")]
            crawlable_files = [
                f for f in crawlable_files 
                if f["extension"].lower() in requested_extensions
            ]
        
        # Track files skipped due to max file limit
        total_eligible_files = len(crawlable_files)
        files_skipped_by_limit = 0
        
        # Limit number of files
        if len(crawlable_files) > max_files:
            files_skipped_by_limit = len(crawlable_files) - max_files
            crawlable_files = crawlable_files[:max_files]
            
        # Initialize collections for batch processing
        doc_urls = []
        doc_chunk_numbers = []
        doc_contents = []
        doc_metadatas = []
        
        code_urls = []
        code_chunk_numbers = []
        code_examples = []
        code_summaries = []
        code_metadatas = []
        
        processed_files = 0
        errors = []
        
        # Initialize LLM failure tracking
        llm_failures = {
            "api_400_errors": [],
            "rate_limit_errors": [],
            "timeout_errors": [],
            "source_summary_errors": [],
            "other_llm_errors": []
        }
        
        # Process files from local repository
        for file_info in crawlable_files:
            try:
                # Read file content from local repository
                file_data = read_local_file_content(file_info["full_path"])
                
                if file_data.get("is_binary") or not file_data.get("decoded_content"):
                    continue
                    
                content = file_data["decoded_content"]
                
                # Create GitHub file URL for reference (use HEAD to avoid hardcoded branch)
                github_file_url = f"https://github.com/{owner}/{repo}/blob/HEAD/{file_info['path']}"
                
                # Special handling for PDF files
                if file_info["extension"] == ".pdf":
                    # Use existing PDF crawling functionality via raw GitHub URL
                    raw_file_url = f"https://github.com/{owner}/{repo}/raw/HEAD/{file_info['path']}"
                    pdf_results = await crawl_pdf_file(raw_file_url)
                    
                    if pdf_results:
                        for pdf_result in pdf_results:
                            pdf_content = pdf_result["markdown"]
                            
                            # Create PDF-specific metadata
                            pdf_metadata = {
                                "url": github_file_url,
                                "source": f"github.com/{owner}/{repo}",
                                "headers": f"PDF: {file_info['path']}",
                                "char_count": len(pdf_content),
                                "chunk_size": len(pdf_content),
                                "crawl_time": "github_local_clone",
                                "word_count": len(pdf_content.split()),
                                "chunk_index": 0,
                                "contextual_embedding": False,
                                "source_type": "github",
                                "repo_owner": owner,
                                "repo_name": repo,
                                "file_path": file_info["path"],
                                "file_extension": ".pdf",
                                "file_category": "pdf_document",
                                "commit_sha": "",  # Not available from local clone
                                "language": "pdf",
                                "content_type": "document"
                            }
                            
                            # Add to documents collection
                            doc_urls.append(github_file_url)
                            doc_chunk_numbers.append(0)
                            doc_contents.append(pdf_content)
                            doc_metadatas.append(pdf_metadata)
                    
                    processed_files += 1
                    await asyncio.sleep(0.05)
                    continue
                
                # Special handling for Jupyter notebooks
                if file_info["extension"] == ".ipynb":
                    # Process notebook as script
                    combined_code, context, notebook_metadata = process_notebook_as_script(content, file_info["path"])
                    
                    if combined_code:
                        # Use combined code as content and treat according to notebook language
                        content_to_store = combined_code
                        lang_metadata = extract_language_metadata(combined_code, f"{file_info['path']}_combined.{notebook_metadata.get('language', 'py')}")
                        
                        # Add notebook-specific metadata
                        lang_metadata.update({
                            "notebook_language": notebook_metadata.get("language", "python"),
                            "notebook_context": context,
                            "cell_count": notebook_metadata.get("cell_count", 0),
                            "code_cells": notebook_metadata.get("code_cells", 0),
                            "context_cells": notebook_metadata.get("context_cells", 0)
                        })
                    else:
                        # If no code, store the context as content
                        content_to_store = context
                        lang_metadata = {"language": "jupyter", **notebook_metadata}
                else:
                    # Regular file processing
                    content_to_store = content
                    lang_metadata = extract_language_metadata(content, file_info["path"])
                
                # Create base metadata following existing schema
                base_metadata = {
                    # Existing core fields
                    "url": github_file_url,
                    "source": f"github.com/{owner}/{repo}",
                    "headers": lang_metadata.get("function_signature", ""),
                    "char_count": len(content_to_store),
                    "chunk_size": len(content_to_store),
                    "crawl_time": "github_local_clone",
                    "word_count": len(content_to_store.split()),
                    "chunk_index": 0,
                    "contextual_embedding": False,
                    
                    # GitHub-specific extensions
                    "source_type": "github",
                    "repo_owner": owner,
                    "repo_name": repo,
                    "file_path": file_info["path"],
                    "file_extension": file_info["extension"],
                    "file_category": "repository_file",
                    "commit_sha": "",  # Not available from local clone
                    
                    # Language-specific metadata
                    **lang_metadata
                }
                
                # Add to documents collection
                doc_urls.append(github_file_url)
                doc_chunk_numbers.append(0)
                doc_contents.append(content_to_store)
                doc_metadatas.append(base_metadata.copy())
                
                # If it's a code file or notebook with code, also add to code examples
                is_code_file = (
                    file_info["extension"] in [".m", ".py", ".cpp", ".c", ".h", ".js", ".ts"] or
                    (file_info["extension"] == ".ipynb" and lang_metadata.get("language") != "jupyter")
                )
                
                if is_code_file:
                    # Generate code summary with error tracking
                    try:
                        code_summary = generate_code_example_summary(
                            content_to_store, 
                            content_to_store[:500], 
                            content_to_store[-500:],
                            file_path=file_info['path']
                        )
                    except LLMProcessingError as e:
                        # Track LLM-specific errors
                        error_dict = e.to_dict()
                        if e.error_type == "api_400":
                            llm_failures["api_400_errors"].append(error_dict)
                        elif e.error_type == "rate_limit":
                            llm_failures["rate_limit_errors"].append(error_dict)
                        elif e.error_type == "timeout":
                            llm_failures["timeout_errors"].append(error_dict)
                        else:
                            llm_failures["other_llm_errors"].append(error_dict)
                        
                        # Use fallback summary
                        code_summary = "Code example for demonstration purposes."
                    
                    # Code-specific metadata
                    code_metadata = base_metadata.copy()
                    code_metadata.update({
                        "language": lang_metadata.get("language", "unknown"),
                        "char_count": len(content_to_store),
                        "word_count": len(content_to_store.split()),
                        "chunk_index": 0
                    })
                    
                    code_urls.append(github_file_url)
                    code_chunk_numbers.append(0)
                    code_examples.append(content_to_store)
                    code_summaries.append(code_summary)
                    code_metadatas.append(code_metadata)
                
                processed_files += 1
                
                # Add brief delay between file processing to reduce system load
                await asyncio.sleep(0.05)
                
            except Exception as e:
                errors.append(f"Error processing {file_info['path']}: {str(e)}")
                continue
        
        # Update source info FIRST (before inserting documents with foreign key references)
        source_id = f"github.com/{owner}/{repo}"
        total_content = " ".join(doc_contents)
        
        # Generate source summary with error tracking
        try:
            # Use parallel processing for source summary generation (matching smart_crawl_url pattern)
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                source_summary_args = [(source_id, total_content)]
                source_summaries = list(
                    executor.map(
                        lambda args: extract_source_summary(args[0], args[1]),
                        source_summary_args,
                    )
                )
            
            source_summary = source_summaries[0]
        except LLMProcessingError as e:
            # Track source summary LLM errors
            error_dict = e.to_dict()
            llm_failures["source_summary_errors"].append(error_dict)
            
            # Use fallback summary
            source_summary = f"Content from {source_id}"
        except Exception as e:
            # Track other source summary errors
            error_dict = {
                "file": source_id,
                "error": f"Source summary generation failed: {str(e)}",
                "error_type": "source_summary_unknown",
                "size": len(total_content),
                "operation": "source_summary"
            }
            llm_failures["source_summary_errors"].append(error_dict)
            
            # Use fallback summary
            source_summary = f"Content from {source_id}"
        
        update_source_info(supabase_client, source_id, source_summary, len(total_content.split()))
        
        # Store in Supabase AFTER source exists
        batch_size = 10
        if doc_urls:
            await add_documents_to_supabase(
                supabase_client,
                doc_urls,
                doc_chunk_numbers, 
                doc_contents,
                doc_metadatas,
                {url: content for url, content in zip(doc_urls, doc_contents)},
                batch_size=batch_size
            )
            
        if code_urls:
            await add_code_examples_to_supabase(
                supabase_client,
                code_urls,
                code_chunk_numbers,
                code_examples,
                code_summaries,
                code_metadatas,
                batch_size=batch_size
            )
        
        # Prepare result with LLM failure tracking
        result = {
            "success": True,
            "repository": f"{owner}/{repo}",
            "total_eligible_files": total_eligible_files,
            "processed_files": processed_files,
            "documents_stored": len(doc_urls),
            "code_examples_stored": len(code_urls),
            "files_by_extension": {},
            "skipped_files": {
                "by_extension": skip_report,
                "by_extension_total": sum(skip_report.values()),
                "by_file_limit": files_skipped_by_limit,
                "total_skipped": sum(skip_report.values()) + files_skipped_by_limit
            },
            "errors": errors,
            "crawl_method": "local_clone"
        }
        
        # Add LLM failure information if any failures occurred
        total_llm_failures = sum(len(failures) for failures in llm_failures.values())
        if total_llm_failures > 0:
            result["llm_issues"] = {
                "total_affected_files": total_llm_failures,
                "api_400_errors": llm_failures["api_400_errors"],
                "rate_limit_errors": llm_failures["rate_limit_errors"],
                "timeout_errors": llm_failures["timeout_errors"],
                "source_summary_errors": llm_failures["source_summary_errors"],
                "other_llm_errors": llm_failures["other_llm_errors"],
                "summary": {
                    "api_400_count": len(llm_failures["api_400_errors"]),
                    "rate_limit_count": len(llm_failures["rate_limit_errors"]),
                    "timeout_count": len(llm_failures["timeout_errors"]),
                    "source_summary_count": len(llm_failures["source_summary_errors"]),
                    "other_count": len(llm_failures["other_llm_errors"])
                }
            }
        
        # Add extension breakdown
        for file_info in crawlable_files[:processed_files]:
            ext = file_info["extension"]
            if ext not in result["files_by_extension"]:
                result["files_by_extension"][ext] = 0
            result["files_by_extension"][ext] += 1
            
        return json.dumps(result, indent=2)
        
    except subprocess.CalledProcessError as e:
        return json.dumps({
            "error": f"Git operation failed: {str(e)}",
            "error_type": "git_error",
            "repository": github_url,
            "crawl_method": "local_clone"
        })
    except subprocess.TimeoutExpired as e:
        return json.dumps({
            "error": f"Git operation timed out: {str(e)}",
            "error_type": "timeout_error", 
            "repository": github_url,
            "crawl_method": "local_clone"
        })
    except PermissionError as e:
        return json.dumps({
            "error": f"Permission denied during file operations: {str(e)}",
            "error_type": "permission_error",
            "repository": github_url,
            "crawl_method": "local_clone"
        })
    except ValueError as e:
        return json.dumps({
            "error": f"Invalid input parameters: {str(e)}",
            "error_type": "validation_error",
            "repository": github_url,
            "crawl_method": "local_clone"
        })
    except Exception as e:
        return json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "error_type": "unknown_error",
            "repository": github_url,
            "crawl_method": "local_clone"
        })
    finally:
        # Always clean up temporary directory
        if temp_dir:
            cleanup_temp_directory(temp_dir)


async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == "sse":
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())
