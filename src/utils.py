"""
Utility functions for the Crawl4AI MCP server.
"""

import os
import ast
import concurrent.futures
import asyncio
import gc
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from supabase import create_client, Client
from urllib.parse import urlparse
import requests
import base64
import re

# Import local embeddings (optional - used for BGE embeddings)
try:
    from local_embeddings import create_embedding, create_embeddings_batch

    LOCAL_EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    print(f"Local embeddings not available: {e}")
    LOCAL_EMBEDDINGS_AVAILABLE = False

    # Create safe fallback functions that return empty vectors


# ============================================================================
# Custom Exception Classes for LLM Error Tracking
# ============================================================================


class LLMProcessingError(Exception):
    """Exception raised when LLM processing fails with specific error details."""

    def __init__(
        self,
        message: str,
        file_path: str = None,
        error_type: str = "unknown",
        file_size: int = None,
        **kwargs,
    ):
        super().__init__(message)
        self.message = message
        self.file_path = file_path
        self.error_type = error_type  # "api_400", "timeout", "content_filter", etc.
        self.file_size = file_size
        self.additional_info = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for reporting."""
        return {
            "file": self.file_path,
            "error": self.message,
            "error_type": self.error_type,
            "size": self.file_size,
            **self.additional_info,
        }

    def create_embedding(text):
        print("Warning: Local embeddings not available, returning empty vector")
        return [0.0] * 1024  # BGE model returns 1024-dimensional vectors

    def create_embeddings_batch(texts):
        print("Warning: Local embeddings not available, returning empty vectors")
        return [[0.0] * 1024 for _ in texts]


# Import Google AI API service
try:
    try:
        from .google_ai_api import (
            generate_text as generate_google_ai_text,
            is_available as google_ai_available,
        )
    except ImportError:
        # Fallback for direct execution
        from google_ai_api import (
            generate_text as generate_google_ai_text,
            is_available as google_ai_available,
        )
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False


def generate_text_with_fallback(
    prompt: str, max_tokens: int = 200, temperature: float = 0.3
) -> str:
    """
    Generate text using Google AI API.

    Args:
        prompt: The text prompt
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated text response
    """
    if GOOGLE_AI_AVAILABLE and google_ai_available():
        try:
            response = generate_google_ai_text(
                prompt=prompt, max_output_tokens=max_tokens, temperature=temperature
            )
            return response.strip()
        except Exception as e:
            print(f"Google AI API generation failed: {str(e)}")
            return "Text generation unavailable"

    print("Google AI API is not available or not configured properly")
    return "Text generation unavailable"


def get_supabase_client() -> Client:
    """
    Get a Supabase client with the URL and key from environment variables.
    Uses DEV or PROD environment based on USE_DEV_SUPABASE flag.

    Returns:
        Supabase client instance
    """
    # Check if we should use dev environment (default to True for testing)
    use_dev = os.getenv("USE_DEV_SUPABASE", "true").lower() == "true"

    if use_dev:
        # DEV environment
        url = os.getenv("SUPABASE_DEV_URL")
        key = os.getenv("SUPABASE_DEV_SERVICE_KEY")

        if not url or not key:
            raise ValueError(
                "SUPABASE_DEV_URL and SUPABASE_DEV_SERVICE_KEY must be set in environment variables"
            )
    else:
        # PRODUCTION environment
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables"
            )

    return create_client(url, key)


def generate_contextual_embedding(full_document: str, chunk: str) -> Tuple[str, bool]:
    """
    Generate contextual information for a chunk within a document to improve retrieval.

    Args:
        full_document: The complete document text
        chunk: The specific chunk of text to generate context for

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    try:
        # Create the prompt for generating contextual information
        prompt = f"""Given this document:

{full_document[:25000]}

Please provide a short, succinct context (1-2 sentences) to situate this specific chunk within the overall document for search retrieval purposes:

"{chunk}"

Context:"""

        # Use text generation with fallback
        context = generate_text_with_fallback(
            prompt=prompt, max_tokens=200, temperature=0.3
        )

        # Clean up the response
        if context:
            # Combine the context with the original chunk
            contextual_text = f"{context}\n---\n{chunk}"
            return contextual_text, True
        else:
            return chunk, False

    except Exception as e:
        print(
            f"Error generating contextual embedding: {e}. Using original chunk instead."
        )
        return chunk, False


def process_chunk_with_context(args):
    """
    Process a single chunk with contextual embedding.
    This function is designed to be used with concurrent.futures.

    Args:
        args: Tuple containing (url, content, full_document)

    Returns:
        Tuple containing:
        - The contextual text that situates the chunk within the document
        - Boolean indicating if contextual embedding was performed
    """
    url, content, full_document = args
    return generate_contextual_embedding(full_document, content)


async def add_documents_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 10,
) -> None:
    """
    Add documents to the Supabase crawled_pages table in batches.
    Deletes existing records with the same URLs before inserting to prevent duplicates.

    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        contents: List of document contents
        metadatas: List of document metadata
        url_to_full_document: Dictionary mapping URLs to their full document content
        batch_size: Size of each batch for insertion
    """
    # Get unique URLs to delete existing records
    unique_urls = list(set(urls))

    # Delete existing records for these URLs in a single operation
    try:
        if unique_urls:
            # Use the .in_() filter to delete all records with matching URLs
            client.table("crawled_pages").delete().in_("url", unique_urls).execute()
    except Exception as e:
        print(f"Batch delete failed: {e}. Trying one-by-one deletion as fallback.")
        # Fallback: delete records one by one
        for url in unique_urls:
            try:
                client.table("crawled_pages").delete().eq("url", url).execute()
            except Exception as inner_e:
                print(f"Error deleting record for URL {url}: {inner_e}")
                # Continue with the next URL even if one fails

    # Check if MODEL_CHOICE is set for contextual embeddings
    use_contextual_embeddings = (
        os.getenv("USE_CONTEXTUAL_EMBEDDINGS", "false") == "true"
    )
    print(f"\n\nUse contextual embeddings: {use_contextual_embeddings}\n\n")

    # Process in batches to avoid memory issues
    for i in range(0, len(contents), batch_size):
        batch_end = min(i + batch_size, len(contents))

        # Get batch slices
        batch_urls = urls[i:batch_end]
        batch_chunk_numbers = chunk_numbers[i:batch_end]
        batch_contents = contents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]

        # Apply contextual embedding to each chunk if MODEL_CHOICE is set
        if use_contextual_embeddings:
            # Prepare arguments for parallel processing
            process_args = []
            for j, content in enumerate(batch_contents):
                url = batch_urls[j]
                full_document = url_to_full_document.get(url, "")
                process_args.append((url, content, full_document))

            # Process in parallel using ThreadPoolExecutor
            contextual_contents = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                # Submit all tasks and collect results
                future_to_idx = {
                    executor.submit(process_chunk_with_context, arg): idx
                    for idx, arg in enumerate(process_args)
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result, success = future.result()
                        contextual_contents.append(result)
                        if success:
                            batch_metadatas[idx]["contextual_embedding"] = True
                    except Exception as e:
                        print(f"Error processing chunk {idx}: {e}")
                        # Use original content as fallback
                        contextual_contents.append(batch_contents[idx])

            # Sort results back into original order if needed
            if len(contextual_contents) != len(batch_contents):
                print(
                    f"Warning: Expected {len(batch_contents)} results but got {len(contextual_contents)}"
                )
                # Use original contents as fallback
                contextual_contents = batch_contents
        else:
            # If not using contextual embeddings, use original contents
            contextual_contents = batch_contents

        # Create embeddings for the entire batch at once
        batch_embeddings = create_embeddings_batch(contextual_contents)

        batch_data = []
        for j in range(len(contextual_contents)):
            # Extract metadata fields
            chunk_size = len(contextual_contents[j])

            # Extract source_id: use metadata source for GitHub repos, URL parsing for others
            metadata_source = batch_metadatas[j].get("source")
            if (
                metadata_source
                and "/" in metadata_source
                and metadata_source.startswith("github.com/")
            ):
                # GitHub repository format: github.com/owner/repo
                source_id = metadata_source
            else:
                # Regular webpage: extract from URL
                parsed_url = urlparse(batch_urls[j])
                source_id = parsed_url.netloc or parsed_url.path

            # Prepare data for insertion
            data = {
                "url": batch_urls[j],
                "chunk_number": batch_chunk_numbers[j],
                "content": contextual_contents[j],  # Store original content
                "metadata": {"chunk_size": chunk_size, **batch_metadatas[j]},
                "source_id": source_id,  # Add source_id field
                "embedding": batch_embeddings[
                    j
                ],  # Use embedding from contextual content
            }

            batch_data.append(data)

        # Insert batch into Supabase with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay

        for retry in range(max_retries):
            try:
                client.table("crawled_pages").insert(batch_data).execute()
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(
                        f"Error inserting batch into Supabase (attempt {retry + 1}/{max_retries}): {e}"
                    )
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.table("crawled_pages").insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(
                                f"Failed to insert individual record for URL {record['url']}: {individual_error}"
                            )

                    if successful_inserts > 0:
                        print(
                            f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually"
                        )

        # Add brief delay between batches to reduce system load
        await asyncio.sleep(0.1)

        # Trigger garbage collection after each batch to manage memory
        gc.collect()


def search_documents(
    client: Client,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for documents in Supabase using vector similarity.

    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter

    Returns:
        List of matching documents
    """
    # Create embedding for the query
    query_embedding = create_embedding(query)

    # Execute the search using the match_crawled_pages function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {"query_embedding": query_embedding, "match_count": match_count}

        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params["filter"] = (
                filter_metadata  # Pass the dictionary directly, not JSON-encoded
            )

        result = client.rpc("match_crawled_pages", params).execute()

        return result.data
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []


def extract_code_blocks(
    markdown_content: str, min_length: int = 1000
) -> List[Dict[str, Any]]:
    """
    Extract code blocks from markdown content along with context.

    Args:
        markdown_content: The markdown content to extract code blocks from
        min_length: Minimum length of code blocks to extract (default: 1000 characters)

    Returns:
        List of dictionaries containing code blocks and their context
    """
    code_blocks = []

    # Skip if content starts with triple backticks (edge case for files wrapped in backticks)
    content = markdown_content.strip()
    start_offset = 0
    if content.startswith("```"):
        # Skip the first triple backticks
        start_offset = 3
        print("Skipping initial triple backticks")

    # Find all occurrences of triple backticks
    backtick_positions = []
    pos = start_offset
    while True:
        pos = markdown_content.find("```", pos)
        if pos == -1:
            break
        backtick_positions.append(pos)
        pos += 3

    # Process pairs of backticks
    i = 0
    while i < len(backtick_positions) - 1:
        start_pos = backtick_positions[i]
        end_pos = backtick_positions[i + 1]

        # Extract the content between backticks
        code_section = markdown_content[start_pos + 3 : end_pos]

        # Check if there's a language specifier on the first line
        lines = code_section.split("\n", 1)
        if len(lines) > 1:
            # Check if first line is a language specifier (no spaces, common language names)
            first_line = lines[0].strip()
            if first_line and " " not in first_line and len(first_line) < 20:
                language = first_line
                code_content = lines[1].strip() if len(lines) > 1 else ""
            else:
                language = ""
                code_content = code_section.strip()
        else:
            language = ""
            code_content = code_section.strip()

        # Skip if code block is too short
        if len(code_content) < min_length:
            i += 2  # Move to next pair
            continue

        # Extract context before (1000 chars)
        context_start = max(0, start_pos - 1000)
        context_before = markdown_content[context_start:start_pos].strip()

        # Extract context after (1000 chars)
        context_end = min(len(markdown_content), end_pos + 3 + 1000)
        context_after = markdown_content[end_pos + 3 : context_end].strip()

        code_blocks.append(
            {
                "code": code_content,
                "language": language,
                "context_before": context_before,
                "context_after": context_after,
                "full_context": f"{context_before}\n\n{code_content}\n\n{context_after}",
            }
        )

        # Move to next pair (skip the closing backtick we just processed)
        i += 2

    return code_blocks


def analyze_content_distribution(markdown_content: str) -> Dict[str, float]:
    """
    Analyze the distribution of code vs text content in markdown.

    Args:
        markdown_content: The markdown content to analyze

    Returns:
        Dictionary with code percentage and character counts
    """
    # Extract all code blocks without minimum length filter
    code_blocks = extract_code_blocks(markdown_content, min_length=0)

    total_code_chars = sum(len(block["code"]) for block in code_blocks)
    total_content_chars = len(markdown_content)
    text_chars = total_content_chars - total_code_chars

    code_percentage = (
        total_code_chars / total_content_chars if total_content_chars > 0 else 0
    )

    return {
        "code_percentage": code_percentage,
        "total_code_chars": total_code_chars,
        "total_text_chars": text_chars,
        "total_content_chars": total_content_chars,
    }


def detect_primary_language(code_blocks: List[Dict[str, Any]]) -> str:
    """
    Detect the dominant language across all code blocks based on character count.

    Args:
        code_blocks: List of code block dictionaries

    Returns:
        The dominant language or empty string if none detected
    """
    language_chars = {}

    for block in code_blocks:
        lang = block.get("language", "").lower().strip()
        if lang:
            char_count = len(block["code"])
            language_chars[lang] = language_chars.get(lang, 0) + char_count

    if not language_chars:
        return ""

    # Return language with most characters
    return max(language_chars, key=language_chars.get)


def create_combined_code_block(
    markdown_content: str, code_blocks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Combine all code blocks into one seamless block with surrounding text as context.

    Args:
        markdown_content: Original markdown content
        code_blocks: List of individual code blocks

    Returns:
        Dictionary representing the combined code block
    """
    # Combine all code with simple spacing
    combined_code = "\n\n".join(block["code"] for block in code_blocks)

    # Extract all non-code text as context
    text_context = extract_text_between_code_blocks(markdown_content, code_blocks)

    # Detect primary language
    primary_language = detect_primary_language(code_blocks)

    return {
        "code": combined_code,
        "language": primary_language,
        "context_before": text_context,
        "context_after": "",
        "full_context": f"{text_context}\n\n{combined_code}",
    }


def extract_text_between_code_blocks(
    markdown_content: str, code_blocks: List[Dict[str, Any]]
) -> str:
    """
    Extract all text content that is not within code blocks.

    Args:
        markdown_content: The original markdown content
        code_blocks: List of code blocks to exclude

    Returns:
        Concatenated text content outside of code blocks
    """
    if not code_blocks:
        return markdown_content.strip()

    # Find all code block positions in the original content
    code_positions = []

    # Re-find code block positions in original content
    pos = 0
    while True:
        pos = markdown_content.find("```", pos)
        if pos == -1:
            break

        # Find the closing backticks
        end_pos = markdown_content.find("```", pos + 3)
        if end_pos == -1:
            break

        code_positions.append((pos, end_pos + 3))
        pos = end_pos + 3

    # Extract text between code blocks
    text_segments = []
    last_end = 0

    for start_pos, end_pos in code_positions:
        # Add text before this code block
        if start_pos > last_end:
            text_segment = markdown_content[last_end:start_pos].strip()
            if text_segment:
                text_segments.append(text_segment)
        last_end = end_pos

    # Add any remaining text after the last code block
    if last_end < len(markdown_content):
        remaining_text = markdown_content[last_end:].strip()
        if remaining_text:
            text_segments.append(remaining_text)

    return "\n\n".join(text_segments)


def generate_code_example_summary(
    code: str,
    context_before: str,
    context_after: str,
    is_code_dominated: bool = False,
    file_path: str = None,
) -> str:
    """
    Generate a summary for a code example using its surrounding context.

    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        is_code_dominated: Whether this is code-dominated content (increases context limit)
        file_path: Optional file path for error tracking

    Returns:
        A summary of what the code example demonstrates

    Raises:
        LLMProcessingError: When LLM processing fails with detailed error information
    """
    # Dynamic context limits based on content type
    context_limit = 2000 if is_code_dominated else 500

    # Truncate context_before based on content type
    context_before_truncated = (
        context_before[-context_limit:]
        if len(context_before) > context_limit
        else context_before
    )

    # For code-dominated content, don't truncate code; for regular content, apply existing limit
    code_content = (
        code if is_code_dominated else (code[:1500] if len(code) > 1500 else code)
    )

    # Create the prompt with appropriate instructions for content type
    if is_code_dominated:
        instruction = "Based on the code-dominated context and complete code tutorial, provide a comprehensive summary (3-4 sentences) that describes the overall goal, key concepts demonstrated, and practical application. Focus on the learning objectives rather than individual code snippets."
    else:
        instruction = "Based on the code example and its surrounding context, provide a concise summary (2-3 sentences) that describes what this code example demonstrates and its purpose. Focus on the practical application and key concepts illustrated."

    prompt = f"""<context_before>
{context_before_truncated}
</context_before>

<code_example>
{code_content}
</code_example>

<context_after>
{context_after[:500] if len(context_after) > 500 else context_after}
</context_after>

{instruction}
"""

    try:
        # Use Google AI API to generate code example summary
        response = generate_text_with_fallback(
            prompt=prompt, max_tokens=100, temperature=0.3
        )

        return response.strip()

    except requests.exceptions.HTTPError as e:
        # Specific handling for HTTP errors from Google AI API
        if e.response.status_code == 400:
            error_msg = "Google AI API 400 error - likely content safety filter or invalid request"
            print(
                f"Error generating code example summary for {file_path or 'unknown file'}: {error_msg}"
            )
            raise LLMProcessingError(
                error_msg,
                file_path=file_path,
                error_type="api_400",
                file_size=len(code),
                extension=file_path.split(".")[-1]
                if file_path and "." in file_path
                else "unknown",
            )
        elif e.response.status_code == 429:
            error_msg = "Google AI API rate limit exceeded"
            print(
                f"Error generating code example summary for {file_path or 'unknown file'}: {error_msg}"
            )
            raise LLMProcessingError(
                error_msg,
                file_path=file_path,
                error_type="rate_limit",
                file_size=len(code),
            )
        else:
            error_msg = f"Google AI API HTTP {e.response.status_code} error"
            print(
                f"Error generating code example summary for {file_path or 'unknown file'}: {error_msg}"
            )
            raise LLMProcessingError(
                error_msg,
                file_path=file_path,
                error_type=f"api_{e.response.status_code}",
                file_size=len(code),
            )
    except requests.exceptions.Timeout:
        error_msg = "Google AI API timeout"
        print(
            f"Error generating code example summary for {file_path or 'unknown file'}: {error_msg}"
        )
        raise LLMProcessingError(
            error_msg, file_path=file_path, error_type="timeout", file_size=len(code)
        )
    except Exception as e:
        error_msg = f"Unexpected error during code summary generation: {str(e)}"
        print(
            f"Error generating code example summary for {file_path or 'unknown file'}: {error_msg}"
        )
        raise LLMProcessingError(
            error_msg, file_path=file_path, error_type="unknown", file_size=len(code)
        )


async def add_code_examples_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 10,
):
    """
    Add code examples to the Supabase code_examples table in batches.

    Args:
        client: Supabase client
        urls: List of URLs
        chunk_numbers: List of chunk numbers
        code_examples: List of code example contents
        summaries: List of code example summaries
        metadatas: List of metadata dictionaries
        batch_size: Size of each batch for insertion
    """
    if not urls:
        return

    # Delete existing records for these URLs
    unique_urls = list(set(urls))
    for url in unique_urls:
        try:
            client.table("code_examples").delete().eq("url", url).execute()
        except Exception as e:
            print(f"Error deleting existing code examples for {url}: {e}")

    # Process in batches
    total_items = len(urls)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_texts = []

        # Create combined texts for embedding (code + summary)
        for j in range(i, batch_end):
            combined_text = f"{code_examples[j]}\n\nSummary: {summaries[j]}"
            batch_texts.append(combined_text)

        # Create embeddings for the batch
        embeddings = create_embeddings_batch(batch_texts)

        # Check if embeddings are valid (not all zeros)
        valid_embeddings = []
        for embedding in embeddings:
            if embedding and not all(v == 0.0 for v in embedding):
                valid_embeddings.append(embedding)
            else:
                print(
                    "Warning: Zero or invalid embedding detected, creating new one..."
                )
                # Try to create a single embedding as fallback
                single_embedding = create_embedding(batch_texts[len(valid_embeddings)])
                valid_embeddings.append(single_embedding)

        # Prepare batch data
        batch_data = []
        for j, embedding in enumerate(valid_embeddings):
            idx = i + j

            # Extract source_id: use metadata source for GitHub repos, URL parsing for others
            metadata_source = metadatas[idx].get("source")
            if (
                metadata_source
                and "/" in metadata_source
                and metadata_source.startswith("github.com/")
            ):
                # GitHub repository format: github.com/owner/repo
                source_id = metadata_source
            else:
                # Regular webpage: extract from URL
                parsed_url = urlparse(urls[idx])
                source_id = parsed_url.netloc or parsed_url.path

            batch_data.append(
                {
                    "url": urls[idx],
                    "chunk_number": chunk_numbers[idx],
                    "content": code_examples[idx],
                    "summary": summaries[idx],
                    "metadata": metadatas[idx],  # Store as JSON object, not string (includes dependencies)
                    "source_id": source_id,
                    "embedding": embedding,
                }
            )

        # Insert batch into Supabase with retry logic
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay

        for retry in range(max_retries):
            try:
                client.table("code_examples").insert(batch_data).execute()
                # Success - break out of retry loop
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(
                        f"Error inserting batch into Supabase (attempt {retry + 1}/{max_retries}): {e}"
                    )
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    # Final attempt failed
                    print(f"Failed to insert batch after {max_retries} attempts: {e}")
                    # Optionally, try inserting records one by one as a last resort
                    print("Attempting to insert records individually...")
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            client.table("code_examples").insert(record).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(
                                f"Failed to insert individual record for URL {record['url']}: {individual_error}"
                            )

                    if successful_inserts > 0:
                        print(
                            f"Successfully inserted {successful_inserts}/{len(batch_data)} records individually"
                        )
        print(
            f"Inserted batch {i // batch_size + 1} of {(total_items + batch_size - 1) // batch_size} code examples"
        )

        # Add brief delay between batches to reduce system load
        await asyncio.sleep(0.1)

        # Trigger garbage collection after each batch to manage memory
        gc.collect()


async def add_api_references_to_supabase(
    client: Client,
    names: List[str],
    languages: List[str],
    signatures: List[str],
    descriptions: List[str],
    parameters: List[Dict[str, Any]],
    returns: List[str],
    source_ids: List[str],
    pulseq_versions: List[str],
    batch_size: int = 10,
):
    """
    Add API references to the Supabase api_reference table in batches.

    Args:
        client: Supabase client
        names: List of function/class names
        languages: List of programming languages
        signatures: List of function signatures
        descriptions: List of descriptions/docstrings
        parameters: List of parameter dictionaries
        returns: List of return descriptions
        source_ids: List of source identifiers
        pulseq_versions: List of Pulseq versions
        batch_size: Size of each batch for insertion
    """
    if not names:
        return

    # Process in batches
    total_items = len(names)
    for i in range(0, total_items, batch_size):
        batch_end = min(i + batch_size, total_items)
        batch_texts = []

        # Create texts for embedding (name + description)
        for j in range(i, batch_end):
            # Combine name, signature, and description for embedding
            embedding_text = f"{names[j]}\n{signatures[j]}\n{descriptions[j]}"
            batch_texts.append(embedding_text)

        # Create embeddings for the batch
        embeddings = create_embeddings_batch(batch_texts)

        # Prepare batch data
        batch_data = []
        for j, embedding in enumerate(embeddings):
            idx = i + j

            # Prepare data for insertion
            data = {
                "name": names[idx],
                "language": languages[idx],
                "signature": signatures[idx],
                "description": descriptions[idx],
                "parameters": parameters[idx],  # Already a JSON object
                "returns": returns[idx],
                "source_id": source_ids[idx],
                "pulseq_version": pulseq_versions[idx],
                "embedding": embedding,
            }

            batch_data.append(data)

        # Deduplicate within the batch to avoid constraint violations
        # Use memory-efficient deduplication (keep last occurrence)
        seen_keys = set()
        deduplicated_batch = []
        for data in batch_data:
            key = (data["name"], data["language"], data["pulseq_version"])
            if key not in seen_keys:
                seen_keys.add(key)
                deduplicated_batch.append(data)
        
        batch_data = deduplicated_batch
        print(f"Deduplicated batch from {len(batch_texts)} to {len(batch_data)} unique API references")

        # Use UPSERT strategy to avoid race conditions
        # Note: Supabase supports PostgreSQL's ON CONFLICT functionality
        max_retries = 3
        retry_delay = 1.0

        for retry in range(max_retries):
            try:
                # Use UPSERT to handle conflicts automatically
                # This will insert new records or update existing ones based on the unique constraint
                client.table("api_reference").upsert(
                    batch_data,
                    on_conflict="name,language,pulseq_version"
                ).execute()
                print(f"Successfully upserted {len(batch_data)} API references")
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"Error upserting API references (attempt {retry + 1}/{max_retries}): {e}")
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(
                        f"Failed to upsert API references after {max_retries} attempts: {e}"
                    )
                    # Try upserting records individually as last resort
                    successful_inserts = 0
                    for record in batch_data:
                        try:
                            # Use individual UPSERT as fallback
                            client.table("api_reference").upsert(
                                [record],
                                on_conflict="name,language,pulseq_version"
                            ).execute()
                            successful_inserts += 1
                        except Exception as individual_error:
                            print(
                                f"Failed to upsert API reference {record['name']}: {individual_error}"
                            )

                    if successful_inserts > 0:
                        print(
                            f"Successfully upserted {successful_inserts}/{len(batch_data)} API references individually"
                        )

        print(
            f"Processed batch {i // batch_size + 1} of {(total_items + batch_size - 1) // batch_size} API references"
        )

        # Add brief delay between batches
        await asyncio.sleep(0.1)

        # Trigger garbage collection
        gc.collect()


def update_source_info(client: Client, source_id: str, summary: str, word_count: int):
    """
    Update or insert source information in the sources table.

    Args:
        client: Supabase client
        source_id: The source ID (domain)
        summary: Summary of the source
        word_count: Total word count for the source
    """
    try:
        # Try to update existing source
        result = (
            client.table("sources")
            .update(
                {
                    "summary": summary,
                    "total_word_count": word_count,
                    "updated_at": "now()",
                }
            )
            .eq("source_id", source_id)
            .execute()
        )

        # If no rows were updated, insert new source
        if not result.data:
            client.table("sources").insert(
                {
                    "source_id": source_id,
                    "summary": summary,
                    "total_word_count": word_count,
                }
            ).execute()
            print(f"Created new source: {source_id}")
        else:
            print(f"Updated source: {source_id}")

    except Exception as e:
        print(f"Error updating source {source_id}: {e}")


def extract_source_summary(source_id: str, content: str, max_length: int = 500) -> str:
    """
    Extract a summary for a source from its content using an LLM.

    This function uses Google AI API to generate a concise summary of the source content.

    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary

    Returns:
        A summary string

    Raises:
        LLMProcessingError: When LLM processing fails with detailed error information
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"

    if not content or len(content.strip()) == 0:
        return default_summary

    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content

    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""

    try:
        # Use Google AI API to generate the summary
        response = generate_text_with_fallback(
            prompt=prompt, max_tokens=150, temperature=0.3
        )

        # Extract the generated summary
        summary = response.strip()

        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary

    except requests.exceptions.HTTPError as e:
        # Specific handling for HTTP errors from Google AI API
        if e.response.status_code == 400:
            error_msg = "Google AI API 400 error during source summary generation"
            print(f"Error generating source summary for {source_id}: {error_msg}")
            raise LLMProcessingError(
                error_msg,
                file_path=source_id,
                error_type="api_400",
                file_size=len(content),
                operation="source_summary",
            )
        elif e.response.status_code == 429:
            error_msg = "Google AI API rate limit exceeded during source summary"
            print(f"Error generating source summary for {source_id}: {error_msg}")
            raise LLMProcessingError(
                error_msg,
                file_path=source_id,
                error_type="rate_limit",
                file_size=len(content),
                operation="source_summary",
            )
        else:
            error_msg = f"Google AI API HTTP {e.response.status_code} error during source summary"
            print(f"Error generating source summary for {source_id}: {error_msg}")
            raise LLMProcessingError(
                error_msg,
                file_path=source_id,
                error_type=f"api_{e.response.status_code}",
                file_size=len(content),
                operation="source_summary",
            )
    except requests.exceptions.Timeout:
        error_msg = "Google AI API timeout during source summary generation"
        print(f"Error generating source summary for {source_id}: {error_msg}")
        raise LLMProcessingError(
            error_msg,
            file_path=source_id,
            error_type="timeout",
            file_size=len(content),
            operation="source_summary",
        )
    except Exception as e:
        error_msg = f"Unexpected error during source summary generation: {str(e)}"
        print(f"Error generating source summary for {source_id}: {error_msg}")
        raise LLMProcessingError(
            error_msg,
            file_path=source_id,
            error_type="unknown",
            file_size=len(content),
            operation="source_summary",
        )


def search_code_examples(
    client: Client,
    query: str,
    match_count: int = 10,
    filter_metadata: Optional[Dict[str, Any]] = None,
    source_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for code examples in Supabase using vector similarity.

    Args:
        client: Supabase client
        query: Query text
        match_count: Maximum number of results to return
        filter_metadata: Optional metadata filter
        source_id: Optional source ID to filter results

    Returns:
        List of matching code examples
    """
    # Create a more descriptive query for better embedding match
    # Since code examples are embedded with their summaries, we should make the query more descriptive
    enhanced_query = (
        f"Code example for {query}\n\nSummary: Example code showing {query}"
    )

    # Create embedding for the enhanced query
    query_embedding = create_embedding(enhanced_query)

    # Execute the search using the match_code_examples function
    try:
        # Only include filter parameter if filter_metadata is provided and not empty
        params = {"query_embedding": query_embedding, "match_count": match_count}

        # Only add the filter if it's actually provided and not empty
        if filter_metadata:
            params["filter"] = filter_metadata

        # Add source filter if provided
        if source_id:
            params["source_filter"] = source_id

        result = client.rpc("match_code_examples", params).execute()

        return result.data
    except Exception as e:
        print(f"Error searching code examples: {e}")
        return []


# ============================================================================
# Pre-compiled Regex Patterns for Performance and Security
# ============================================================================

# MATLAB patterns - compiled once for performance
MATLAB_FUNCTION_PATTERN = re.compile(
    r"function\s+(?:\[?[^\]]*\]?\s*=\s*)?(\w+)\s*\([^)]*\)", re.MULTILINE
)
MATLAB_FUNC_CALL_PATTERN = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")
MATLAB_METHOD_CALL_PATTERN = re.compile(r"\.([a-zA-Z_]\w*)\s*\(")
MATLAB_PULSEQ_PATTERNS = [
    re.compile(r"\b(make[A-Z]\w*)\s*\(", re.IGNORECASE),
    re.compile(r"\b(calc[A-Z]\w*)\s*\(", re.IGNORECASE),
    re.compile(r"\b(write[A-Z]\w*)\s*\(", re.IGNORECASE),
    re.compile(r"\b(read[A-Z]\w*)\s*\(", re.IGNORECASE),
    re.compile(r"\b(add[A-Z]\w*)\s*\(", re.IGNORECASE),
    re.compile(r"\b(set[A-Z]\w*)\s*\(", re.IGNORECASE),
    re.compile(r"\b(get[A-Z]\w*)\s*\(", re.IGNORECASE),
]

# C++ patterns - compiled once for performance, ReDoS-safe
CPP_CLASS_PATTERN = re.compile(r"class\s+(\w+)")
CPP_FUNCTION_PATTERN = re.compile(r"(?:^|\s)(\w+)\s*\([^)]*\)\s*{", re.MULTILINE)
CPP_INCLUDE_PATTERN = re.compile(r"#include\s+[<\"](.*)[>\"]")
CPP_FUNC_CALL_PATTERN = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")
CPP_METHOD_CALL_PATTERN = re.compile(r"(?:\.|->)([a-zA-Z_]\w*)\s*\(")
CPP_SCOPE_CALL_PATTERN = re.compile(r"\b([a-zA-Z_]\w*)::\s*([a-zA-Z_]\w*)\s*\(")
CPP_CLASS_INSTANTIATION_PATTERN = re.compile(r"\b([A-Z]\w*)\s*\(")
# SAFE template pattern - limits nesting depth to prevent ReDoS
CPP_TEMPLATE_PATTERN = re.compile(r"\b([a-zA-Z_]\w*)\s*<[^<>]{0,200}>\s*\(")

# Python patterns - compiled once for performance, with improved accuracy
PYTHON_CLASS_PATTERN = re.compile(r"^class\s+(\w+)", re.MULTILINE)
PYTHON_FUNCTION_PATTERN = re.compile(r"^def\s+(\w+)", re.MULTILINE)
PYTHON_IMPORT_PATTERN = re.compile(
    r"^(?:from\s+[\w.]+\s+)?import\s+([\w., ]+)", re.MULTILINE
)
PYTHON_FUNC_CALL_PATTERN = re.compile(r"\b([a-zA-Z_]\w*)\s*\(")
PYTHON_METHOD_CALL_PATTERN = re.compile(r"\.([a-zA-Z_]\w*)\s*\(")
# More restrictive class instantiation - avoid built-ins and exceptions
# Only match classes that are likely custom (longer names, not in builtins)
PYTHON_CLASS_INSTANTIATION_PATTERN = re.compile(
    r"\b([A-Z][a-z]{2,}\w*|[A-Z]{2,}[a-z]\w*)\s*\("
)
PYTHON_ATTR_ACCESS_PATTERN = re.compile(r"\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)")

# Documentation patterns
DOC_HEADER_PATTERN = re.compile(r"^#+\s+(.+)$", re.MULTILINE)

# GitHub URL patterns - more restrictive to prevent domain spoofing
GITHUB_URL_PATTERNS = [
    re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/?$"),  # Full GitHub URLs only
    re.compile(
        r"^([a-zA-Z0-9][a-zA-Z0-9\-]{0,38})/([a-zA-Z0-9][a-zA-Z0-9\-_.]{0,99})$"
    ),  # owner/repo format with valid chars
]

# GitHub repository name validation pattern
GITHUB_REPO_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9\-_.]+$")

# Safe keyword sets for each language
MATLAB_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "elseif",
        "end",
        "for",
        "while",
        "try",
        "catch",
        "function",
        "return",
        "break",
        "continue",
        "switch",
        "case",
        "otherwise",
        "classdef",
    }
)

CPP_KEYWORDS = frozenset(
    {
        "if",
        "else",
        "for",
        "while",
        "do",
        "switch",
        "case",
        "default",
        "break",
        "continue",
        "return",
        "goto",
        "try",
        "catch",
        "throw",
        "new",
        "delete",
        "sizeof",
        "typeof",
        "class",
        "struct",
        "union",
        "enum",
        "namespace",
        "using",
        "template",
        "typename",
        "public",
        "private",
        "protected",
        "virtual",
        "static",
        "const",
        "mutable",
        "int",
        "char",
        "short",
        "long",
        "float",
        "double",
        "bool",
        "void",
        "auto",
        "std",
        "cout",
        "cin",
        "endl",
        "string",
        "vector",
        "map",
        "set",
        "list",
    }
)

PYTHON_KEYWORDS = frozenset(
    {
        "and",
        "as",
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "exec",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "not",
        "or",
        "pass",
        "print",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
        "None",
        "True",
        "False",
    }
)

# Python built-ins and common exceptions to exclude from class instantiations
PYTHON_BUILTINS = frozenset(
    {
        # Basic types
        "int",
        "float",
        "str",
        "list",
        "dict",
        "tuple",
        "set",
        "bool",
        "bytes",
        "bytearray",
        "complex",
        "frozenset",
        "memoryview",
        "object",
        "slice",
        "type",
        "super",
        "property",
        # Functions
        "len",
        "range",
        "enumerate",
        "zip",
        "open",
        "abs",
        "min",
        "max",
        "sum",
        "all",
        "any",
        "bin",
        "hex",
        "oct",
        "ord",
        "chr",
        "round",
        "pow",
        "divmod",
        "hash",
        "id",
        "repr",
        "sorted",
        "reversed",
        "filter",
        "map",
        "iter",
        "next",
        "vars",
        "dir",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "callable",
        "isinstance",
        "issubclass",
        "eval",
        "exec",
        "compile",
        "format",
        "print",
        "input",
        # Decorators and metaclasses
        "staticmethod",
        "classmethod",
        # Common exceptions
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "ImportError",
        "RuntimeError",
        "OSError",
        "IOError",
        "StopIteration",
        "FileNotFoundError",
        "PermissionError",
        "NotImplementedError",
        "ZeroDivisionError",
        "OverflowError",
        "MemoryError",
        "SystemError",
        "AssertionError",
        "SyntaxError",
        "IndentationError",
        "TabError",
        "NameError",
        "UnboundLocalError",
        "RecursionError",
        "UnicodeError",
    }
)


def validate_content_for_regex(content: str, max_length: int = 1000000) -> str:
    """
    Validate and sanitize content before regex processing to prevent attacks.

    Args:
        content: Content to validate
        max_length: Maximum allowed content length

    Returns:
        Validated content

    Raises:
        ValueError: If content is invalid or too large
    """
    if not isinstance(content, str):
        raise ValueError("Content must be a string")

    if len(content) > max_length:
        raise ValueError(f"Content too large: {len(content)} > {max_length} characters")

    # Truncate extremely long lines that could cause ReDoS
    lines = content.split("\n")
    sanitized_lines = []
    for line in lines:
        if len(line) > 10000:  # Truncate lines longer than 10k chars
            line = line[:10000] + "...[truncated]"
        sanitized_lines.append(line)

    return "\n".join(sanitized_lines)


def safe_regex_findall(
    pattern: re.Pattern, content: str, max_matches: int = 1000
) -> List[str]:
    """
    Safely execute regex findall with limits to prevent ReDoS and memory issues.

    Args:
        pattern: Compiled regex pattern
        content: Content to search
        max_matches: Maximum number of matches to return

    Returns:
        List of matches (truncated if necessary)
    """
    try:
        matches = pattern.findall(content)
        if len(matches) > max_matches:
            print(f"Warning: Truncating {len(matches)} matches to {max_matches}")
            matches = matches[:max_matches]
        return matches
    except re.error as e:
        print(f"Regex error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in regex processing: {e}")
        return []


def safe_regex_search(pattern: re.Pattern, content: str):
    """
    Safely execute regex search with error handling.

    Args:
        pattern: Compiled regex pattern
        content: Content to search

    Returns:
        Match object or None
    """
    try:
        return pattern.search(content)
    except re.error as e:
        print(f"Regex error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error in regex processing: {e}")
        return None


# ============================================================================
# GitHub API Utilities
# ============================================================================

# File extensions to skip during crawling
SKIP_EXTENSIONS = {
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".bmp",
    ".tiff",
    ".ico",
    # Data files
    ".mat",
    ".seq",
    ".dat",
    ".csv",
    ".h5",
    ".hdf5",
    ".nc",
    ".nii",
    # Archives
    ".zip",
    ".gz",
    ".tar",
    ".rar",
    ".7z",
    ".bz2",
    # Compiled code/executables
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".mex",
    ".mexw64",
    ".mexa64",
    ".mexmaci64",
    # Other binary formats
    ".docx",
    ".xlsx",
    ".pptx",
    ".bin",
    # Build system files (low RAG value)
    ".am",
    ".in",
    ".ac",
    ".cmake",
    ".pro",
    ".pri",
    # IDE/Editor files (no RAG value)
    ".sln",
    ".vcxproj",
    ".filters",
    ".user",
    ".prj",
    # Test output/data files (no RAG value)
    ".out",
    ".log",
    ".tmp",
    # Package management (usually not useful for RAG)
    ".lock",
    ".resolved",
}


def get_github_client():
    """
    Get GitHub API client with authentication.

    Returns:
        dict: Headers for GitHub API requests
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("GITHUB_TOKEN must be set in environment variables")

    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Crawl4AI-MCP-Server",
    }


def parse_github_url(github_url: str) -> Tuple[str, str]:
    """
    Parse GitHub repository URL to extract owner and repository name.

    Args:
        github_url: GitHub repository URL (e.g., "https://github.com/pulseq/pulseq")

    Returns:
        Tuple of (owner, repo_name)
    """
    try:
        # Handle various GitHub URL formats
        github_url = github_url.strip().rstrip("/")

        # Validate input length to prevent ReDoS
        if len(github_url) > 500:
            raise ValueError(f"GitHub URL too long: {len(github_url)} characters")

        # Extract from various formats using pre-compiled safe patterns
        for pattern in GITHUB_URL_PATTERNS:
            match = safe_regex_search(pattern, github_url)
            if match:
                return match.group(1), match.group(2)

        raise ValueError(f"Invalid GitHub URL format: {github_url}")

    except Exception as e:
        print(f"Error parsing GitHub URL '{github_url}': {e}")
        raise ValueError(f"Invalid GitHub URL format: {github_url}") from e


def get_repository_tree(owner: str, repo: str, sha: str = "HEAD") -> Dict[str, Any]:
    """
    Get the complete file tree of a GitHub repository.

    Args:
        owner: Repository owner
        repo: Repository name
        sha: Git SHA or branch name (default: "HEAD")

    Returns:
        Dictionary containing repository tree information
    """
    headers = get_github_client()

    # First get the default branch if sha is "HEAD"
    if sha == "HEAD":
        repo_info_url = f"https://api.github.com/repos/{owner}/{repo}"
        repo_response = requests.get(repo_info_url, headers=headers)
        repo_response.raise_for_status()
        default_branch = repo_response.json().get("default_branch", "main")
        sha = default_branch

    # Get repository tree recursively
    tree_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{sha}"
    tree_response = requests.get(tree_url, headers=headers, params={"recursive": "1"})
    tree_response.raise_for_status()

    return tree_response.json()


def get_file_content(
    owner: str, repo: str, file_path: str, sha: str = "HEAD"
) -> Dict[str, Any]:
    """
    Get the content of a specific file from GitHub repository.

    Args:
        owner: Repository owner
        repo: Repository name
        file_path: Path to the file in the repository
        sha: Git SHA or branch name (default: "HEAD")

    Returns:
        Dictionary containing file content and metadata
    """
    headers = get_github_client()

    content_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    params = {}
    if sha != "HEAD":
        params["ref"] = sha

    response = requests.get(content_url, headers=headers, params=params)
    response.raise_for_status()

    file_data = response.json()

    # Decode base64 content if it's a file (not a directory)
    if file_data.get("type") == "file" and "content" in file_data:
        try:
            decoded_content = base64.b64decode(file_data["content"]).decode("utf-8")
            file_data["decoded_content"] = decoded_content
        except UnicodeDecodeError:
            # Handle binary files
            file_data["decoded_content"] = None
            file_data["is_binary"] = True

    return file_data


def filter_crawlable_files(
    tree_data: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Filter repository files to exclude binary/data files and return crawlable files.

    Args:
        tree_data: Repository tree data from GitHub API

    Returns:
        Tuple of (crawlable_files, skip_report)
    """
    crawlable_files = []
    skip_report = {}

    for item in tree_data.get("tree", []):
        if item["type"] != "blob":  # Skip directories
            continue

        file_path = item["path"]
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()

        # Check if file should be skipped
        if file_ext in SKIP_EXTENSIONS:
            skip_report[file_ext] = skip_report.get(file_ext, 0) + 1
            continue

        # Add to crawlable files
        crawlable_files.append(
            {
                "path": file_path,
                "name": file_name,
                "directory": os.path.dirname(file_path),
                "extension": file_ext,
                "size": item.get("size", 0),
                "sha": item["sha"],
                "url": item["url"],
            }
        )

    return crawlable_files, skip_report


def extract_language_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """
    Extract language-specific metadata from file content based on file extension.

    Args:
        content: File content
        file_path: Path to the file

    Returns:
        Dictionary containing language-specific metadata
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".m":
        return extract_matlab_metadata(content, file_path)
    elif file_ext in [".cpp", ".c", ".h"]:
        return extract_cpp_metadata(content, file_path)
    elif file_ext in [".py"]:
        return extract_python_metadata(content, file_path)
    elif file_ext in [".md", ".txt", ".rst"]:
        return extract_doc_metadata(content, file_path)
    elif file_ext == ".ipynb":
        return extract_notebook_metadata(content, file_path)
    else:
        return {
            "language": "unknown",
            "file_type": file_ext[1:] if file_ext else "no_extension",
        }


def extract_matlab_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """Extract MATLAB-specific metadata from file content."""
    try:
        # Validate input
        content = validate_content_for_regex(content)

        metadata = {
            "language": "matlab",
            "function_name": "",
            "function_signature": "",
            "dependencies": [],
        }

        # Extract function information using pre-compiled pattern
        function_match = safe_regex_search(MATLAB_FUNCTION_PATTERN, content)
        if function_match:
            metadata["function_name"] = function_match.group(1)
            metadata["function_signature"] = function_match.group(0)

        # Enhanced dependency extraction for MATLAB
        dependencies = set()

        # Find function calls using safe patterns
        func_calls = safe_regex_findall(MATLAB_FUNC_CALL_PATTERN, content)
        for func in func_calls:
            if func.lower() not in MATLAB_KEYWORDS and len(func) > 1:
                dependencies.add(func)

        # Extract method calls
        method_calls = safe_regex_findall(MATLAB_METHOD_CALL_PATTERN, content)
        for method in method_calls:
            if method.lower() not in MATLAB_KEYWORDS:
                dependencies.add(method)

        # Extract Pulseq-specific function patterns using pre-compiled patterns
        for pattern in MATLAB_PULSEQ_PATTERNS:
            matches = safe_regex_findall(pattern, content)
            for match in matches:
                dependencies.add(match)

        # Remove the current function name from dependencies to avoid self-reference
        if metadata["function_name"]:
            dependencies.discard(metadata["function_name"])

        metadata["dependencies"] = sorted(list(dependencies))
        return metadata

    except ValueError as e:
        print(
            f"Input validation error in MATLAB metadata extraction for {file_path}: {e}"
        )
        return {
            "language": "matlab",
            "function_name": "",
            "function_signature": "",
            "dependencies": [],
            "error": str(e),
        }
    except Exception as e:
        print(f"Unexpected error in MATLAB metadata extraction for {file_path}: {e}")
        return {
            "language": "matlab",
            "function_name": "",
            "function_signature": "",
            "dependencies": [],
            "error": str(e),
        }


def extract_cpp_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """Extract C++-specific metadata from file content."""
    try:
        # Validate input
        content = validate_content_for_regex(content)

        metadata = {
            "language": "cpp",
            "is_header": file_path.endswith(".h"),
            "class_names": [],
            "function_names": [],
            "includes": [],
            "dependencies": [],
        }

        # Extract class names using safe patterns
        class_matches = safe_regex_findall(CPP_CLASS_PATTERN, content)
        metadata["class_names"] = list(set(class_matches))

        # Extract function names using safe patterns
        func_matches = safe_regex_findall(CPP_FUNCTION_PATTERN, content)
        metadata["function_names"] = list(set(func_matches))

        # Extract includes using safe patterns
        include_matches = safe_regex_findall(CPP_INCLUDE_PATTERN, content)
        metadata["includes"] = list(set(include_matches))

        # Enhanced dependency extraction for C++
        dependencies = set()

        # Extract function calls using safe patterns
        func_calls = safe_regex_findall(CPP_FUNC_CALL_PATTERN, content)
        for func in func_calls:
            if func.lower() not in CPP_KEYWORDS and len(func) > 1:
                dependencies.add(func)

        # Extract method calls using safe patterns
        method_calls = safe_regex_findall(CPP_METHOD_CALL_PATTERN, content)
        for method in method_calls:
            if method.lower() not in CPP_KEYWORDS:
                dependencies.add(method)

        # Extract scope resolution calls using safe patterns
        scope_calls = safe_regex_findall(CPP_SCOPE_CALL_PATTERN, content)
        for scope, func in scope_calls:
            if scope.lower() not in CPP_KEYWORDS and func.lower() not in CPP_KEYWORDS:
                dependencies.add(f"{scope}::{func}")

        # Extract class/struct instantiations using safe patterns
        class_instantiations = safe_regex_findall(
            CPP_CLASS_INSTANTIATION_PATTERN, content
        )
        for cls in class_instantiations:
            if cls not in CPP_KEYWORDS:
                dependencies.add(cls)

        # Extract template instantiations using SAFE pattern (ReDoS-resistant)
        template_instantiations = safe_regex_findall(CPP_TEMPLATE_PATTERN, content)
        for template in template_instantiations:
            if template.lower() not in CPP_KEYWORDS:
                dependencies.add(template)

        # Add include files as dependencies (header dependencies)
        for include in metadata["includes"]:
            # Extract base filename without path and extension for include dependencies
            include_name = include.split("/")[-1].split(".")[0]
            if include_name and include_name not in CPP_KEYWORDS:
                dependencies.add(f"include:{include}")

        # Remove current function/class names from dependencies to avoid self-reference
        for func_name in metadata["function_names"]:
            dependencies.discard(func_name)
        for class_name in metadata["class_names"]:
            dependencies.discard(class_name)

        metadata["dependencies"] = sorted(list(dependencies))
        return metadata

    except ValueError as e:
        print(f"Input validation error in C++ metadata extraction for {file_path}: {e}")
        return {
            "language": "cpp",
            "is_header": file_path.endswith(".h"),
            "class_names": [],
            "function_names": [],
            "includes": [],
            "dependencies": [],
            "error": str(e),
        }
    except Exception as e:
        print(f"Unexpected error in C++ metadata extraction for {file_path}: {e}")
        return {
            "language": "cpp",
            "is_header": file_path.endswith(".h"),
            "class_names": [],
            "function_names": [],
            "includes": [],
            "dependencies": [],
            "error": str(e),
        }


def extract_python_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """Extract Python-specific metadata from file content."""
    try:
        # Validate input
        content = validate_content_for_regex(content)

        metadata = {
            "language": "python",
            "class_names": [],
            "function_names": [],
            "imports": [],
            "dependencies": [],
        }

        # Extract class names using safe patterns
        class_matches = safe_regex_findall(PYTHON_CLASS_PATTERN, content)
        metadata["class_names"] = list(set(class_matches))

        # Extract function names using safe patterns
        func_matches = safe_regex_findall(PYTHON_FUNCTION_PATTERN, content)
        metadata["function_names"] = list(set(func_matches))

        # Extract imports using safe patterns
        import_matches = safe_regex_findall(PYTHON_IMPORT_PATTERN, content)
        metadata["imports"] = list(
            set([imp.strip() for line in import_matches for imp in line.split(",")])
        )

        # Enhanced dependency extraction for Python
        dependencies = set()

        # Extract function calls using safe patterns
        func_calls = safe_regex_findall(PYTHON_FUNC_CALL_PATTERN, content)
        for func in func_calls:
            if (
                func.lower() not in PYTHON_KEYWORDS
                and func not in PYTHON_BUILTINS
                and len(func) > 1
            ):
                dependencies.add(func)

        # Extract method calls using safe patterns
        method_calls = safe_regex_findall(PYTHON_METHOD_CALL_PATTERN, content)
        for method in method_calls:
            if method.lower() not in PYTHON_KEYWORDS:
                dependencies.add(method)

        # Extract class instantiations using IMPROVED pattern (avoids built-ins and exceptions)
        class_instantiations = safe_regex_findall(
            PYTHON_CLASS_INSTANTIATION_PATTERN, content
        )
        for cls in class_instantiations:
            # Filter out built-ins and common exceptions to reduce false positives
            if cls not in PYTHON_BUILTINS:
                dependencies.add(cls)

        # Extract attribute access patterns using safe patterns
        attr_access = safe_regex_findall(PYTHON_ATTR_ACCESS_PATTERN, content)
        for module, attr in attr_access:
            # Add both module and attribute as potential dependencies
            if module not in PYTHON_KEYWORDS:
                dependencies.add(f"{module}.{attr}")

        # Remove current function/class names from dependencies to avoid self-reference
        for func_name in metadata["function_names"]:
            dependencies.discard(func_name)
        for class_name in metadata["class_names"]:
            dependencies.discard(class_name)

        metadata["dependencies"] = sorted(list(dependencies))
        return metadata

    except ValueError as e:
        print(
            f"Input validation error in Python metadata extraction for {file_path}: {e}"
        )
        return {
            "language": "python",
            "class_names": [],
            "function_names": [],
            "imports": [],
            "dependencies": [],
            "error": str(e),
        }
    except Exception as e:
        print(f"Unexpected error in Python metadata extraction for {file_path}: {e}")
        return {
            "language": "python",
            "class_names": [],
            "function_names": [],
            "imports": [],
            "dependencies": [],
            "error": str(e),
        }


def extract_doc_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """Extract documentation-specific metadata from file content."""
    try:
        # Validate input
        content = validate_content_for_regex(content)

        metadata = {
            "language": "markdown" if file_path.endswith(".md") else "text",
            "doc_type": "readme" if "readme" in file_path.lower() else "documentation",
            "section_headers": [],
        }

        # Extract section headers for markdown using safe patterns
        if file_path.endswith(".md"):
            header_matches = safe_regex_findall(DOC_HEADER_PATTERN, content)
            metadata["section_headers"] = header_matches

        return metadata

    except ValueError as e:
        print(
            f"Input validation error in documentation metadata extraction for {file_path}: {e}"
        )
        return {
            "language": "markdown" if file_path.endswith(".md") else "text",
            "doc_type": "documentation",
            "section_headers": [],
            "error": str(e),
        }
    except Exception as e:
        print(
            f"Unexpected error in documentation metadata extraction for {file_path}: {e}"
        )
        return {
            "language": "markdown" if file_path.endswith(".md") else "text",
            "doc_type": "documentation",
            "section_headers": [],
            "error": str(e),
        }


def process_notebook_as_script(
    content: str, file_path: str
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Process Jupyter notebook by combining code cells into a single script
    and using markdown/raw cells for context.

    Args:
        content: Raw notebook JSON content
        file_path: Path to the notebook file

    Returns:
        Tuple of (combined_code, context, notebook_metadata)
    """
    try:
        import json

        notebook_data = json.loads(content)

        if "cells" not in notebook_data:
            return (
                "",
                "Empty notebook - no cells found",
                {"language": "unknown", "cell_count": 0},
            )

        code_blocks = []
        context_blocks = []

        # Extract language from notebook metadata
        notebook_metadata = notebook_data.get("metadata", {})
        kernel_info = notebook_metadata.get("kernelspec", {})
        language = kernel_info.get("language", "python").lower()

        # Process each cell
        for i, cell in enumerate(notebook_data["cells"]):
            cell_type = cell.get("cell_type", "")
            source = cell.get("source", [])

            # Convert source to string if it's a list
            if isinstance(source, list):
                source_text = "".join(source).strip()
            else:
                source_text = str(source).strip()

            if cell_type == "code" and source_text:
                code_blocks.append(source_text)

            elif cell_type in ["markdown", "raw"] and source_text:
                context_blocks.append(f"[{cell_type.upper()} CELL {i}]\n{source_text}")

        # Combine all code blocks into a single seamless script
        combined_code = "\n\n".join(code_blocks) if code_blocks else ""

        # Create context from markdown and raw cells
        context_text = (
            "\n\n".join(context_blocks)
            if context_blocks
            else "No markdown context available"
        )

        # Return notebook metadata
        nb_metadata = {
            "language": language,
            "cell_count": len(notebook_data["cells"]),
            "code_cells": len(code_blocks),
            "context_cells": len(context_blocks),
            "kernel_name": kernel_info.get("name", ""),
            "has_combined_code": bool(combined_code),
        }

        return combined_code, context_text, nb_metadata

    except json.JSONDecodeError:
        return (
            "",
            "Invalid notebook JSON format",
            {"language": "unknown", "error": "JSON parse error"},
        )
    except Exception as e:
        return (
            "",
            f"Error processing notebook: {str(e)}",
            {"language": "unknown", "error": str(e)},
        )


def extract_notebook_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """
    Extract Jupyter notebook metadata by processing it as a script.
    This integrates with existing language-specific processing.
    """
    combined_code, context, nb_metadata = process_notebook_as_script(content, file_path)

    if not combined_code:
        return {
            "language": "jupyter",
            "notebook_language": nb_metadata.get("language", "unknown"),
            **nb_metadata,
        }

    # Get the notebook's primary language
    notebook_language = nb_metadata.get("language", "python")

    # Process the combined code using existing language-specific extractors
    if notebook_language == "matlab":
        script_metadata = extract_matlab_metadata(
            combined_code, f"{file_path}_combined.m"
        )
    elif notebook_language == "python":
        script_metadata = extract_python_metadata(
            combined_code, f"{file_path}_combined.py"
        )
    elif notebook_language in ["cpp", "c++"]:
        script_metadata = extract_cpp_metadata(
            combined_code, f"{file_path}_combined.cpp"
        )
    else:
        # Default metadata for unknown languages
        script_metadata = {
            "language": notebook_language,
            "function_names": [],
            "dependencies": [],
        }

    # Combine notebook metadata with script metadata
    return {
        **script_metadata,
        "notebook_language": notebook_language,
        "combined_code_length": len(combined_code),
        "context_length": len(context),
        **nb_metadata,
    }


# ============================================================================
# API Reference Extraction for Functions, Classes, and Methods
# ============================================================================


def extract_python_api_references(content: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Extract API references from Python code using AST parsing.

    Args:
        content: Python source code
        file_path: Path to the file being analyzed

    Returns:
        List of API reference dictionaries
    """
    api_refs = []

    try:
        tree = ast.parse(content)

        # Walk through all nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Extract function information
                api_ref = {
                    "name": node.name,
                    "language": "python",
                    "signature": _get_python_function_signature(node),
                    "description": ast.get_docstring(node) or "",
                    "parameters": _extract_python_parameters(node),
                    "returns": _extract_python_return_type(node),
                    "file_path": file_path,
                    "line_number": node.lineno if hasattr(node, "lineno") else None,
                    "is_async": isinstance(node, ast.AsyncFunctionDef),
                }
                api_refs.append(api_ref)

            elif isinstance(node, ast.ClassDef):
                # Extract class information
                class_ref = {
                    "name": node.name,
                    "language": "python",
                    "signature": f"class {node.name}",
                    "description": ast.get_docstring(node) or "",
                    "parameters": {},  # Classes don't have parameters in the same way
                    "returns": None,
                    "file_path": file_path,
                    "line_number": node.lineno if hasattr(node, "lineno") else None,
                    "is_class": True,
                }
                api_refs.append(class_ref)

                # Extract methods within the class
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_ref = {
                            "name": f"{node.name}.{item.name}",
                            "language": "python",
                            "signature": _get_python_function_signature(
                                item, class_name=node.name
                            ),
                            "description": ast.get_docstring(item) or "",
                            "parameters": _extract_python_parameters(item),
                            "returns": _extract_python_return_type(item),
                            "file_path": file_path,
                            "line_number": item.lineno
                            if hasattr(item, "lineno")
                            else None,
                            "is_async": isinstance(item, ast.AsyncFunctionDef),
                            "is_method": True,
                        }
                        api_refs.append(method_ref)

    except Exception as e:
        print(f"Error parsing Python file {file_path}: {e}")

    return api_refs


def _get_python_function_signature(
    node: ast.FunctionDef, class_name: Optional[str] = None
) -> str:
    """Extract function signature from AST node."""
    args = []

    # Handle positional arguments
    for i, arg in enumerate(node.args.args):
        arg_str = arg.arg
        # Add type annotation if available
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        # Add default value if available
        default_offset = len(node.args.args) - len(node.args.defaults)
        if i >= default_offset:
            default_idx = i - default_offset
            default_val = ast.unparse(node.args.defaults[default_idx])
            arg_str += f"={default_val}"
        args.append(arg_str)

    # Handle *args
    if node.args.vararg:
        arg_str = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            arg_str += f": {ast.unparse(node.args.vararg.annotation)}"
        args.append(arg_str)

    # Handle **kwargs
    if node.args.kwarg:
        arg_str = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            arg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
        args.append(arg_str)

    # Build signature
    func_name = f"{class_name}.{node.name}" if class_name else node.name
    signature = f"{func_name}({', '.join(args)})"

    # Add return type if available
    if node.returns:
        signature += f" -> {ast.unparse(node.returns)}"

    return signature


def _extract_python_parameters(node: ast.FunctionDef) -> Dict[str, Any]:
    """Extract parameter information from function node."""
    params = {}

    # Handle positional arguments
    for i, arg in enumerate(node.args.args):
        param_info = {
            "type": ast.unparse(arg.annotation) if arg.annotation else "Any",
            "required": True,
        }

        # Check if has default value
        default_offset = len(node.args.args) - len(node.args.defaults)
        if i >= default_offset:
            param_info["required"] = False
            default_idx = i - default_offset
            param_info["default"] = ast.unparse(node.args.defaults[default_idx])

        params[arg.arg] = param_info

    # Handle *args
    if node.args.vararg:
        params[f"*{node.args.vararg.arg}"] = {
            "type": ast.unparse(node.args.vararg.annotation)
            if node.args.vararg.annotation
            else "Any",
            "required": False,
            "variadic": True,
        }

    # Handle **kwargs
    if node.args.kwarg:
        params[f"**{node.args.kwarg.arg}"] = {
            "type": ast.unparse(node.args.kwarg.annotation)
            if node.args.kwarg.annotation
            else "Any",
            "required": False,
            "variadic": True,
        }

    return params


def _extract_python_return_type(node: ast.FunctionDef) -> str:
    """Extract return type from function node."""
    if node.returns:
        return ast.unparse(node.returns)

    # Try to infer from docstring or return statements if no annotation
    docstring = ast.get_docstring(node)
    if docstring:
        # Look for Returns: or :returns: in docstring
        returns_match = re.search(
            r"(?:Returns?|:returns?):\s*(.+?)(?:\n|$)", docstring, re.IGNORECASE
        )
        if returns_match:
            return returns_match.group(1).strip()

    return "Any"


def extract_matlab_api_references(content: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Extract API references from MATLAB code using regex patterns.

    Args:
        content: MATLAB source code
        file_path: Path to the file being analyzed

    Returns:
        List of API reference dictionaries
    """
    api_refs = []

    def _find_matching_paren(text: str, start: int) -> int:
        """Find the matching closing parenthesis, handling nested parentheses."""
        count = 0
        for i in range(start, len(text)):
            if text[i] == '(':
                count += 1
            elif text[i] == ')':
                count -= 1
                if count == 0:
                    return i
        return -1
    
    def _parse_matlab_function_line(func_line: str, line_number: int) -> dict:
        """Parse a single MATLAB function line with robust parentheses handling."""
        try:
            # Use a simple approach: find "function" keyword, then parse manually
            func_line = func_line.strip()
            
            if not func_line.startswith('function'):
                return None
            
            # Remove "function" and leading whitespace
            remainder = func_line[8:].strip()  # 8 is len("function")
            
            # Check if there's an assignment (=)
            if '=' in remainder:
                parts = remainder.split('=', 1)
                outputs = parts[0].strip()
                remainder = parts[1].strip()
            else:
                outputs = ""
            
            # Find the function name and opening parenthesis
            paren_pos = remainder.find('(')
            if paren_pos == -1:
                return None
            
            func_name = remainder[:paren_pos].strip()
            
            # Find matching closing parenthesis using our helper function
            paren_end = _find_matching_paren(remainder, paren_pos)
            
            if paren_end <= paren_pos:
                return None
            
            inputs = remainder[paren_pos+1:paren_end]
            
            return {
                'outputs': outputs,
                'func_name': func_name,
                'inputs': inputs,
                'line_number': line_number
            }
            
        except Exception as e:
            print(f"Error parsing function line '{func_line}': {e}")
            return None

    MATLAB_CLASSDEF_PATTERN = re.compile(
        r"^\s*classdef\s+(\w+)(?:\s*<\s*(\w+))?", re.MULTILINE
    )

    # Find all function definitions using improved parsing approach
    try:
        # Find function declaration lines first
        function_starts = re.finditer(r'^\s*function\s+', content, re.MULTILINE)
        
        for match in function_starts:
            try:
                # Extract the line containing the function declaration
                line_start = match.start()
                line_end = content.find('\n', line_start)
                if line_end == -1:
                    line_end = len(content)
                
                func_line = content[line_start:line_end]
                line_number = content[:line_start].count('\n') + 1
                
                # Parse the function line with robust parentheses handling
                parsed_func = _parse_matlab_function_line(func_line, line_number)
                
                if not parsed_func:
                    continue
                
                outputs = parsed_func['outputs']
                func_name = parsed_func['func_name']
                inputs = parsed_func['inputs']
                
                # Clean up outputs to remove extra brackets if present
                if outputs:
                    if outputs.startswith('[') and outputs.endswith(']'):
                        outputs = outputs[1:-1]

                # Extract docstring (comments immediately after function declaration)
                docstring = _extract_matlab_docstring(content, line_end)

                # Build signature
                signature = "function "
                if outputs:
                    if ',' in outputs or outputs.count(' ') > 0:
                        signature += f"[{outputs}] = "
                    else:
                        signature += f"{outputs} = "
                signature += f"{func_name}({inputs})"

                # Parse parameters
                parameters = _parse_matlab_parameters(inputs, docstring)

                # Parse return description
                returns = _parse_matlab_returns(outputs, docstring)

                api_ref = {
                    "name": func_name,
                    "language": "matlab",
                    "signature": signature,
                    "description": docstring,
                    "parameters": parameters,
                    "returns": returns,
                    "file_path": file_path,
                    "line_number": line_number,
                }
                api_refs.append(api_ref)
                
            except Exception as e:
                # Skip this match if there's an issue parsing it
                print(f"Warning: Failed to parse MATLAB function at line {content[:line_start].count('\n') + 1} in {file_path}: {e}")
                continue
                
    except Exception as e:
        # Handle any overall parsing errors
        print(f"Error in MATLAB function parsing for {file_path}: {e}")
        return api_refs

    # Find all class definitions
    for match in MATLAB_CLASSDEF_PATTERN.finditer(content):
        class_name = match.group(1)
        parent_class = match.group(2) or ""

        # Extract class docstring
        docstring = _extract_matlab_docstring(content, match.end())

        signature = f"classdef {class_name}"
        if parent_class:
            signature += f" < {parent_class}"

        class_ref = {
            "name": class_name,
            "language": "matlab",
            "signature": signature,
            "description": docstring,
            "parameters": {},
            "returns": None,
            "file_path": file_path,
            "line_number": content[: match.start()].count("\n") + 1,
            "is_class": True,
        }
        api_refs.append(class_ref)

        # TODO: Extract methods within classes (more complex parsing needed)

    return api_refs


def _extract_matlab_docstring(content: str, start_pos: int) -> str:
    """Extract MATLAB docstring (comments after function/class declaration)."""
    lines = content[start_pos:].split("\n")
    docstring_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("%"):
            # Remove % and leading space
            comment = stripped[1:].lstrip()
            docstring_lines.append(comment)
        elif stripped and not stripped.startswith("%"):
            # Stop at first non-comment line
            break

    return "\n".join(docstring_lines)


def _parse_matlab_parameters(param_str: str, docstring: str) -> Dict[str, Any]:
    """Parse MATLAB function parameters."""
    params = {}

    # Split parameters by comma
    if param_str.strip():
        param_names = [p.strip() for p in param_str.split(",")]

        for param in param_names:
            # Basic parameter info
            param_info = {
                "type": "double",  # Default MATLAB type
                "required": True,
            }

            # Try to extract type from docstring
            # Look for patterns like "param - type description" or "param (type) description"
            try:
                # Escape the parameter name to handle special regex characters
                escaped_param = re.escape(param)
                type_pattern = rf"{escaped_param}\s*[-:]\s*\(?([\w\[\]]+)\)?"
                type_match = re.search(type_pattern, docstring, re.IGNORECASE)
                if type_match:
                    param_info["type"] = type_match.group(1)
            except re.error as e:
                # If regex fails, skip type extraction for this parameter
                print(f"Warning: Failed to extract type for parameter '{param}': {e}")
                pass

            params[param] = param_info

    return params


def _parse_matlab_returns(output_str: str, docstring: str) -> str:
    """Parse MATLAB function return values."""
    if not output_str:
        return "None"

    # Look for Returns: or Output: in docstring
    returns_match = re.search(
        r"(?:Returns?|Outputs?):\s*(.+?)(?:\n|$)", docstring, re.IGNORECASE
    )
    if returns_match:
        return returns_match.group(1).strip()

    # Default description based on output variables
    outputs = [o.strip() for o in output_str.split(",")]
    if len(outputs) == 1:
        return f"{outputs[0]}"
    else:
        return f"[{', '.join(outputs)}]"


def extract_cpp_api_references(content: str, file_path: str) -> List[Dict[str, Any]]:
    """
    Extract API references from C++ code using regex patterns.

    Args:
        content: C++ source code
        file_path: Path to the file being analyzed

    Returns:
        List of API reference dictionaries
    """
    api_refs = []

    # Pre-compiled regex patterns for C++
    CPP_FUNCTION_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:(?:static|inline|virtual|extern|const)\s+)*"
        r"([\w:]+(?:\s*[*&]+)?)\s+"  # Return type
        r"(\w+)\s*"  # Function name
        r"\(([^)]*)\)\s*"  # Parameters
        r"(?:const)?\s*"  # Optional const
        r"(?:override|final)?\s*"  # Optional override/final
        r"[{;]",  # Opening brace or semicolon
        re.MULTILINE,
    )

    CPP_CLASS_PATTERN = re.compile(
        r"(?:^|\n)\s*(?:template\s*<[^>]+>\s*)?"
        r"(?:class|struct)\s+(\w+)"
        r"(?:\s*:\s*(?:public|private|protected)\s+(\w+))?",
        re.MULTILINE,
    )

    # Find all function definitions
    for match in CPP_FUNCTION_PATTERN.finditer(content):
        return_type = match.group(1).strip()
        func_name = match.group(2)
        params = match.group(3) or ""

        # Skip if it's a control structure
        if func_name in ["if", "for", "while", "switch", "catch"]:
            continue

        # Extract docstring (Doxygen comments before function)
        docstring = _extract_cpp_docstring(content, match.start())

        # Build signature
        signature = f"{return_type} {func_name}({params})"

        # Parse parameters
        parameters = _parse_cpp_parameters(params, docstring)

        # Parse return description
        returns = _parse_cpp_returns(return_type, docstring)

        api_ref = {
            "name": func_name,
            "language": "cpp",
            "signature": signature,
            "description": docstring,
            "parameters": parameters,
            "returns": returns,
            "file_path": file_path,
            "line_number": content[: match.start()].count("\n") + 1,
        }
        api_refs.append(api_ref)

    # Find all class definitions
    for match in CPP_CLASS_PATTERN.finditer(content):
        class_name = match.group(1)
        parent_class = match.group(2) or ""

        # Extract class docstring
        docstring = _extract_cpp_docstring(content, match.start())

        signature = f"class {class_name}"
        if parent_class:
            signature += f" : public {parent_class}"

        class_ref = {
            "name": class_name,
            "language": "cpp",
            "signature": signature,
            "description": docstring,
            "parameters": {},
            "returns": None,
            "file_path": file_path,
            "line_number": content[: match.start()].count("\n") + 1,
            "is_class": True,
        }
        api_refs.append(class_ref)

    return api_refs


def _extract_cpp_docstring(content: str, pos: int) -> str:
    """Extract C++ docstring (Doxygen comments before declaration)."""
    lines = content[:pos].split("\n")
    docstring_lines = []

    # Look backwards for comment block
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("///") or stripped.startswith("//!"):
            # Doxygen single-line comment
            comment = stripped[3:].lstrip()
            docstring_lines.insert(0, comment)
        elif stripped.startswith("//"):
            # Regular comment (might be documentation)
            comment = stripped[2:].lstrip()
            docstring_lines.insert(0, comment)
        elif stripped.endswith("*/"):
            # End of multi-line comment block
            # TODO: Parse multi-line Doxygen comments
            break
        elif stripped and not any(stripped.startswith(c) for c in ["///", "//!", "//"]):
            # Stop at non-comment line
            break

    return "\n".join(docstring_lines)


def _parse_cpp_parameters(param_str: str, docstring: str) -> Dict[str, Any]:
    """Parse C++ function parameters."""
    params = {}

    if param_str.strip():
        # Split parameters by comma (handling nested templates)
        param_list = []
        current_param = ""
        depth = 0

        for char in param_str:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                param_list.append(current_param.strip())
                current_param = ""
                continue
            current_param += char

        if current_param.strip():
            param_list.append(current_param.strip())

        # Parse each parameter
        for param in param_list:
            # Match type and name
            param_match = re.match(r"(.+?)\s+(\w+)(?:\s*=\s*(.+))?$", param.strip())
            if param_match:
                param_type = param_match.group(1).strip()
                param_name = param_match.group(2)
                default_value = param_match.group(3)

                param_info = {"type": param_type, "required": default_value is None}

                if default_value:
                    param_info["default"] = default_value.strip()

                # Try to extract description from docstring
                # Look for @param or \param
                param_desc_pattern = rf"[@\\]param\s+{param_name}\s+(.+?)(?=[@\\]|$)"
                desc_match = re.search(
                    param_desc_pattern, docstring, re.IGNORECASE | re.DOTALL
                )
                if desc_match:
                    param_info["description"] = desc_match.group(1).strip()

                params[param_name] = param_info

    return params


def _parse_cpp_returns(return_type: str, docstring: str) -> str:
    """Parse C++ function return description."""
    if return_type == "void":
        return "void"

    # Look for @return or \return in docstring
    returns_match = re.search(
        r"[@\\]returns?\s+(.+?)(?=[@\\]|$)", docstring, re.IGNORECASE | re.DOTALL
    )
    if returns_match:
        return returns_match.group(1).strip()

    return return_type


# ============================================================================
# Pulseq API Reference Filtering System
# ============================================================================

# Module-level blacklist for performance (initialized once)
BUILTIN_BLACKLIST = {
    # MATLAB built-ins
    'round', 'floor', 'ceil', 'max', 'min', 'sum', 'length', 'size', 
    'ones', 'zeros', 'find', 'ismember', 'strcmp', 'isempty', 'any', 'all',
    'sin', 'cos', 'tan', 'sqrt', 'abs', 'mod', 'diff', 'mean', 'std',
    'plot', 'figure', 'title', 'xlabel', 'ylabel', 'legend', 'axis',
    'fprintf', 'disp', 'error', 'warning', 'assert',
    'if', 'for', 'while', 'switch', 'try', 'catch', 'end',
    'exist', 'which', 'clear', 'clc', 'close', 'hold', 'grid',
    'sprintf', 'num2str', 'str2num', 'regexp', 'regexprep',
    # Python built-ins
    'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter',
    'int', 'float', 'str', 'list', 'dict', 'set', 'tuple',
    'min', 'max', 'sum', 'abs', 'round', 'bool', 'type', 'isinstance',
    'hasattr', 'getattr', 'setattr', 'delattr', 'open', 'close',
    # C++ built-ins
    'printf', 'cout', 'cin', 'endl', 'std', 'vector', 'string',
    'int', 'float', 'double', 'char', 'bool', 'void', 'malloc', 'free'
}


def is_official_pulseq_repo(github_url: str) -> bool:
    """
    Determine if this is an official Pulseq repository using secure URL parsing.
    
    Args:
        github_url: GitHub repository URL
        
    Returns:
        True if this is an official Pulseq repository, False otherwise
        
    Note:
        This function fails safely - malformed URLs or None inputs return False
    """
    # Input validation - fail safely for None or non-string inputs
    if not github_url or not isinstance(github_url, str):
        return False
    
    # Strip whitespace and validate basic format
    github_url = github_url.strip()
    if not github_url:
        return False
    
    try:
        # Parse URL securely to prevent bypass attacks
        parsed_url = urlparse(github_url)
        
        # Handle URLs without protocol (e.g., "github.com/pulseq/pulseq")
        if not parsed_url.netloc and not parsed_url.scheme:
            # Try prepending https:// and re-parsing
            parsed_url = urlparse(f"https://{github_url}")
        
        # Validate protocol (allow http/https or no protocol)
        if parsed_url.scheme and parsed_url.scheme.lower() not in ['http', 'https']:
            return False
        
        # Validate it's actually a GitHub URL
        if parsed_url.netloc.lower() not in ['github.com', 'www.github.com']:
            return False
        
        # Extract path and normalize (remove leading/trailing slashes)
        path = parsed_url.path.strip('/')
        if not path:
            return False
        
        # Official repository paths (case-insensitive matching)
        official_repo_paths = {
            'pulseq/pulseq',
            'imr-framework/pypulseq'
        }
        
        # Check if the path matches an official repository
        return path.lower() in official_repo_paths
        
    except (ValueError, AttributeError) as e:
        # URL parsing failed - fail safely
        return False


def is_pulseq_core_function(func_name: str, file_path: str, language: str) -> bool:
    """
    Determine if a function is core Pulseq functionality that should be in API references.
    
    Args:
        func_name: Function name to check
        file_path: File path where function is defined (can be None)
        language: Programming language (python, matlab, cpp, etc.)
        
    Returns:
        True if this is a core Pulseq function, False otherwise
        
    Note:
        This function fails safely - None or malformed inputs return False
    """
    # Input validation - fail safely for None or non-string inputs
    if not func_name or not isinstance(func_name, str):
        return False
    
    # Strip whitespace and validate
    func_name = func_name.strip()
    if not func_name:
        return False
    
    # Validate file_path parameter (can be None, but if provided must be string)
    if file_path is not None and not isinstance(file_path, str):
        return False
    
    # Validate and safely handle language parameter
    if language is not None:
        if not isinstance(language, str):
            return False
        try:
            language = language.lower().strip()
        except (AttributeError, TypeError):
            # Language parameter is malformed - fail safely
            return False
    else:
        language = ""
    
    # Use module-level blacklist for performance
    if func_name.lower() in BUILTIN_BLACKLIST:
        return False
    
    # Pulseq-specific function patterns
    pulseq_patterns = [
        'make',       # makeAdc, makeTrapezoid, makeSincPulse, etc.
        'calc',       # calcDuration, calcRfCenter, etc.
        'read',       # readasc, readdat, etc. (domain-specific readers)
        'write',      # writeSeq, etc. (domain-specific writers)
        'split',      # splitGradientAt, etc.
        'add',        # addBlock, addGradients, etc.
        'align',      # align gradients
        'scale',      # scaleGrad, etc.
        'check',      # checkTiming, etc.
        'set',        # setDefinition, etc.
        'calculate',  # calculateKspace, etc.
        'parse',      # parsemr, etc.
        'install',    # Pulseq install functions
        'test',       # testReport, etc.
    ]
    
    # Check if function starts with Pulseq patterns
    func_lower = func_name.lower()
    for pattern in pulseq_patterns:
        if func_lower.startswith(pattern):
            return True
    
    # Check file path patterns for Pulseq namespace (with cross-platform support)
    if file_path and isinstance(file_path, str):
        try:
            # Normalize path separators for cross-platform compatibility
            normalized_path = file_path.replace('\\', '/').lower()
            
            # MATLAB package structure
            if '/+mr/' in normalized_path:
                return True
            # Vendor-specific utilities
            vendor_patterns = ['/+siemens/', '/+philips/', '/+ge/', '/+bruker/']
            if any(vendor in normalized_path for vendor in vendor_patterns):
                return True
            # Python pulseq modules
            if '/pypulseq/' in normalized_path or 'pulseq/' in normalized_path:
                return True
        except (AttributeError, TypeError):
            # Path processing failed - continue without path-based checks
            pass
    
    # Domain-specific function names (even without standard prefixes)
    domain_functions = {
        # File I/O specific to MRI/Pulseq
        'readasc', 'readdat', 'parsemr', 'writeseq',
        # Core sequence operations  
        'addblock', 'setdefinition', 'checktiming', 'testreport',
        # Pulseq-specific utilities
        'sequence', 'opts', 'sound',  # if these are Pulseq methods
        # Domain-specific operations
        'align', 'splitgradientat', 'addgradients',
        # MRI-specific terminology
        'siemens', 'philips', 'bruker', 'ge',  # vendor functions
        'adc', 'rf', 'gradient', 'sequence', 'pulse', 'trapezoid',
        # Hardware/format specific
        'asc', 'dat', 'seq', 'ismrmrd'
    }
    
    if func_name.lower() in domain_functions:
        return True
    
    # Additional checks for specific language patterns
    if language == "matlab":
        # MATLAB-specific Pulseq patterns
        if func_name.lower().endswith('pulse') or func_name.lower().endswith('grad'):
            return True
        # Package functions (functions in +mr namespace)
        if file_path and isinstance(file_path, str):
            try:
                normalized_path = file_path.replace('\\', '/')
                if '+mr' in normalized_path:
                    return True
            except (AttributeError, TypeError):
                pass
    
    elif language == "python":
        # Python-specific patterns
        if 'pulseq' in func_name.lower() or 'mr_' in func_name.lower():
            return True
    
    elif language in ["cpp", "c"]:
        # C++ specific patterns - more conservative due to less documentation
        # Only include if clearly domain-specific
        domain_keywords = ['pulse', 'grad', 'sequence', 'adc', 'rf', 'siemens']
        if any(keyword in func_name.lower() for keyword in domain_keywords):
            return True
    
    return False


def extract_api_references(
    content: str, metadata: Dict[str, Any], file_info: Dict[str, Any], github_url: str = ""
) -> List[Dict[str, Any]]:
    """
    Extract API references based on file type and language, with optional filtering.

    Args:
        content: Source code content
        metadata: Language metadata from extract_language_metadata
        file_info: File information including path and extension
        github_url: GitHub repository URL for filtering (optional)

    Returns:
        List of API reference dictionaries (filtered for official repos only)
    """
    language = metadata.get("language", "").lower()
    file_path = file_info.get("path", "")
    
    # Only extract API references from official Pulseq repositories
    if not is_official_pulseq_repo(github_url):
        return []

    # Extract all API references first
    if language == "python":
        raw_api_refs = extract_python_api_references(content, file_path)
    elif language == "matlab":
        raw_api_refs = extract_matlab_api_references(content, file_path)
    elif language in ["cpp", "c"]:
        raw_api_refs = extract_cpp_api_references(content, file_path)
    else:
        # Unsupported language
        return []
    
    # Filter API references to include only core Pulseq functions
    filtered_api_refs = []
    for api_ref in raw_api_refs:
        func_name = api_ref.get("name", "")
        if is_pulseq_core_function(func_name, file_path, language):
            filtered_api_refs.append(api_ref)
    
    return filtered_api_refs


# ============================================================================
# Pulseq Version Detection System
# ============================================================================


def detect_pulseq_version(repo_path: str, source_id: str) -> str:
    """
    Detect Pulseq version using multiple fallback strategies.

    Args:
        repo_path: Path to the repository
        source_id: Source identifier (e.g., "github.com/pulseq/pulseq")

    Returns:
        Version string in semantic versioning format or "unknown"
    """
    # Strategy 1: Check standard version files
    version = _check_version_files(repo_path)
    if version:
        return _normalize_version(version)

    # Strategy 2: Parse git tags
    version = _get_latest_git_tag(repo_path)
    if version:
        return _normalize_version(version)

    # Strategy 3: Scan code for version constants
    version = _scan_code_for_version(repo_path)
    if version:
        return _normalize_version(version)

    # Strategy 4: Parse file headers
    version = _extract_version_from_headers(repo_path)
    if version:
        return _normalize_version(version)

    # Strategy 5: Default fallback
    return "unknown"


def _check_version_files(repo_path: str) -> Optional[str]:
    """Check standard version files in repository."""
    version_files = [
        "VERSION",
        "version.txt",
        "setup.py",
        "Contents.m",
        "CMakeLists.txt",
        "configure.ac",
        "package.json",
        "pyproject.toml",
    ]

    for version_file in version_files:
        file_path = os.path.join(repo_path, version_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Different patterns for different file types
                if version_file == "setup.py":
                    # Look for version= or __version__
                    match = re.search(r'version\s*=\s*[\'"]([^\'\"]+)[\'"]', content)
                    if match:
                        return match.group(1)

                elif version_file == "Contents.m":
                    # MATLAB toolbox version
                    match = re.search(r"%\s*Version\s*:\s*(\S+)", content)
                    if match:
                        return match.group(1)

                elif version_file in ["CMakeLists.txt", "configure.ac"]:
                    # CMake/autoconf project version
                    match = re.search(r"VERSION\s+(\d+\.\d+(?:\.\d+)?)", content)
                    if match:
                        return match.group(1)

                elif version_file == "package.json":
                    # JSON format
                    try:
                        import json

                        data = json.loads(content)
                        return data.get("version", "")
                    except (json.JSONDecodeError, KeyError):
                        pass

                elif version_file == "pyproject.toml":
                    # TOML format
                    match = re.search(r'version\s*=\s*[\'"]([^\'\"]+)[\'"]', content)
                    if match:
                        return match.group(1)

                else:
                    # Generic version file - just read first line
                    first_line = content.strip().split("\n")[0]
                    # Extract version-like pattern
                    match = re.search(r"(\d+\.\d+(?:\.\d+)?)", first_line)
                    if match:
                        return match.group(1)

            except Exception as e:
                print(f"Error reading version file {file_path}: {e}")
                continue

    return None


def _get_latest_git_tag(repo_path: str) -> Optional[str]:
    """Get latest git tag as version."""
    try:
        # Get latest tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            tag = result.stdout.strip()
            # Remove 'v' prefix if present
            if tag.startswith("v"):
                tag = tag[1:]
            return tag

    except Exception as e:
        print(f"Error getting git tags: {e}")

    return None


def _scan_code_for_version(repo_path: str) -> Optional[str]:
    """Scan code files for version constants."""
    patterns = [
        r'__version__\s*=\s*[\'"]([^\'\"]+)[\'"]',  # Python
        r'VERSION\s*=\s*[\'"]([^\'\"]+)[\'"]',  # Generic
        r'version\s*=\s*[\'"]([^\'\"]+)[\'"]',  # Generic lowercase
        r'#define\s+VERSION\s+[\'"]([^\'\"]+)[\'"]',  # C/C++
        r'mr_version\s*=\s*[\'"]([^\'\"]+)[\'"]',  # Pulseq-specific
    ]

    # Common locations for version definitions
    search_paths = [
        "",  # Root
        "src",
        "lib",
        "pulseq",
        "matlab",
        "python",
    ]

    for search_path in search_paths:
        path = os.path.join(repo_path, search_path)
        if not os.path.exists(path):
            continue

        # Look for version in common files
        for filename in ["__init__.py", "version.py", "mr_version.m", "config.h"]:
            file_path = os.path.join(path, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    for pattern in patterns:
                        match = re.search(pattern, content)
                        if match:
                            return match.group(1)

                except Exception as e:
                    print(f"Error scanning file {file_path}: {e}")
                    continue

    return None


def _extract_version_from_headers(repo_path: str) -> Optional[str]:
    """Extract version from file headers/comments."""
    # Look for main entry files
    entry_files = ["README.md", "README.txt", "pulseq.m", "mr.py", "__init__.py"]

    for entry_file in entry_files:
        file_path = os.path.join(repo_path, entry_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    # Read first 50 lines
                    lines = f.readlines()[:50]
                    content = "".join(lines)

                # Look for version patterns in headers
                patterns = [
                    r"Version\s*:\s*(\d+\.\d+(?:\.\d+)?)",
                    r"v(\d+\.\d+(?:\.\d+)?)",
                    r"Release\s*(\d+\.\d+(?:\.\d+)?)",
                    r"Pulseq\s+(?:version\s+)?(\d+\.\d+(?:\.\d+)?)",
                ]

                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        return match.group(1)

            except Exception as e:
                print(f"Error reading header from {file_path}: {e}")
                continue

    return None


def _normalize_version(version: str) -> str:
    """
    Normalize version string to semantic versioning format.

    Examples:
        "1.4" -> "1.4.0"
        "v2.0.1" -> "2.0.1"
        "2024.1" -> "2024.1.0"
    """
    # Remove common prefixes
    version = version.strip()
    if version.startswith("v"):
        version = version[1:]

    # Split by dots
    parts = version.split(".")

    # Ensure at least 3 parts for semantic versioning
    while len(parts) < 3:
        parts.append("0")

    # Take only first 3 parts
    parts = parts[:3]

    # Validate each part is numeric
    try:
        parts = [str(int(p)) for p in parts]
        return ".".join(parts)
    except ValueError:
        # If not all numeric, return as-is
        return version


# ============================================================================
# Local Git Operations for GitHub Repository Crawling
# ============================================================================


def clone_github_repository(owner: str, repo: str, temp_dir: str) -> str:
    """
    Clone a GitHub repository to a temporary directory.

    Args:
        owner: Repository owner
        repo: Repository name
        temp_dir: Temporary directory path

    Returns:
        Path to the cloned repository

    Raises:
        ValueError: If owner/repo names are invalid
        subprocess.CalledProcessError: If git clone fails
    """
    # Add input validation to prevent command injection using pre-compiled pattern
    if not safe_regex_search(GITHUB_REPO_NAME_PATTERN, owner) or not safe_regex_search(
        GITHUB_REPO_NAME_PATTERN, repo
    ):
        raise ValueError("Invalid owner or repository name format")

    repo_url = f"https://github.com/{owner}/{repo}.git"
    repo_path = os.path.join(temp_dir, repo)

    # Ensure repo_path is within temp_dir boundaries
    temp_dir = os.path.abspath(temp_dir)
    repo_path = os.path.abspath(repo_path)
    if not repo_path.startswith(temp_dir):
        raise ValueError("Repository path outside temporary directory")

    try:
        # Use shallow clone to minimize download size and time, with timeout
        cmd = ["git", "clone", "--depth", "1", repo_url, repo_path]
        subprocess.run(
            cmd, capture_output=True, text=True, check=True, timeout=300
        )  # 5 minute timeout

        print(f"Successfully cloned {owner}/{repo} to {repo_path}")
        return repo_path

    except subprocess.TimeoutExpired as e:
        error_msg = f"Git clone timed out for {owner}/{repo}"
        print(error_msg)
        raise RuntimeError(error_msg) from e
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to clone {owner}/{repo}: {e.stderr}"
        print(error_msg)
        raise RuntimeError(error_msg) from e


def get_local_file_tree(
    repo_path: str, max_files: int = 10000
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Walk through a local repository and get file structure, similar to filter_crawlable_files.

    Args:
        repo_path: Path to the local repository
        max_files: Maximum number of files to process (default: 10000)

    Returns:
        Tuple of (crawlable_files, skip_report)
    """
    crawlable_files = []
    skip_report = {}
    file_count = 0

    repo_path_obj = Path(repo_path)

    # Walk through all files in the repository
    for file_path in repo_path_obj.rglob("*"):
        if file_count >= max_files:
            print(f"Warning: File limit ({max_files}) reached, stopping traversal")
            skip_report["__truncated__"] = 1  # Indicate truncation occurred
            break

        if file_path.is_file():
            file_count += 1

            # Get relative path from repo root
            relative_path = file_path.relative_to(repo_path_obj)

            # Skip .git directory files
            if ".git" in relative_path.parts:
                continue

            file_name = file_path.name
            file_ext = file_path.suffix.lower()

            # Check if file should be skipped
            if file_ext in SKIP_EXTENSIONS:
                skip_report[file_ext] = skip_report.get(file_ext, 0) + 1
                continue

            # Get file size
            try:
                file_size = file_path.stat().st_size
                # Skip files larger than 10MB to prevent memory issues
                if file_size > 10 * 1024 * 1024:
                    skip_report["__large_files__"] = (
                        skip_report.get("__large_files__", 0) + 1
                    )
                    continue
            except OSError:
                file_size = 0

            # Add to crawlable files
            crawlable_files.append(
                {
                    "path": str(relative_path),
                    "name": file_name,
                    "directory": str(relative_path.parent)
                    if relative_path.parent != Path(".")
                    else "",
                    "extension": file_ext,
                    "size": file_size,
                    "full_path": str(file_path),  # Add full path for local reading
                }
            )

    return crawlable_files, skip_report


def read_local_file_content(file_path: str) -> Dict[str, Any]:
    """
    Read content from a local file, similar to get_file_content but for local files.

    Args:
        file_path: Full path to the local file

    Returns:
        Dictionary containing file content and metadata
    """
    try:
        # Try to read as text with UTF-8 encoding first
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {"type": "file", "decoded_content": content, "is_binary": False}

    except UnicodeDecodeError:
        # Try with latin-1 encoding as fallback
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

            return {"type": "file", "decoded_content": content, "is_binary": False}
        except Exception:
            # Mark as binary file
            return {"type": "file", "decoded_content": None, "is_binary": True}

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {
            "type": "file",
            "decoded_content": None,
            "is_binary": True,
            "error": str(e),
        }


def cleanup_temp_directory(temp_dir: str) -> None:
    """
    Safely clean up a temporary directory.

    Args:
        temp_dir: Path to the temporary directory to clean up
    """
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Warning: Failed to clean up temporary directory {temp_dir}: {e}")


def create_temp_directory() -> str:
    """
    Create a temporary directory for git operations.

    Returns:
        Path to the created temporary directory
    """
    return tempfile.mkdtemp(prefix="github_clone_")
