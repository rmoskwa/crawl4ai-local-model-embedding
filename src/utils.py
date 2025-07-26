"""
Utility functions for the Crawl4AI MCP server.
"""

import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple
from supabase import create_client, Client
from urllib.parse import urlparse
import time
import requests
import base64
import re
from local_embeddings import create_embedding, create_embeddings_batch

import google.generativeai as genai

# Configure Google Generative AI
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    genai.configure(api_key=google_api_key)


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
    model_choice = os.getenv("MODEL_CHOICE")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # Check if we have both model choice and API key
    if not model_choice or not google_api_key:
        print(
            "Contextual embeddings require MODEL_CHOICE and GOOGLE_API_KEY - using original chunk"
        )
        return chunk, False

    try:
        # Create the prompt for generating contextual information
        prompt = f"""<document> 
{full_document[:25000]} 
</document>
Here is the chunk we want to situate within the whole document 
<chunk> 
{chunk}
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

        # Use Google Generative AI to generate contextual information
        model = genai.GenerativeModel(model_choice)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=200,
            ),
        )

        # Extract the generated context
        context = response.text.strip()

        # Combine the context with the original chunk
        contextual_text = f"{context}\n---\n{chunk}"

        return contextual_text, True

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


def add_documents_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    contents: List[str],
    metadatas: List[Dict[str, Any]],
    url_to_full_document: Dict[str, str],
    batch_size: int = 20,
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

            # Extract source_id from URL
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
                    time.sleep(retry_delay)
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
    code: str, context_before: str, context_after: str, is_code_dominated: bool = False
) -> str:
    """
    Generate a summary for a code example using its surrounding context.

    Args:
        code: The code example
        context_before: Context before the code
        context_after: Context after the code
        is_code_dominated: Whether this is code-dominated content (increases context limit)

    Returns:
        A summary of what the code example demonstrates
    """
    model_choice = os.getenv("MODEL_CHOICE")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # Check if we have the required configuration
    if not model_choice or not google_api_key:
        print(
            "Code example summary generation requires MODEL_CHOICE and GOOGLE_API_KEY - using generic summary"
        )
        return "Code example for demonstration purposes."

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
        # Use Google Generative AI to generate code example summary
        model = genai.GenerativeModel(model_choice)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=100,
            ),
        )

        return response.text.strip()

    except Exception as e:
        print(f"Error generating code example summary: {e}")
        return "Code example for demonstration purposes."


def add_code_examples_to_supabase(
    client: Client,
    urls: List[str],
    chunk_numbers: List[int],
    code_examples: List[str],
    summaries: List[str],
    metadatas: List[Dict[str, Any]],
    batch_size: int = 20,
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

            # Extract source_id from URL
            parsed_url = urlparse(urls[idx])
            source_id = parsed_url.netloc or parsed_url.path

            batch_data.append(
                {
                    "url": urls[idx],
                    "chunk_number": chunk_numbers[idx],
                    "content": code_examples[idx],
                    "summary": summaries[idx],
                    "metadata": metadatas[idx],  # Store as JSON object, not string
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
                    time.sleep(retry_delay)
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

    This function uses Google Generative AI to generate a concise summary of the source content.

    Args:
        source_id: The source ID (domain)
        content: The content to extract a summary from
        max_length: Maximum length of the summary

    Returns:
        A summary string
    """
    # Default summary if we can't extract anything meaningful
    default_summary = f"Content from {source_id}"

    if not content or len(content.strip()) == 0:
        return default_summary

    # Get the model choice from environment variables
    model_choice = os.getenv("MODEL_CHOICE")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    # Check if we have the required configuration
    if not model_choice or not google_api_key:
        print(
            f"Source summary generation requires MODEL_CHOICE and GOOGLE_API_KEY - using generic summary for {source_id}"
        )
        return f"Documentation and content from {source_id}"

    # Limit content length to avoid token limits
    truncated_content = content[:25000] if len(content) > 25000 else content

    # Create the prompt for generating the summary
    prompt = f"""<source_content>
{truncated_content}
</source_content>

The above content is from the documentation for '{source_id}'. Please provide a concise summary (3-5 sentences) that describes what this library/tool/framework is about. The summary should help understand what the library/tool/framework accomplishes and the purpose.
"""

    try:
        # Use Google Generative AI to generate the summary
        model = genai.GenerativeModel(model_choice)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=150,
            ),
        )

        # Extract the generated summary
        summary = response.text.strip()

        # Ensure the summary is not too long
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."

        return summary

    except Exception as e:
        print(
            f"Error generating summary with LLM for {source_id}: {e}. Using default summary."
        )
        return default_summary


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
# GitHub API Utilities
# ============================================================================

# File extensions to skip during crawling
SKIP_EXTENSIONS = {
    # Images
    '.png', '.jpg', '.jpeg', '.gif', '.svg', '.bmp', '.tiff', '.ico',
    
    # Data files  
    '.mat', '.seq', '.dat', '.csv', '.h5', '.hdf5', '.nc', '.nii',
    
    # Archives
    '.zip', '.gz', '.tar', '.rar', '.7z', '.bz2',
    
    # Compiled code/executables
    '.exe', '.dll', '.so', '.dylib', '.mex', '.mexw64', '.mexa64', '.mexmaci64',
    
    # Other binary formats
    '.pdf', '.docx', '.xlsx', '.pptx', '.bin'
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
        "User-Agent": "Crawl4AI-MCP-Server"
    }


def parse_github_url(github_url: str) -> Tuple[str, str]:
    """
    Parse GitHub repository URL to extract owner and repository name.
    
    Args:
        github_url: GitHub repository URL (e.g., "https://github.com/pulseq/pulseq")
        
    Returns:
        Tuple of (owner, repo_name)
    """
    # Handle various GitHub URL formats
    github_url = github_url.strip().rstrip('/')
    
    # Extract from various formats
    patterns = [
        r'github\.com/([^/]+)/([^/]+)',  # https://github.com/owner/repo
        r'^([^/]+)/([^/]+)$'            # owner/repo
    ]
    
    for pattern in patterns:
        match = re.search(pattern, github_url)
        if match:
            return match.group(1), match.group(2)
    
    raise ValueError(f"Invalid GitHub URL format: {github_url}")


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


def get_file_content(owner: str, repo: str, file_path: str, sha: str = "HEAD") -> Dict[str, Any]:
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


def filter_crawlable_files(tree_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
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
        crawlable_files.append({
            "path": file_path,
            "name": file_name,
            "directory": os.path.dirname(file_path),
            "extension": file_ext,
            "size": item.get("size", 0),
            "sha": item["sha"],
            "url": item["url"]
        })
    
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
        return {"language": "unknown", "file_type": file_ext[1:] if file_ext else "no_extension"}


def extract_matlab_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """Extract MATLAB-specific metadata from file content."""
    metadata = {
        "language": "matlab",
        "function_name": "",
        "function_signature": "",
        "dependencies": []
    }
    
    # Extract function information
    function_match = re.search(r"function\s+(?:\[?[^\]]*\]?\s*=\s*)?(\w+)\s*\([^)]*\)", content, re.MULTILINE)
    if function_match:
        metadata["function_name"] = function_match.group(1)
        metadata["function_signature"] = function_match.group(0)
    
    # Extract function calls (dependencies)
    func_calls = re.findall(r"(\w+)\s*\(", content)
    metadata["dependencies"] = list(set(func_calls))
    
    return metadata


def extract_cpp_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """Extract C++-specific metadata from file content."""
    metadata = {
        "language": "cpp",
        "is_header": file_path.endswith(".h"),
        "class_names": [],
        "function_names": [],
        "includes": []
    }
    
    # Extract class names
    class_matches = re.findall(r"class\s+(\w+)", content)
    metadata["class_names"] = list(set(class_matches))
    
    # Extract function names
    func_matches = re.findall(r"(?:^|\s)(\w+)\s*\([^)]*\)\s*{", content, re.MULTILINE)
    metadata["function_names"] = list(set(func_matches))
    
    # Extract includes
    include_matches = re.findall(r"#include\s+[<\"](.*)[>\"]", content)
    metadata["includes"] = list(set(include_matches))
    
    return metadata


def extract_python_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """Extract Python-specific metadata from file content."""
    metadata = {
        "language": "python",
        "class_names": [],
        "function_names": [],
        "imports": []
    }
    
    # Extract class names
    class_matches = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
    metadata["class_names"] = list(set(class_matches))
    
    # Extract function names
    func_matches = re.findall(r"^def\s+(\w+)", content, re.MULTILINE)
    metadata["function_names"] = list(set(func_matches))
    
    # Extract imports
    import_matches = re.findall(r"^(?:from\s+[\w.]+\s+)?import\s+([\w., ]+)", content, re.MULTILINE)
    metadata["imports"] = list(set([imp.strip() for line in import_matches for imp in line.split(",")]))
    
    return metadata


def extract_doc_metadata(content: str, file_path: str) -> Dict[str, Any]:
    """Extract documentation-specific metadata from file content."""
    metadata = {
        "language": "markdown" if file_path.endswith(".md") else "text",
        "doc_type": "readme" if "readme" in file_path.lower() else "documentation",
        "section_headers": []
    }
    
    # Extract section headers for markdown
    if file_path.endswith(".md"):
        header_matches = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
        metadata["section_headers"] = header_matches
    
    return metadata


def process_notebook_as_script(content: str, file_path: str) -> Tuple[str, str, Dict[str, Any]]:
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
            return "", "Empty notebook - no cells found", {"language": "unknown", "cell_count": 0}
        
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
        context_text = "\n\n".join(context_blocks) if context_blocks else "No markdown context available"
        
        # Return notebook metadata
        nb_metadata = {
            "language": language,
            "cell_count": len(notebook_data["cells"]),
            "code_cells": len(code_blocks),
            "context_cells": len(context_blocks),
            "kernel_name": kernel_info.get("name", ""),
            "has_combined_code": bool(combined_code)
        }
        
        return combined_code, context_text, nb_metadata
        
    except json.JSONDecodeError:
        return "", "Invalid notebook JSON format", {"language": "unknown", "error": "JSON parse error"}
    except Exception as e:
        return "", f"Error processing notebook: {str(e)}", {"language": "unknown", "error": str(e)}


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
            **nb_metadata
        }
    
    # Get the notebook's primary language
    notebook_language = nb_metadata.get("language", "python")
    
    # Process the combined code using existing language-specific extractors
    if notebook_language == "matlab":
        script_metadata = extract_matlab_metadata(combined_code, f"{file_path}_combined.m")
    elif notebook_language == "python":
        script_metadata = extract_python_metadata(combined_code, f"{file_path}_combined.py")
    elif notebook_language in ["cpp", "c++"]:
        script_metadata = extract_cpp_metadata(combined_code, f"{file_path}_combined.cpp")
    else:
        # Default metadata for unknown languages
        script_metadata = {
            "language": notebook_language,
            "function_names": [],
            "dependencies": []
        }
    
    # Combine notebook metadata with script metadata
    return {
        **script_metadata,
        "notebook_language": notebook_language,
        "combined_code_length": len(combined_code),
        "context_length": len(context),
        **nb_metadata
    }
