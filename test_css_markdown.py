#!/usr/bin/env python3
"""
Test script to see how Crawl4AI converts CSS content to markdown.
This will help examine the raw markdown output to understand CSS formatting.
"""

import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode

async def test_css_markdown():
    """Test how Crawl4AI handles CSS content in markdown conversion."""
    
    url = "https://pulseq.github.io/writeGradientEcho.html"
    
    # Create crawler instance
    async with AsyncWebCrawler(verbose=True) as crawler:
        # Configure to bypass cache and get fresh content
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            print("=== RAW MARKDOWN OUTPUT ===")
            print(f"Content length: {len(result.markdown)} characters")
            print()
            
            # Look for CSS-like content at the beginning
            markdown_start = result.markdown[:2000]
            print("=== FIRST 2000 CHARACTERS ===")
            print(repr(markdown_start))
            print()
            
            # Look for any CSS-like patterns
            css_indicators = ['margin:', 'padding:', 'border:', 'color:', 'font-', 'background']
            css_found = []
            
            for indicator in css_indicators:
                if indicator in result.markdown:
                    # Find the context around CSS indicators
                    idx = result.markdown.find(indicator)
                    if idx != -1:
                        start = max(0, idx - 100)
                        end = min(len(result.markdown), idx + 200)
                        context = result.markdown[start:end]
                        css_found.append({
                            'indicator': indicator,
                            'position': idx,
                            'context': context
                        })
            
            if css_found:
                print("=== CSS-LIKE CONTENT FOUND ===")
                for item in css_found[:3]:  # Show first 3 matches
                    print(f"Indicator: '{item['indicator']}' at position {item['position']}")
                    print("Context:")
                    print(repr(item['context']))
                    print("-" * 50)
            else:
                print("=== NO CSS-LIKE CONTENT FOUND ===")
            
            # Look for triple backticks
            backtick_count = result.markdown.count('```')
            print(f"=== TRIPLE BACKTICKS COUNT: {backtick_count} ===")
            
            if backtick_count > 0:
                # Find first few code blocks
                pos = 0
                code_blocks = []
                while pos < len(result.markdown) and len(code_blocks) < 3:
                    start = result.markdown.find('```', pos)
                    if start == -1:
                        break
                    end = result.markdown.find('```', start + 3)
                    if end == -1:
                        break
                    
                    # Extract the code block
                    code_block = result.markdown[start:end + 3]
                    code_blocks.append({
                        'start': start,
                        'end': end + 3,
                        'content': code_block[:300] + '...' if len(code_block) > 300 else code_block
                    })
                    pos = end + 3
                
                print("=== FIRST FEW CODE BLOCKS ===")
                for i, block in enumerate(code_blocks):
                    print(f"Block {i + 1} (positions {block['start']}-{block['end']}):")
                    print(repr(block['content']))
                    print("-" * 50)
        else:
            print(f"Failed to crawl: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(test_css_markdown())