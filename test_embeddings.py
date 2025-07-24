#!/usr/bin/env python3
"""
Test script for local embedding functionality.
"""
import sys
import os
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from local_embeddings import create_embedding, create_embeddings_batch

def test_single_embedding():
    """Test creating a single embedding."""
    print("Testing single embedding...")
    
    text = "This is a test sentence for embedding generation."
    embedding = create_embedding(text)
    
    print(f"Text: {text}")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Embedding type: {type(embedding)}")
    
    # Verify embedding is not all zeros
    non_zero_count = sum(1 for x in embedding if x != 0.0)
    print(f"Non-zero values: {non_zero_count}/{len(embedding)}")
    
    assert len(embedding) == 1024, f"Expected 1024 dimensions, got {len(embedding)}"
    assert non_zero_count > 0, "Embedding should not be all zeros"
    
    print("✓ Single embedding test passed\n")

def test_batch_embeddings():
    """Test creating batch embeddings."""
    print("Testing batch embeddings...")
    
    texts = [
        "This is the first test sentence.",
        "Here is another sentence for testing.",
        "The third sentence in our test batch.",
        "",  # Test empty string
        "Final sentence with some technical terms like embeddings and transformers."
    ]
    
    embeddings = create_embeddings_batch(texts)
    
    print(f"Number of texts: {len(texts)}")
    print(f"Number of embeddings: {len(embeddings)}")
    
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        non_zero_count = sum(1 for x in embedding if x != 0.0)
        print(f"Text {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"  Dimensions: {len(embedding)}, Non-zero: {non_zero_count}")
    
    # Verify all embeddings have correct dimensions
    for i, embedding in enumerate(embeddings):
        assert len(embedding) == 1024, f"Embedding {i} has wrong dimensions: {len(embedding)}"
    
    # Verify non-empty texts have non-zero embeddings
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        if text.strip():  # Non-empty text
            non_zero_count = sum(1 for x in embedding if x != 0.0)
            assert non_zero_count > 0, f"Non-empty text {i} should have non-zero embedding"
    
    print("✓ Batch embeddings test passed\n")

def test_similarity():
    """Test semantic similarity between embeddings."""
    print("Testing semantic similarity...")
    
    # Create embeddings for similar and dissimilar texts
    similar_texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug."
    ]
    
    different_texts = [
        "The cat sat on the mat.",
        "Quantum physics involves complex mathematical equations."
    ]
    
    similar_embeddings = create_embeddings_batch(similar_texts)
    different_embeddings = create_embeddings_batch(different_texts)
    
    # Calculate cosine similarity
    def cosine_similarity(a, b):
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot_product / (norm_a * norm_b)
    
    similar_score = cosine_similarity(similar_embeddings[0], similar_embeddings[1])
    different_score = cosine_similarity(different_embeddings[0], different_embeddings[1])
    
    print(f"Similarity between similar texts: {similar_score:.4f}")
    print(f"Similarity between different texts: {different_score:.4f}")
    
    # Similar texts should have higher similarity than different texts
    assert similar_score > different_score, f"Similar texts ({similar_score}) should be more similar than different texts ({different_score})"
    
    print("✓ Semantic similarity test passed\n")

def main():
    """Run all tests."""
    print("Starting local embedding tests...\n")
    
    try:
        test_single_embedding()
        test_batch_embeddings()
        test_similarity()
        
        print("🎉 All tests passed! Local embedding service is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()