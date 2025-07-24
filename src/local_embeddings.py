"""
Local embedding service using BAAI-bge-large-en-v1.5 model.
"""

import os
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional
import time


class LocalEmbeddingService:
    """
    Service for generating embeddings using the local BAAI-bge-large-en-v1.5 model.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the local embedding service.

        Args:
            model_path: Path to the local model directory. If None, uses environment variable.
        """
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine model path - Docker container path only
        if model_path is None:
            model_path = os.getenv("BGE_MODEL_PATH")
            if model_path is None:
                raise ValueError("BGE_MODEL_PATH environment variable must be set")

        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the BGE model and tokenizer from local path only."""
        try:
            print(f"Loading BGE model from local path: {self.model_path}")

            # If path points to snapshots directory, find the latest snapshot
            actual_model_path = self.model_path
            if self.model_path.endswith("/snapshots"):
                if os.path.exists(self.model_path):
                    snapshots = [
                        d
                        for d in os.listdir(self.model_path)
                        if os.path.isdir(os.path.join(self.model_path, d))
                    ]
                    if snapshots:
                        # Use the first (and likely only) snapshot
                        actual_model_path = os.path.join(self.model_path, snapshots[0])
                        print(f"Found snapshot: {actual_model_path}")

            # Check if local path exists
            if not os.path.exists(actual_model_path):
                raise FileNotFoundError(
                    f"Local model path does not exist: {actual_model_path}"
                )

            # Load from local path only
            print(f"Loading model from: {actual_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                actual_model_path, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                actual_model_path, local_files_only=True
            )

            # Move model to appropriate device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode

            print(
                f"BGE model loaded successfully from local path on device: {self.device}"
            )

        except Exception as e:
            print(f"Error loading BGE model from local path: {e}")
            raise RuntimeError(f"Failed to load BGE model from {self.model_path}: {e}")

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into embeddings using the BGE model.

        Args:
            texts: List of texts to encode

        Returns:
            Tensor of embeddings
        """
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # BGE model max length
            return_tensors="pt",
        )  # type: ignore

        # Move inputs to device
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)

            # Perform mean pooling on the token embeddings
            # Mask padding tokens for proper mean pooling
            attention_mask = encoded_input["attention_mask"]
            token_embeddings = model_output.last_hidden_state

            # Expand attention mask to match token embeddings dimensions
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )

            # Sum embeddings and divide by actual length (excluding padding)
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # Normalize embeddings for cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings

    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding for a single text.

        Args:
            text: Text to create an embedding for

        Returns:
            List of floats representing the embedding
        """
        try:
            if not text or not text.strip():
                # Return zero embedding for empty text
                return [0.0] * 1024  # BGE-large embedding dimension

            embeddings = self._encode_texts([text])
            return embeddings[0].cpu().tolist()

        except Exception as e:
            print(f"Error creating single embedding: {e}")
            # Return zero embedding as fallback
            return [0.0] * 1024

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts in a batch.

        Args:
            texts: List of texts to create embeddings for

        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        if not texts:
            return []

        max_retries = 3
        retry_delay = 1.0

        for retry in range(max_retries):
            try:
                # Filter out empty texts and keep track of original indices
                non_empty_texts = []
                text_indices = []

                for i, text in enumerate(texts):
                    if text and text.strip():
                        non_empty_texts.append(text)
                        text_indices.append(i)

                if not non_empty_texts:
                    # All texts are empty, return zero embeddings
                    return [[0.0] * 1024 for _ in texts]

                # Process in smaller batches to avoid memory issues
                batch_size = 32
                all_embeddings = []

                for i in range(0, len(non_empty_texts), batch_size):
                    batch_texts = non_empty_texts[i : i + batch_size]
                    batch_embeddings = self._encode_texts(batch_texts)
                    all_embeddings.extend(batch_embeddings.cpu().tolist())

                # Reconstruct full results array with zero embeddings for empty texts
                results = []
                non_empty_idx = 0

                for i, text in enumerate(texts):
                    if text and text.strip():
                        results.append(all_embeddings[non_empty_idx])
                        non_empty_idx += 1
                    else:
                        results.append([0.0] * 1024)

                return results

            except Exception as e:
                if retry < max_retries - 1:
                    print(
                        f"Error creating batch embeddings (attempt {retry + 1}/{max_retries}): {e}"
                    )
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(
                        f"Failed to create batch embeddings after {max_retries} attempts: {e}"
                    )
                    # Try creating embeddings one by one as fallback
                    print("Attempting to create embeddings individually...")
                    embeddings = []
                    successful_count = 0

                    for text in texts:
                        try:
                            embedding = self.create_embedding(text)
                            embeddings.append(embedding)
                            successful_count += 1
                        except Exception as individual_error:
                            print(
                                f"Failed to create individual embedding: {individual_error}"
                            )
                            # Add zero embedding as fallback
                            embeddings.append([0.0] * 1024)

                    print(
                        f"Successfully created {successful_count}/{len(texts)} embeddings individually"
                    )
                    return embeddings


# Global instance
_embedding_service: Optional[LocalEmbeddingService] = None


def get_embedding_service() -> LocalEmbeddingService:
    """
    Get the global embedding service instance.

    Returns:
        LocalEmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = LocalEmbeddingService()
    return _embedding_service


def create_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text using the local BGE model.

    Args:
        text: Text to create an embedding for

    Returns:
        List of floats representing the embedding
    """
    service = get_embedding_service()
    return service.create_embedding(text)


def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple texts in a batch using the local BGE model.

    Args:
        texts: List of texts to create embeddings for

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    service = get_embedding_service()
    return service.create_embeddings_batch(texts)
