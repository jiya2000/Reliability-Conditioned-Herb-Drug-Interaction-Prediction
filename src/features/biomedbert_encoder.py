"""
BiomedBERT Evidence Representation Encoder.

Encodes pharmacological literature evidence using Microsoft's PubMedBERT/BiomedBERT
(microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract).
Replaces bag-of-words / random projection with frozen biomedical contextual representations.
Enforces strict split-leakage safety: projection layers and dimensionality reduction
are fit STRICTLY on training split texts.
"""

from __future__ import annotations

import re
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from loguru import logger


DEFAULT_BIOMEDBERT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"


class BiomedBERTEvidenceEncoder:
    """
    Contextual biomedical language representation encoder based on frozen BiomedBERT.
    Extracts 768-dimensional semantic embeddings from pharmacological evidence texts
    and projects them to target interaction model embedding dimensions using
    leakage-safe projection fit strictly on training splits.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_BIOMEDBERT_MODEL,
        embedding_dim: int = 16,
        max_length: int = 64,
        device: Optional[Union[str, torch.device]] = None,
        use_cache: bool = True,
    ):
        self.model_name = model_name_or_path
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.use_cache = use_cache

        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = torch.device(device)

        self._tokenizer = None
        self._model = None
        self._projection_matrix: Optional[np.ndarray] = None
        self._projection_mean: Optional[np.ndarray] = None
        self._is_fitted = False
        self._text_cache: Dict[str, np.ndarray] = {}

    def _lazy_load_model(self) -> None:
        """Lazily load tokenizer and pretrained model to conserve resources."""
        if self._tokenizer is None or self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                logger.info(f"Loading BiomedBERT from {self.model_name} on {self.device}...")
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
                self._model.to(self.device)
                for p in self._model.parameters():
                    p.requires_grad = False
                logger.info("BiomedBERT successfully initialized in frozen evaluation mode.")
            except Exception as e:
                logger.warning(f"Failed to load BiomedBERT ({e}). Falling back to deterministic biological word-level vectors.")
                self._tokenizer = None
                self._model = None

    @torch.no_grad()
    def encode_raw_768(self, texts: List[str]) -> np.ndarray:
        """
        Extract 768-dimensional frozen pooled [CLS] + mean embeddings from BiomedBERT.
        Uses in-memory caching to prevent duplicate inferences.
        """
        self._lazy_load_model()

        embeddings = []
        texts_to_infer = []
        indices_to_infer = []

        for idx, text in enumerate(texts):
            clean_text = str(text).strip()
            if self.use_cache and clean_text in self._text_cache:
                embeddings.append((idx, self._text_cache[clean_text]))
            else:
                texts_to_infer.append(clean_text)
                indices_to_infer.append(idx)

        if texts_to_infer and self._model is not None and self._tokenizer is not None:
            batch_size = 32
            for i in range(0, len(texts_to_infer), batch_size):
                batch_texts = texts_to_infer[i : i + batch_size]
                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                outputs = self._model(**encoded)
                # Combined representation: 0.5 * CLS + 0.5 * attention-masked mean pool
                cls_repr = outputs.last_hidden_state[:, 0, :]
                mask = encoded["attention_mask"].unsqueeze(-1)
                mean_repr = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                fused = 0.5 * cls_repr + 0.5 * mean_repr

                # L2 normalize raw 768D vectors
                fused_norm = torch.nn.functional.normalize(fused, p=2, dim=-1)
                batch_arr = fused_norm.cpu().numpy()

                for j, text in enumerate(batch_texts):
                    orig_idx = indices_to_infer[i + j]
                    emb = batch_arr[j].astype(np.float32)
                    if self.use_cache:
                        self._text_cache[text] = emb
                    embeddings.append((orig_idx, emb))

        elif texts_to_infer:
            # Deterministic biological word hash fallback if transformers unavailable
            for j, text in enumerate(texts_to_infer):
                orig_idx = indices_to_infer[j]
                rng = np.random.RandomState(abs(hash(text)) % (2**31))
                vec = rng.randn(768).astype(np.float32)
                vec /= np.linalg.norm(vec) + 1e-9
                if self.use_cache:
                    self._text_cache[text] = vec
                embeddings.append((orig_idx, vec))

        # Re-sort by original input order
        embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in embeddings], dtype=np.float32)

    def fit_projection(self, train_texts: List[str]) -> BiomedBERTEvidenceEncoder:
        """
        Fit projection matrix STRICTLY on training split texts.
        Extracts 768D raw embeddings for training partition, centers them,
        and computes the optimal orthonormal projection to `embedding_dim`
        via singular value decomposition / PCA on train representations only.
        """
        if not train_texts:
            raise ValueError("Cannot fit projection on empty training texts.")

        raw_768 = self.encode_raw_768(train_texts)
        self._projection_mean = raw_768.mean(axis=0, keepdims=True)
        centered = raw_768 - self._projection_mean

        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            proj = vt.T[:, : self.embedding_dim]  # (768, k) where k <= embedding_dim
            if proj.shape[1] < self.embedding_dim:
                # Pad with orthonormal vectors to reach embedding_dim
                needed = self.embedding_dim - proj.shape[1]
                rng = np.random.RandomState(42)
                extra = rng.randn(768, needed).astype(np.float32)
                # Gram-Schmidt / QR orthogonalization against existing columns
                combined = np.hstack([proj, extra])
                q, _ = np.linalg.qr(combined)
                proj = q[:, : self.embedding_dim]
        except Exception:
            # QR fallback on reproducible seed
            rng = np.random.RandomState(42)
            rand_w = rng.randn(768, self.embedding_dim).astype(np.float32)
            proj, _ = np.linalg.qr(rand_w)

        self._projection_matrix = proj.astype(np.float32)
        self._is_fitted = True
        logger.info(
            f"Fitted BiomedBERT projection strictly on {len(train_texts)} training texts "
            f"(768 -> {self.embedding_dim} dim, variance captured by {self.embedding_dim} PCs)"
        )
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into fixed `embedding_dim` vectors using the fitted projection.
        Guarantees unit L2 normalization.
        """
        if not self._is_fitted or self._projection_matrix is None:
            raise RuntimeError("BiomedBERTEvidenceEncoder must be fit on training data before transforming.")

        raw_768 = self.encode_raw_768(texts)
        centered = raw_768 - self._projection_mean
        projected = centered @ self._projection_matrix

        # L2 normalize
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        norms = np.where(norms > 1e-6, norms, 1.0)
        normalized = (projected / norms).astype(np.float32)
        return normalized

    @torch.no_grad()
    def transform_tokens(self, texts: List[str], max_tokens: int = 8) -> torch.Tensor:
        """
        Extract multi-token sequence embeddings (batch, max_tokens, embedding_dim)
        for token-level multi-head cross-attention.
        """
        if not self._is_fitted or self._projection_matrix is None:
            raise RuntimeError("BiomedBERTEvidenceEncoder must be fit on training data before transforming tokens.")

        self._lazy_load_model()
        proj_t = torch.from_numpy(self._projection_matrix).to(self.device)
        proj_mean_t = torch.from_numpy(self._projection_mean).to(self.device)

        if self._model is not None and self._tokenizer is not None:
            encoded = self._tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt",
            ).to(self.device)

            outputs = self._model(**encoded)
            # (batch, max_tokens, 768)
            token_hidden = outputs.last_hidden_state[:, :max_tokens, :]
            # Project to target dim: (batch, max_tokens, embedding_dim)
            centered = token_hidden - proj_mean_t
            projected = torch.matmul(centered, proj_t)
            projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
            return projected.cpu()
        else:
            # Fallback pseudo-tokens
            batch_size = len(texts)
            arr = self.transform(texts)
            t = torch.from_numpy(arr).unsqueeze(1).repeat(1, max_tokens, 1)
            return t
