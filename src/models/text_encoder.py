"""
Text Encoder

Encodes biomedical text (formal and code-mixed) into fixed-dimension
embeddings for the cross-attention fusion layer.

Uses pretrained transformer models:
- PubMedBERT for formal biomedical text
- IndicBERT/MuRIL for code-mixed Hindi-English text
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


class TextEncoder(nn.Module):
    """
    Biomedical text encoder producing fixed-dimension embeddings.

    Wraps a pretrained transformer with a projection head to produce
    embeddings compatible with the GNN output dimension.

    Usage:
        encoder = TextEncoder(model_name="...", output_dim=128)
        embeddings = encoder(input_ids, attention_mask)  # (batch, output_dim)
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        output_dim: int = 128,
        pooling: str = "cls",  # "cls", "mean", "max"
        freeze_layers: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.output_dim = output_dim
        self.pooling = pooling
        self._freeze_layers = freeze_layers

        self._encoder = None
        self._tokenizer = None
        self._hidden_size = 768  # Default, updated on load
        self._initialized = False

        # Projection from transformer hidden to output dim
        self.projection = nn.Sequential(
            nn.Linear(self._hidden_size, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        logger.info(
            f"TextEncoder: {model_name}, pool={pooling}, "
            f"out_dim={output_dim}, freeze={freeze_layers}"
        )

    def _lazy_init(self) -> None:
        """Load transformer model on first use."""
        if self._initialized:
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading text encoder: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._encoder = AutoModel.from_pretrained(self.model_name)
            self._hidden_size = self._encoder.config.hidden_size

            # Rebuild projection with correct hidden size
            self.projection = nn.Sequential(
                nn.Linear(self._hidden_size, self.output_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.output_dim, self.output_dim),
            )

            # Freeze early layers
            if self._freeze_layers > 0 and hasattr(self._encoder, "encoder"):
                for i, layer in enumerate(self._encoder.encoder.layer):
                    if i < self._freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

        except Exception as e:
            logger.warning(
                f"Could not load transformer: {e}. "
                "Using random projections for development."
            )

        self._initialized = True

    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool transformer hidden states to a single vector.

        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)

        Returns:
            pooled: (batch, hidden_size)
        """
        if self.pooling == "cls":
            return hidden_states[:, 0, :]  # CLS token

        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            return summed / counts

        elif self.pooling == "max":
            mask = attention_mask.unsqueeze(-1).float()
            masked = hidden_states * mask + (1 - mask) * (-1e9)
            return masked.max(dim=1).values

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_list: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """
        Encode text into fixed-dimension embeddings.

        Args:
            input_ids: (batch, seq_len) pre-tokenized input
            attention_mask: (batch, seq_len) attention mask
            text_list: Alternative: list of raw text strings to tokenize

        Returns:
            embeddings: (batch, output_dim)
        """
        self._lazy_init()

        # If raw text provided, tokenize it
        if text_list is not None and self._tokenizer is not None:
            encoding = self._tokenizer(
                text_list,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True,
            )
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]

            # Move to same device as model
            device = next(self.projection.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        if input_ids is None:
            raise ValueError("Must provide either input_ids or text_list")

        # Get transformer hidden states
        if self._encoder is not None:
            with torch.set_grad_enabled(self.training):
                outputs = self._encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.last_hidden_state
        else:
            # Fallback: random hidden states
            batch_size, seq_len = input_ids.shape
            hidden_states = torch.randn(
                batch_size, seq_len, self._hidden_size,
                device=input_ids.device,
            )

        # Pool to single vector
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape, device=input_ids.device
            )
        pooled = self._pool(hidden_states, attention_mask)

        # Project to output dimension
        embeddings = self.projection(pooled)

        return embeddings

    def get_tokenizer(self):
        """Get the tokenizer for external use."""
        self._lazy_init()
        return self._tokenizer

    @classmethod
    def for_formal_text(cls, output_dim: int = 128, **kwargs) -> "TextEncoder":
        """Create encoder for formal biomedical text."""
        return cls(
            model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            output_dim=output_dim,
            **kwargs,
        )

    @classmethod
    def for_code_mixed(cls, output_dim: int = 128, **kwargs) -> "TextEncoder":
        """Create encoder for code-mixed Hindi-English text."""
        return cls(
            model_name="ai4bharat/IndicBERTv2-MLM-Sam-TLM",
            output_dim=output_dim,
            **kwargs,
        )
