"""
Named Entity Recognition Model

Token-level NER for biomedical entities in both formal and code-mixed text.
Supports:
- PubMedBERT / BioBERT for formal biomedical text
- IndicBERT / MuRIL for code-mixed Hindi-English text
- BIO tagging scheme

Entity types: Drug, Herb, Effect, Mechanism, Dosage, Disease
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class NERPrediction:
    """A single named entity prediction."""

    text: str
    entity_type: str  # Drug, Herb, Effect, Mechanism, Dosage, Disease
    start_char: int
    end_char: int
    confidence: float = 0.0
    token_indices: list[int] = field(default_factory=list)


# BIO tag mapping
DEFAULT_NER_LABELS = [
    "O",
    "B-Drug", "I-Drug",
    "B-Herb", "I-Herb",
    "B-Effect", "I-Effect",
    "B-Mechanism", "I-Mechanism",
    "B-Dosage", "I-Dosage",
    "B-Disease", "I-Disease",
]


class NERHead(nn.Module):
    """
    Token classification head for NER.

    Takes hidden states from a transformer encoder and predicts
    BIO tags for each token.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = len(DEFAULT_NER_LABELS),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size) from transformer
            attention_mask: (batch, seq_len) mask for padding tokens

        Returns:
            logits: (batch, seq_len, num_labels)
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits


class NERModel(nn.Module):
    """
    Full NER model combining a pretrained transformer encoder
    with a token classification head.

    Supports two encoder modes:
    1. Formal biomedical text: PubMedBERT / BioBERT
    2. Code-mixed text: IndicBERT / MuRIL

    Usage:
        model = NERModel(model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        predictions = model.predict("Warfarin interacts with St. John's Wort")
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        num_labels: int = len(DEFAULT_NER_LABELS),
        label_list: Optional[list[str]] = None,
        max_length: int = 512,
        freeze_encoder_layers: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.label_list = label_list or DEFAULT_NER_LABELS
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
        self.label2id = {l: i for i, l in enumerate(self.label_list)}

        # Lazy-load transformer to avoid import overhead
        self._encoder = None
        self._tokenizer = None
        self._encoder_hidden_size = 768  # Default, updated on load

        # Classification head (can be initialized before encoder loads)
        self.ner_head = NERHead(
            hidden_size=self._encoder_hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )

        self._freeze_layers = freeze_encoder_layers
        self._initialized = False

        logger.info(
            f"NERModel configured with {model_name}, "
            f"{num_labels} labels, freeze={freeze_encoder_layers}"
        )

    def _lazy_init(self) -> None:
        """Initialize transformer encoder and tokenizer on first use."""
        if self._initialized:
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading NER encoder: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._encoder = AutoModel.from_pretrained(self.model_name)

            # Update hidden size from actual model config
            self._encoder_hidden_size = self._encoder.config.hidden_size
            self.ner_head = NERHead(
                hidden_size=self._encoder_hidden_size,
                num_labels=len(self.label_list),
            )

            # Freeze early encoder layers if specified
            if self._freeze_layers > 0 and hasattr(
                self._encoder, "encoder"
            ):
                layers = self._encoder.encoder.layer
                for i, layer in enumerate(layers):
                    if i < self._freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                logger.info(
                    f"Froze first {self._freeze_layers} encoder layers"
                )

            self._initialized = True

        except Exception as e:
            logger.warning(
                f"Could not load transformer model: {e}. "
                "Using random initialization for development."
            )
            self._encoder_hidden_size = 768
            self._initialized = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for NER.

        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) attention mask
            token_type_ids: (batch, seq_len) optional segment IDs
            labels: (batch, seq_len) optional gold BIO tag IDs

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        self._lazy_init()

        if self._encoder is not None:
            encoder_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                encoder_kwargs["token_type_ids"] = token_type_ids

            outputs = self._encoder(**encoder_kwargs)
            hidden_states = outputs.last_hidden_state
        else:
            # Fallback: random hidden states for testing
            batch_size, seq_len = input_ids.shape
            hidden_states = torch.randn(
                batch_size, seq_len, self._encoder_hidden_size,
                device=input_ids.device,
            )

        logits = self.ner_head(hidden_states, attention_mask)

        result = {"logits": logits}

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                logits.view(-1, len(self.label_list)),
                labels.view(-1),
            )
            result["loss"] = loss

        return result

    def predict(self, text: str) -> list[NERPrediction]:
        """
        Run NER on a single text string.

        Args:
            text: Input text to extract entities from.

        Returns:
            List of NERPrediction objects with entity spans and types.
        """
        self._lazy_init()
        self.eval()

        if self._tokenizer is None:
            logger.warning("Tokenizer not available. Returning empty predictions.")
            return []

        # Tokenize
        encoding = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
        )

        offset_mapping = encoding.pop("offset_mapping")[0]  # (seq_len, 2)

        with torch.no_grad():
            outputs = self.forward(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )

        logits = outputs["logits"][0]  # (seq_len, num_labels)
        predictions = torch.argmax(logits, dim=-1)  # (seq_len,)
        confidences = torch.softmax(logits, dim=-1).max(dim=-1).values

        # Decode BIO tags into entity spans
        entities = []
        current_entity = None

        for idx, (pred_id, (start, end)) in enumerate(
            zip(predictions.tolist(), offset_mapping.tolist())
        ):
            if start == 0 and end == 0:  # Special token
                continue

            label = self.id2label.get(pred_id, "O")
            conf = confidences[idx].item()

            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity:
                    entities.append(current_entity)

                entity_type = label[2:]
                current_entity = NERPrediction(
                    text=text[start:end],
                    entity_type=entity_type,
                    start_char=start,
                    end_char=end,
                    confidence=conf,
                    token_indices=[idx],
                )

            elif label.startswith("I-") and current_entity:
                entity_type = label[2:]
                if entity_type == current_entity.entity_type:
                    # Extend current entity
                    current_entity.end_char = end
                    current_entity.text = text[
                        current_entity.start_char : end
                    ]
                    current_entity.token_indices.append(idx)
                    current_entity.confidence = min(
                        current_entity.confidence, conf
                    )
                else:
                    # Type mismatch — save and start new
                    entities.append(current_entity)
                    current_entity = None

            else:  # O tag
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget last entity
        if current_entity:
            entities.append(current_entity)

        return entities

    @classmethod
    def for_formal_text(cls, **kwargs) -> "NERModel":
        """Create NER model configured for formal biomedical text."""
        return cls(
            model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            **kwargs,
        )

    @classmethod
    def for_code_mixed_text(cls, **kwargs) -> "NERModel":
        """Create NER model configured for code-mixed Hindi-English text."""
        return cls(
            model_name="ai4bharat/IndicBERTv2-MLM-Sam-TLM",
            **kwargs,
        )
