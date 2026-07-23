"""
Relation Extraction Module

Extracts relations between entity pairs in biomedical text.
Handles negation and uncertainty detection to avoid false-positive
interaction claims.

Relation types: interacts_with, inhibits, induces, potentiates, no_relation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from loguru import logger


@dataclass
class RelationPrediction:
    """A predicted relation between two entities."""

    entity1_text: str
    entity1_type: str
    entity2_text: str
    entity2_type: str
    relation_type: str
    confidence: float = 0.0
    is_negated: bool = False
    is_uncertain: bool = False
    evidence_span: str = ""  # The sentence or span supporting this relation


DEFAULT_RELATION_LABELS = [
    "no_relation",
    "interacts_with",
    "inhibits",
    "induces",
    "potentiates",
]

# Negation and uncertainty cue words
NEGATION_CUES = {
    "no", "not", "nor", "neither", "never", "without", "absent",
    "deny", "denied", "denies", "lack", "lacking", "fail", "failed",
    "fails", "unable", "nahi", "naa", "mat", "na",  # Hindi negation cues
}

UNCERTAINTY_CUES = {
    "may", "might", "could", "possibly", "potentially", "likely",
    "unlikely", "suggest", "suggests", "suggested", "appear", "appears",
    "seem", "seems", "probable", "probably", "shayad", "ho sakta",
    "lagta hai", "shaayad",  # Hindi uncertainty cues
}


class RelationClassifier(nn.Module):
    """
    Sentence-level relation classifier.

    Takes entity pair marker representations from a transformer encoder
    and classifies the relation between them.

    Architecture:
    - Entity marker pooling (extract representations at [E1] and [E2] markers)
    - Concatenation of entity pair representations
    - MLP classification head
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_labels: int = len(DEFAULT_RELATION_LABELS),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels

        # Entity pair representation → relation classification
        # Input: concat of [CLS], entity1_repr, entity2_repr = 3 * hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(
        self,
        cls_repr: torch.Tensor,
        entity1_repr: torch.Tensor,
        entity2_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_repr: (batch, hidden_size) CLS token representation
            entity1_repr: (batch, hidden_size) entity 1 representation
            entity2_repr: (batch, hidden_size) entity 2 representation

        Returns:
            logits: (batch, num_labels)
        """
        combined = torch.cat([cls_repr, entity1_repr, entity2_repr], dim=-1)
        return self.classifier(combined)


class RelationExtractor(nn.Module):
    """
    End-to-end relation extraction between entity pairs.

    Combines a transformer encoder with entity marker pooling and
    a relation classification head. Also detects negation and
    uncertainty modifiers.

    Usage:
        extractor = RelationExtractor()
        predictions = extractor.predict(
            text="Warfarin interacts with St. John's Wort",
            entity1=("Warfarin", "Drug", 0, 8),
            entity2=("St. John's Wort", "Herb", 25, 40),
        )
    """

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        num_labels: int = len(DEFAULT_RELATION_LABELS),
        label_list: Optional[list[str]] = None,
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.label_list = label_list or DEFAULT_RELATION_LABELS
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
        self.label2id = {l: i for i, l in enumerate(self.label_list)}

        self._encoder = None
        self._tokenizer = None
        self._encoder_hidden_size = 768
        self._initialized = False

        self.relation_head = RelationClassifier(
            hidden_size=self._encoder_hidden_size,
            num_labels=num_labels,
            dropout=dropout,
        )

        logger.info(
            f"RelationExtractor configured with {model_name}, "
            f"{num_labels} relation types"
        )

    def _lazy_init(self) -> None:
        """Initialize transformer on first use."""
        if self._initialized:
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading RE encoder: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._encoder = AutoModel.from_pretrained(self.model_name)

            # Add entity marker tokens
            special_tokens = {
                "additional_special_tokens": [
                    "[E1]", "[/E1]", "[E2]", "[/E2]",
                ]
            }
            self._tokenizer.add_special_tokens(special_tokens)
            self._encoder.resize_token_embeddings(len(self._tokenizer))

            self._encoder_hidden_size = self._encoder.config.hidden_size
            self.relation_head = RelationClassifier(
                hidden_size=self._encoder_hidden_size,
                num_labels=len(self.label_list),
            )

        except Exception as e:
            logger.warning(f"Could not load transformer: {e}. Using fallback.")

        self._initialized = True

    def _mark_entities(
        self,
        text: str,
        entity1: tuple[str, str, int, int],
        entity2: tuple[str, str, int, int],
    ) -> str:
        """
        Insert entity markers around entity spans.

        Args:
            text: Original text
            entity1: (text, type, start_char, end_char)
            entity2: (text, type, start_char, end_char)

        Returns:
            Text with [E1]/[/E1] and [E2]/[/E2] markers
        """
        e1_text, _, e1_start, e1_end = entity1
        e2_text, _, e2_start, e2_end = entity2

        # Insert markers (handle overlapping spans by sorting)
        if e1_start <= e2_start:
            first, second = (e1_start, e1_end, "[E1]", "[/E1]"), (
                e2_start, e2_end, "[E2]", "[/E2]",
            )
        else:
            first, second = (e2_start, e2_end, "[E2]", "[/E2]"), (
                e1_start, e1_end, "[E1]", "[/E1]",
            )

        # Insert from right to left to preserve offsets
        marked = (
            text[: first[0]]
            + first[2] + " "
            + text[first[0] : first[1]]
            + " " + first[3]
            + text[first[1] : second[0]]
            + second[2] + " "
            + text[second[0] : second[1]]
            + " " + second[3]
            + text[second[1] :]
        )

        return marked

    @staticmethod
    def detect_negation(text: str) -> bool:
        """Check if the text contains negation cues."""
        tokens = text.lower().split()
        return bool(NEGATION_CUES.intersection(tokens))

    @staticmethod
    def detect_uncertainty(text: str) -> bool:
        """Check if the text contains uncertainty cues."""
        tokens = text.lower().split()
        text_lower = text.lower()
        # Check single-word cues
        if UNCERTAINTY_CUES.intersection(tokens):
            return True
        # Check multi-word cues
        for cue in UNCERTAINTY_CUES:
            if " " in cue and cue in text_lower:
                return True
        return False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_mask: torch.Tensor,
        e2_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for relation extraction.

        Args:
            input_ids: (batch, seq_len) token IDs with entity markers
            attention_mask: (batch, seq_len) attention mask
            e1_mask: (batch, seq_len) mask for entity 1 tokens
            e2_mask: (batch, seq_len) mask for entity 2 tokens
            labels: (batch,) optional gold relation labels

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        self._lazy_init()

        if self._encoder is not None:
            outputs = self._encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state
        else:
            batch_size, seq_len = input_ids.shape
            hidden_states = torch.randn(
                batch_size, seq_len, self._encoder_hidden_size,
                device=input_ids.device,
            )

        # Pool CLS representation
        cls_repr = hidden_states[:, 0, :]  # (batch, hidden)

        # Pool entity representations using masks
        e1_repr = self._masked_mean(hidden_states, e1_mask)
        e2_repr = self._masked_mean(hidden_states, e2_mask)

        logits = self.relation_head(cls_repr, e1_repr, e2_repr)

        result = {"logits": logits}
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            result["loss"] = loss_fn(logits, labels)

        return result

    @staticmethod
    def _masked_mean(
        hidden_states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Average pool hidden states where mask is 1."""
        mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        summed = (hidden_states * mask).sum(dim=1)  # (batch, hidden)
        counts = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        return summed / counts

    def predict(
        self,
        text: str,
        entity1: tuple[str, str, int, int],
        entity2: tuple[str, str, int, int],
    ) -> RelationPrediction:
        """
        Predict the relation between two entities in text.

        Args:
            text: Input text
            entity1: (text, type, start_char, end_char)
            entity2: (text, type, start_char, end_char)

        Returns:
            RelationPrediction with type, confidence, negation/uncertainty flags
        """
        self._lazy_init()
        self.eval()

        # Detect negation and uncertainty
        is_negated = self.detect_negation(text)
        is_uncertain = self.detect_uncertainty(text)

        if self._tokenizer is None:
            return RelationPrediction(
                entity1_text=entity1[0],
                entity1_type=entity1[1],
                entity2_text=entity2[0],
                entity2_type=entity2[1],
                relation_type="no_relation",
                confidence=0.0,
                is_negated=is_negated,
                is_uncertain=is_uncertain,
                evidence_span=text,
            )

        # Mark entities in text
        marked_text = self._mark_entities(text, entity1, entity2)

        # Tokenize
        encoding = self._tokenizer(
            marked_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )

        # Create entity masks
        e1_start_id = self._tokenizer.convert_tokens_to_ids("[E1]")
        e2_start_id = self._tokenizer.convert_tokens_to_ids("[E2]")

        input_ids = encoding["input_ids"][0]
        e1_mask = torch.zeros_like(input_ids)
        e2_mask = torch.zeros_like(input_ids)

        in_e1, in_e2 = False, False
        for i, tid in enumerate(input_ids.tolist()):
            if tid == e1_start_id:
                in_e1 = True
                continue
            if tid == e2_start_id:
                in_e2 = True
                continue
            if in_e1:
                e1_mask[i] = 1
            if in_e2:
                e2_mask[i] = 1

        with torch.no_grad():
            outputs = self.forward(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                e1_mask=e1_mask.unsqueeze(0),
                e2_mask=e2_mask.unsqueeze(0),
            )

        logits = outputs["logits"][0]
        probs = torch.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs).item()
        confidence = probs[pred_id].item()

        relation_type = self.id2label.get(pred_id, "no_relation")

        # If negated, flip to no_relation or reduce confidence
        if is_negated and relation_type != "no_relation":
            confidence *= 0.3  # Heavily penalize negated claims

        return RelationPrediction(
            entity1_text=entity1[0],
            entity1_type=entity1[1],
            entity2_text=entity2[0],
            entity2_type=entity2[1],
            relation_type=relation_type,
            confidence=confidence,
            is_negated=is_negated,
            is_uncertain=is_uncertain,
            evidence_span=text,
        )
