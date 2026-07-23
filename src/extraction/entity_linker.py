"""
Entity Linker

Links extracted entity mentions to canonical entries in:
- UMLS Metathesaurus (CUI codes)
- DrugBank (drug IDs)
- IMPPAT / TKDL (herb IDs)

Uses a combination of:
1. Exact string matching
2. Fuzzy string matching (rapidfuzz)
3. Embedding-based semantic similarity
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class LinkedEntity:
    """An entity mention linked to a canonical knowledge base entry."""

    mention_text: str
    mention_type: str  # Drug, Herb, etc.
    canonical_id: str  # UMLS CUI, DrugBank ID, etc.
    canonical_name: str
    source_kb: str  # "umls", "drugbank", "imppat"
    similarity_score: float = 0.0
    match_method: str = ""  # "exact", "fuzzy", "embedding"
    alternative_matches: list[dict] = field(default_factory=list)


class EntityLinker:
    """
    Links entity mentions from NER to canonical knowledge base entries.

    The linking pipeline:
    1. Exact match against the canonical dictionary
    2. Fuzzy match using rapidfuzz (Levenshtein + token sort)
    3. Embedding similarity match using sentence transformers

    This handles the normalization challenge of code-mixed text,
    where herbs may be mentioned in Hindi, Romanized Hindi,
    or English common names.

    Usage:
        linker = EntityLinker()
        linker.load_drug_dictionary(drugbank_drugs_df)
        linker.load_herb_dictionary(imppat_herbs_df)
        result = linker.link("haldi", "Herb")
        # -> LinkedEntity(canonical_name="Turmeric", source_kb="imppat", ...)
    """

    def __init__(
        self,
        fuzzy_threshold: float = 85.0,
        embedding_threshold: float = 0.75,
        umls_api_key: Optional[str] = None,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.embedding_threshold = embedding_threshold
        self.umls_api_key = umls_api_key

        # Canonical dictionaries: name → (id, source_kb)
        self._drug_dict: dict[str, tuple[str, str]] = {}
        self._herb_dict: dict[str, tuple[str, str]] = {}
        self._all_names: dict[str, tuple[str, str, str]] = {}
        # name_lower → (canonical_name, id, source_kb)

        self._embedding_model = None

        # Pre-populate with common herb name aliases (Hindi/English)
        self._herb_aliases = {
            # Romanized Hindi → English canonical name
            "haldi": "Turmeric",
            "adrak": "Ginger",
            "lehsun": "Garlic",
            "tulsi": "Holy Basil",
            "neem": "Neem",
            "amla": "Indian Gooseberry",
            "ashwagandha": "Ashwagandha",
            "brahmi": "Bacopa",
            "shatavari": "Shatavari",
            "arjun": "Arjuna",
            "guduchi": "Guduchi",
            "giloy": "Guduchi",
            "guggul": "Guggul",
            "triphala": "Triphala",
            "mulethi": "Licorice",
            "jeera": "Cumin",
            "dalchini": "Cinnamon",
            "methi": "Fenugreek",
            "ajwain": "Carom Seeds",
            "kali mirch": "Black Pepper",
            "pipali": "Long Pepper",
            "elaichi": "Cardamom",
            "laung": "Clove",
            "jaiphal": "Nutmeg",
            "kesar": "Saffron",
            "aloe vera": "Aloe Vera",
            "ginkgo": "Ginkgo",
            "st john's wort": "St. John's Wort",
            "st. john's wort": "St. John's Wort",
            "echinacea": "Echinacea",
            "valerian": "Valerian",
            "ginseng": "Ginseng",
            "garlic": "Garlic",
            "turmeric": "Turmeric",
            "ginger": "Ginger",
        }

    def load_drug_dictionary(self, drugs_df) -> None:
        """
        Load drug names from a DrugBank DataFrame.

        Args:
            drugs_df: DataFrame with 'drugbank_id' and 'name' columns
        """
        for _, row in drugs_df.iterrows():
            name = str(row.get("name", "")).strip()
            db_id = str(row.get("drugbank_id", "")).strip()
            if name and db_id:
                self._drug_dict[name.lower()] = (db_id, "drugbank")
                self._all_names[name.lower()] = (name, db_id, "drugbank")

        logger.info(f"Loaded {len(self._drug_dict)} drugs into linker dictionary")

    def load_herb_dictionary(self, herbs_df) -> None:
        """
        Load herb names from an IMPPAT DataFrame.

        Args:
            herbs_df: DataFrame with 'plant_id', 'plant_name',
                      and optionally 'botanical_name' columns
        """
        for _, row in herbs_df.iterrows():
            name = str(row.get("plant_name", "")).strip()
            plant_id = str(row.get("plant_id", "")).strip()
            botanical = str(row.get("botanical_name", "")).strip()

            if name and plant_id:
                self._herb_dict[name.lower()] = (plant_id, "imppat")
                self._all_names[name.lower()] = (name, plant_id, "imppat")

            if botanical and plant_id:
                self._herb_dict[botanical.lower()] = (plant_id, "imppat")
                self._all_names[botanical.lower()] = (
                    botanical, plant_id, "imppat",
                )

        # Add aliases
        for alias, canonical in self._herb_aliases.items():
            canonical_lower = canonical.lower()
            if canonical_lower in self._herb_dict:
                herb_id, source = self._herb_dict[canonical_lower]
                self._herb_dict[alias.lower()] = (herb_id, source)
                self._all_names[alias.lower()] = (canonical, herb_id, source)

        logger.info(
            f"Loaded {len(self._herb_dict)} herb names (incl. aliases) "
            "into linker dictionary"
        )

    def link(
        self,
        mention: str,
        entity_type: str = "Drug",
        top_k: int = 3,
    ) -> LinkedEntity:
        """
        Link an entity mention to a canonical KB entry.

        Args:
            mention: The entity mention text (e.g., "haldi", "warfarin")
            entity_type: "Drug" or "Herb"
            top_k: Number of alternative matches to return

        Returns:
            LinkedEntity with the best match
        """
        mention_lower = mention.lower().strip()

        # 1. Exact match
        if entity_type == "Herb" and mention_lower in self._herb_dict:
            herb_id, source = self._herb_dict[mention_lower]
            canonical = self._all_names.get(
                mention_lower, (mention, herb_id, source)
            )
            return LinkedEntity(
                mention_text=mention,
                mention_type=entity_type,
                canonical_id=herb_id,
                canonical_name=canonical[0],
                source_kb=source,
                similarity_score=1.0,
                match_method="exact",
            )

        if entity_type == "Drug" and mention_lower in self._drug_dict:
            drug_id, source = self._drug_dict[mention_lower]
            canonical = self._all_names.get(
                mention_lower, (mention, drug_id, source)
            )
            return LinkedEntity(
                mention_text=mention,
                mention_type=entity_type,
                canonical_id=drug_id,
                canonical_name=canonical[0],
                source_kb=source,
                similarity_score=1.0,
                match_method="exact",
            )

        # Check aliases for herbs
        if mention_lower in self._herb_aliases:
            canonical_name = self._herb_aliases[mention_lower]
            canonical_lower = canonical_name.lower()
            if canonical_lower in self._herb_dict:
                herb_id, source = self._herb_dict[canonical_lower]
                return LinkedEntity(
                    mention_text=mention,
                    mention_type=entity_type,
                    canonical_id=herb_id,
                    canonical_name=canonical_name,
                    source_kb=source,
                    similarity_score=0.95,
                    match_method="alias",
                )

        # 2. Fuzzy match
        fuzzy_result = self._fuzzy_match(mention_lower, entity_type, top_k)
        if fuzzy_result and fuzzy_result.similarity_score >= self.fuzzy_threshold / 100:
            return fuzzy_result

        # 3. Embedding similarity (if fuzzy fails)
        embedding_result = self._embedding_match(mention, entity_type)
        if embedding_result:
            return embedding_result

        # No match found
        return LinkedEntity(
            mention_text=mention,
            mention_type=entity_type,
            canonical_id="",
            canonical_name=mention,
            source_kb="unknown",
            similarity_score=0.0,
            match_method="none",
        )

    def _fuzzy_match(
        self, mention: str, entity_type: str, top_k: int = 3
    ) -> Optional[LinkedEntity]:
        """Fuzzy string matching using rapidfuzz."""
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            logger.debug("rapidfuzz not installed, skipping fuzzy matching")
            return None

        # Choose dictionary based on entity type
        if entity_type == "Herb":
            candidates = self._herb_dict
        elif entity_type == "Drug":
            candidates = self._drug_dict
        else:
            candidates = {**self._drug_dict, **self._herb_dict}

        if not candidates:
            return None

        # Use token_sort_ratio for better matching of reordered names
        results = process.extract(
            mention,
            list(candidates.keys()),
            scorer=fuzz.token_sort_ratio,
            limit=top_k,
        )

        if not results:
            return None

        best_name, best_score, _ = results[0]

        if best_score >= self.fuzzy_threshold:
            cid, source = candidates[best_name]
            canonical = self._all_names.get(
                best_name, (best_name, cid, source)
            )

            alternatives = []
            for name, score, _ in results[1:]:
                alt_id, alt_source = candidates[name]
                alt_canonical = self._all_names.get(
                    name, (name, alt_id, alt_source)
                )
                alternatives.append(
                    {
                        "canonical_name": alt_canonical[0],
                        "canonical_id": alt_id,
                        "source_kb": alt_source,
                        "score": score / 100.0,
                    }
                )

            return LinkedEntity(
                mention_text=mention,
                mention_type=entity_type,
                canonical_id=cid,
                canonical_name=canonical[0],
                source_kb=source,
                similarity_score=best_score / 100.0,
                match_method="fuzzy",
                alternative_matches=alternatives,
            )

        return None

    def _embedding_match(
        self, mention: str, entity_type: str
    ) -> Optional[LinkedEntity]:
        """
        Embedding-based semantic similarity matching.

        Falls back to this when exact and fuzzy matching fail,
        which is common for code-mixed transliterated names.
        """
        # This would use sentence-transformers for embedding similarity
        # Placeholder for when the full pipeline is set up
        logger.debug(
            f"Embedding match not yet implemented for: {mention} ({entity_type})"
        )
        return None

    def batch_link(
        self,
        mentions: list[tuple[str, str]],
    ) -> list[LinkedEntity]:
        """
        Link multiple entity mentions.

        Args:
            mentions: List of (mention_text, entity_type) tuples

        Returns:
            List of LinkedEntity results
        """
        return [self.link(text, etype) for text, etype in mentions]
