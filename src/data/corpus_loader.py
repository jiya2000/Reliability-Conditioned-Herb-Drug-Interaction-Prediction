"""
Biomedical Corpus Loaders

Loads annotated text corpora for NER/RE training:
- DDI Corpus: Drug-drug interaction sentences (PubMed + DrugBank)
- CADEC: Consumer adverse drug event corpus
- PsyTAR: Psychiatric drug adverse event reviews
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class AnnotatedSentence:
    """A sentence with entity and relation annotations."""

    text: str
    source_corpus: str  # "ddi", "cadec", "psytar"
    doc_id: str = ""
    sentence_id: str = ""
    entities: list[dict] = field(default_factory=list)
    # Each entity: {"text": str, "type": str, "start": int, "end": int, "id": str}
    relations: list[dict] = field(default_factory=list)
    # Each relation: {"type": str, "entity1_id": str, "entity2_id": str}
    is_formal: bool = True  # True for PubMed text, False for consumer text


class CorpusLoader:
    """
    Unified loader for biomedical text corpora.

    Provides annotated sentences for training NER and relation extraction
    models. Handles three corpora with different formats:

    - DDI Corpus (XML): Drug-drug interactions from PubMed abstracts
      and DrugBank text. Entities: drug, group, brand, drug_n.
      Relations: mechanism, effect, advise, int.

    - CADEC (BRAT format): Consumer adverse drug events from
      askapatient.com. Entities: Drug, ADR, Disease, Symptom, Finding.

    - PsyTAR (TSV): Psychiatric drug reviews. Entities: Drug, ADR.

    Usage:
        loader = CorpusLoader(ddi_path="data/raw/ddi_corpus",
                              cadec_path="data/raw/cadec",
                              psytar_path="data/raw/psytar")
        sentences = loader.load_all()
        formal_df = loader.load_formal_text()
        informal_df = loader.load_informal_text()
    """

    def __init__(
        self,
        ddi_path: Optional[str | Path] = None,
        cadec_path: Optional[str | Path] = None,
        psytar_path: Optional[str | Path] = None,
    ):
        self.ddi_path = Path(ddi_path) if ddi_path else None
        self.cadec_path = Path(cadec_path) if cadec_path else None
        self.psytar_path = Path(psytar_path) if psytar_path else None
        self._sentences: list[AnnotatedSentence] = []
        self._parsed = False

    def _load(self) -> None:
        """Load all corpora."""
        if self._parsed:
            return

        loaded_any = False

        if self.ddi_path and self.ddi_path.exists():
            self._load_ddi_corpus()
            loaded_any = True
        elif self.ddi_path:
            logger.warning(f"DDI Corpus not found at {self.ddi_path}")

        if self.cadec_path and self.cadec_path.exists():
            self._load_cadec_corpus()
            loaded_any = True
        elif self.cadec_path:
            logger.warning(f"CADEC not found at {self.cadec_path}")

        if self.psytar_path and self.psytar_path.exists():
            self._load_psytar_corpus()
            loaded_any = True
        elif self.psytar_path:
            logger.warning(f"PsyTAR not found at {self.psytar_path}")

        if not loaded_any:
            logger.info("No corpora found. Generating synthetic examples.")
            self._generate_synthetic_data()

        self._parsed = True
        logger.info(f"Total annotated sentences loaded: {len(self._sentences)}")

    def _load_ddi_corpus(self) -> None:
        """Load DDI Corpus from XML files."""
        import xml.etree.ElementTree as ET

        logger.info(f"Loading DDI Corpus from: {self.ddi_path}")

        for xml_file in self.ddi_path.rglob("*.xml"):
            try:
                tree = ET.parse(str(xml_file))
                root = tree.getroot()

                for doc in root.iter("document"):
                    doc_id = doc.attrib.get("id", "")
                    for sent in doc.iter("sentence"):
                        sent_id = sent.attrib.get("id", "")
                        text = sent.attrib.get("text", "")
                        if not text:
                            continue

                        entities = []
                        for ent in sent.iter("entity"):
                            char_offset = ent.attrib.get("charOffset", "")
                            start, end = 0, 0
                            if "-" in char_offset:
                                parts = char_offset.split("-")
                                start = int(parts[0])
                                end = int(parts[1])

                            entities.append(
                                {
                                    "id": ent.attrib.get("id", ""),
                                    "text": ent.attrib.get("text", ""),
                                    "type": ent.attrib.get("type", ""),
                                    "start": start,
                                    "end": end,
                                }
                            )

                        relations = []
                        for pair in sent.iter("pair"):
                            rel_type = pair.attrib.get("type", "")
                            if pair.attrib.get("ddi", "false") == "true":
                                relations.append(
                                    {
                                        "type": rel_type
                                        if rel_type
                                        else "interaction",
                                        "entity1_id": pair.attrib.get("e1", ""),
                                        "entity2_id": pair.attrib.get("e2", ""),
                                    }
                                )

                        self._sentences.append(
                            AnnotatedSentence(
                                text=text,
                                source_corpus="ddi",
                                doc_id=doc_id,
                                sentence_id=sent_id,
                                entities=entities,
                                relations=relations,
                                is_formal=True,
                            )
                        )
            except ET.ParseError as e:
                logger.warning(f"Failed to parse {xml_file}: {e}")

        ddi_count = sum(1 for s in self._sentences if s.source_corpus == "ddi")
        logger.info(f"Loaded {ddi_count} DDI Corpus sentences")

    def _load_cadec_corpus(self) -> None:
        """Load CADEC corpus from BRAT-format annotation files."""
        logger.info(f"Loading CADEC from: {self.cadec_path}")

        for txt_file in self.cadec_path.rglob("*.txt"):
            ann_file = txt_file.with_suffix(".ann")
            if not ann_file.exists():
                continue

            text = txt_file.read_text(encoding="utf-8").strip()
            entities = []

            for line in ann_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("T"):
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        ent_id = parts[0]
                        type_offset = parts[1].split()
                        ent_text = parts[2]
                        ent_type = type_offset[0]
                        start = int(type_offset[1]) if len(type_offset) > 1 else 0
                        end = int(type_offset[2]) if len(type_offset) > 2 else 0
                        entities.append(
                            {
                                "id": ent_id,
                                "text": ent_text,
                                "type": ent_type,
                                "start": start,
                                "end": end,
                            }
                        )

            self._sentences.append(
                AnnotatedSentence(
                    text=text,
                    source_corpus="cadec",
                    doc_id=txt_file.stem,
                    entities=entities,
                    is_formal=False,  # Consumer text
                )
            )

        cadec_count = sum(
            1 for s in self._sentences if s.source_corpus == "cadec"
        )
        logger.info(f"Loaded {cadec_count} CADEC documents")

    def _load_psytar_corpus(self) -> None:
        """Load PsyTAR corpus from TSV files."""
        logger.info(f"Loading PsyTAR from: {self.psytar_path}")

        for tsv_file in self.psytar_path.rglob("*.tsv"):
            try:
                df = pd.read_csv(tsv_file, sep="\t")
                for idx, row in df.iterrows():
                    text = str(row.get("review_text", row.get("text", "")))
                    if not text:
                        continue

                    entities = []
                    if "drug" in row:
                        entities.append(
                            {
                                "id": f"T{idx}_drug",
                                "text": str(row["drug"]),
                                "type": "Drug",
                                "start": 0,
                                "end": 0,
                            }
                        )

                    self._sentences.append(
                        AnnotatedSentence(
                            text=text,
                            source_corpus="psytar",
                            doc_id=tsv_file.stem,
                            sentence_id=str(idx),
                            entities=entities,
                            is_formal=False,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to load {tsv_file}: {e}")

        psytar_count = sum(
            1 for s in self._sentences if s.source_corpus == "psytar"
        )
        logger.info(f"Loaded {psytar_count} PsyTAR sentences")

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic annotated sentences for development."""
        logger.info("Generating synthetic corpus data...")

        # Formal text examples (PubMed-style)
        formal_examples = [
            {
                "text": "Concomitant administration of warfarin and St. John's Wort "
                "resulted in a significant decrease in INR values.",
                "entities": [
                    {"id": "T1", "text": "warfarin", "type": "Drug", "start": 35, "end": 43},
                    {"id": "T2", "text": "St. John's Wort", "type": "Herb", "start": 48, "end": 63},
                ],
                "relations": [
                    {"type": "interacts_with", "entity1_id": "T1", "entity2_id": "T2"}
                ],
            },
            {
                "text": "Curcumin, the active compound in turmeric, inhibits CYP3A4 "
                "and may increase plasma concentrations of cyclosporine.",
                "entities": [
                    {"id": "T1", "text": "Curcumin", "type": "Herb", "start": 0, "end": 8},
                    {"id": "T2", "text": "turmeric", "type": "Herb", "start": 33, "end": 41},
                    {"id": "T3", "text": "CYP3A4", "type": "Mechanism", "start": 52, "end": 58},
                    {"id": "T4", "text": "cyclosporine", "type": "Drug", "start": 101, "end": 113},
                ],
                "relations": [
                    {"type": "inhibits", "entity1_id": "T1", "entity2_id": "T3"},
                    {"type": "interacts_with", "entity1_id": "T1", "entity2_id": "T4"},
                ],
            },
            {
                "text": "Ginkgo biloba extract may potentiate the antiplatelet effect "
                "of aspirin, increasing bleeding risk.",
                "entities": [
                    {"id": "T1", "text": "Ginkgo biloba", "type": "Herb", "start": 0, "end": 13},
                    {"id": "T2", "text": "aspirin", "type": "Drug", "start": 66, "end": 73},
                    {"id": "T3", "text": "bleeding risk", "type": "Effect", "start": 86, "end": 99},
                ],
                "relations": [
                    {"type": "potentiates", "entity1_id": "T1", "entity2_id": "T2"}
                ],
            },
            {
                "text": "Garlic supplements significantly reduced saquinavir bioavailability "
                "by inducing CYP3A4 and P-glycoprotein.",
                "entities": [
                    {"id": "T1", "text": "Garlic", "type": "Herb", "start": 0, "end": 6},
                    {"id": "T2", "text": "saquinavir", "type": "Drug", "start": 41, "end": 51},
                    {"id": "T3", "text": "CYP3A4", "type": "Mechanism", "start": 83, "end": 89},
                ],
                "relations": [
                    {"type": "induces", "entity1_id": "T1", "entity2_id": "T3"},
                    {"type": "interacts_with", "entity1_id": "T1", "entity2_id": "T2"},
                ],
            },
            {
                "text": "Echinacea purpurea was shown to inhibit CYP1A2 while inducing "
                "CYP3A4 in human hepatocytes.",
                "entities": [
                    {"id": "T1", "text": "Echinacea purpurea", "type": "Herb", "start": 0, "end": 18},
                    {"id": "T2", "text": "CYP1A2", "type": "Mechanism", "start": 40, "end": 46},
                    {"id": "T3", "text": "CYP3A4", "type": "Mechanism", "start": 62, "end": 68},
                ],
                "relations": [
                    {"type": "inhibits", "entity1_id": "T1", "entity2_id": "T2"},
                    {"type": "induces", "entity1_id": "T1", "entity2_id": "T3"},
                ],
            },
        ]

        # Informal / consumer text examples
        informal_examples = [
            {
                "text": "I started taking ashwagandha supplements and my thyroid medication "
                "seemed less effective after that.",
                "entities": [
                    {"id": "T1", "text": "ashwagandha", "type": "Herb", "start": 17, "end": 28},
                    {"id": "T2", "text": "thyroid medication", "type": "Drug", "start": 48, "end": 66},
                ],
                "relations": [
                    {"type": "interacts_with", "entity1_id": "T1", "entity2_id": "T2"}
                ],
            },
            {
                "text": "My doctor warned me not to take ginkgo with my blood thinners "
                "because it could cause more bleeding.",
                "entities": [
                    {"id": "T1", "text": "ginkgo", "type": "Herb", "start": 31, "end": 37},
                    {"id": "T2", "text": "blood thinners", "type": "Drug", "start": 46, "end": 60},
                    {"id": "T3", "text": "bleeding", "type": "Effect", "start": 86, "end": 94},
                ],
                "relations": [
                    {"type": "interacts_with", "entity1_id": "T1", "entity2_id": "T2"}
                ],
            },
            {
                "text": "Been using turmeric capsules daily with metformin for diabetes. "
                "Blood sugar readings seem more stable now.",
                "entities": [
                    {"id": "T1", "text": "turmeric", "type": "Herb", "start": 11, "end": 19},
                    {"id": "T2", "text": "metformin", "type": "Drug", "start": 39, "end": 48},
                ],
                "relations": [
                    {"type": "potentiates", "entity1_id": "T1", "entity2_id": "T2"}
                ],
            },
        ]

        for i, ex in enumerate(formal_examples):
            self._sentences.append(
                AnnotatedSentence(
                    text=ex["text"],
                    source_corpus="ddi_synthetic",
                    doc_id=f"synth_formal_{i}",
                    sentence_id=str(i),
                    entities=ex["entities"],
                    relations=ex["relations"],
                    is_formal=True,
                )
            )

        for i, ex in enumerate(informal_examples):
            self._sentences.append(
                AnnotatedSentence(
                    text=ex["text"],
                    source_corpus="cadec_synthetic",
                    doc_id=f"synth_informal_{i}",
                    sentence_id=str(i),
                    entities=ex["entities"],
                    relations=ex["relations"],
                    is_formal=False,
                )
            )

        logger.info(
            f"Generated {len(formal_examples)} formal + "
            f"{len(informal_examples)} informal synthetic sentences"
        )

    def load_all(self) -> list[AnnotatedSentence]:
        """Load all sentences from all corpora."""
        self._load()
        return self._sentences

    def load_formal_text(self) -> pd.DataFrame:
        """Load only formal (PubMed-style) annotated text."""
        self._load()
        formal = [s for s in self._sentences if s.is_formal]
        return self._sentences_to_df(formal)

    def load_informal_text(self) -> pd.DataFrame:
        """Load only informal (consumer-generated) annotated text."""
        self._load()
        informal = [s for s in self._sentences if not s.is_formal]
        return self._sentences_to_df(informal)

    @staticmethod
    def _sentences_to_df(sentences: list[AnnotatedSentence]) -> pd.DataFrame:
        """Convert annotated sentences to a DataFrame."""
        return pd.DataFrame(
            [
                {
                    "text": s.text,
                    "source_corpus": s.source_corpus,
                    "doc_id": s.doc_id,
                    "sentence_id": s.sentence_id,
                    "entities": s.entities,
                    "relations": s.relations,
                    "is_formal": s.is_formal,
                }
                for s in sentences
            ]
        )
