"""
IMPPAT 2.0 Data Loader

Loads Indian Medicinal Plants, Phytochemistry And Therapeutics data.
IMPPAT provides herb-phytochemical-therapeutic use associations critical
for herb-drug interaction prediction.

Data source: https://cb.imsc.res.in/imppat/
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class HerbEntity:
    """Represents a medicinal plant/herb from IMPPAT."""

    plant_id: str
    plant_name: str
    botanical_name: str = ""
    family: str = ""
    traditional_uses: list[str] = field(default_factory=list)
    plant_parts_used: list[str] = field(default_factory=list)


@dataclass
class Phytochemical:
    """Represents a phytochemical compound from IMPPAT."""

    compound_id: str
    compound_name: str
    smiles: str = ""
    molecular_weight: float = 0.0
    pubchem_cid: str = ""
    source_plants: list[str] = field(default_factory=list)


@dataclass
class HerbTherapeuticUse:
    """A herb-therapeutic use association."""

    plant_id: str
    plant_name: str
    therapeutic_use: str
    evidence_type: str = ""  # Traditional, Experimental, Clinical


class IMPPATLoader:
    """
    Loads herb, phytochemical, and therapeutic data from IMPPAT 2.0.

    IMPPAT provides the critical herb-side data for building the
    heterogeneous knowledge graph. It links:
    - Herbs → Phytochemicals (contains relationship)
    - Herbs → Therapeutic uses
    - Phytochemicals → Molecular properties (SMILES, PubChem CIDs)

    Usage:
        loader = IMPPATLoader("data/raw/imppat/")
        herbs_df = loader.load_herbs()
        phytochemicals_df = loader.load_phytochemicals()
        herb_compound_edges = loader.load_herb_compound_edges()
    """

    def __init__(self, data_dir: Optional[str | Path] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self._herbs: list[HerbEntity] = []
        self._phytochemicals: list[Phytochemical] = []
        self._therapeutic_uses: list[HerbTherapeuticUse] = []
        self._herb_compound_edges: list[tuple[str, str]] = []
        self._parsed = False

    def _load(self) -> None:
        """Load data from IMPPAT files or generate synthetic data."""
        if self._parsed:
            return

        if self.data_dir and self.data_dir.exists():
            self._load_from_files()
        else:
            logger.warning(
                f"IMPPAT data not found at {self.data_dir}. "
                "Using synthetic data. Request access at: "
                "https://cb.imsc.res.in/imppat/"
            )
            self._generate_synthetic_data()

        self._parsed = True

    def _load_from_files(self) -> None:
        """Load from IMPPAT TSV/CSV files."""
        logger.info(f"Loading IMPPAT data from: {self.data_dir}")

        # IMPPAT typically provides data in TSV format
        plants_file = self.data_dir / "plants.tsv"
        compounds_file = self.data_dir / "phytochemicals.tsv"
        uses_file = self.data_dir / "therapeutic_uses.tsv"
        associations_file = self.data_dir / "plant_compound_associations.tsv"

        if plants_file.exists():
            df = pd.read_csv(plants_file, sep="\t")
            for _, row in df.iterrows():
                self._herbs.append(
                    HerbEntity(
                        plant_id=str(row.get("plant_id", "")),
                        plant_name=str(row.get("plant_name", "")),
                        botanical_name=str(row.get("botanical_name", "")),
                        family=str(row.get("family", "")),
                    )
                )

        if compounds_file.exists():
            df = pd.read_csv(compounds_file, sep="\t")
            for _, row in df.iterrows():
                self._phytochemicals.append(
                    Phytochemical(
                        compound_id=str(row.get("compound_id", "")),
                        compound_name=str(row.get("compound_name", "")),
                        smiles=str(row.get("smiles", "")),
                        pubchem_cid=str(row.get("pubchem_cid", "")),
                    )
                )

        if associations_file.exists():
            df = pd.read_csv(associations_file, sep="\t")
            for _, row in df.iterrows():
                self._herb_compound_edges.append(
                    (str(row["plant_id"]), str(row["compound_id"]))
                )

        logger.info(
            f"Loaded {len(self._herbs)} herbs, "
            f"{len(self._phytochemicals)} phytochemicals, "
            f"{len(self._herb_compound_edges)} herb-compound edges"
        )

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic IMPPAT-like data for development."""
        logger.info("Generating synthetic IMPPAT data...")

        # Herbs commonly involved in herb-drug interactions
        herbs = [
            ("HERB001", "Ashwagandha", "Withania somnifera", "Solanaceae",
             ["Adaptogen", "Anti-inflammatory", "Anxiolytic"]),
            ("HERB002", "Turmeric", "Curcuma longa", "Zingiberaceae",
             ["Anti-inflammatory", "Antioxidant", "Hepatoprotective"]),
            ("HERB003", "St. John's Wort", "Hypericum perforatum", "Hypericaceae",
             ["Antidepressant", "Anti-inflammatory", "Wound healing"]),
            ("HERB004", "Ginkgo", "Ginkgo biloba", "Ginkgoaceae",
             ["Cognitive enhancement", "Vasodilator", "Antioxidant"]),
            ("HERB005", "Garlic", "Allium sativum", "Amaryllidaceae",
             ["Antimicrobial", "Cardiovascular", "Antihypertensive"]),
            ("HERB006", "Ginger", "Zingiber officinale", "Zingiberaceae",
             ["Antiemetic", "Anti-inflammatory", "Digestive"]),
            ("HERB007", "Neem", "Azadirachta indica", "Meliaceae",
             ["Antimicrobial", "Antipyretic", "Anti-inflammatory"]),
            ("HERB008", "Tulsi", "Ocimum tenuiflorum", "Lamiaceae",
             ["Adaptogen", "Antimicrobial", "Immunomodulator"]),
            ("HERB009", "Brahmi", "Bacopa monnieri", "Plantaginaceae",
             ["Nootropic", "Anxiolytic", "Antioxidant"]),
            ("HERB010", "Shatavari", "Asparagus racemosus", "Asparagaceae",
             ["Galactagogue", "Adaptogen", "Immunomodulator"]),
            ("HERB011", "Guggul", "Commiphora wightii", "Burseraceae",
             ["Hypolipidemic", "Anti-inflammatory", "Thyroid stimulant"]),
            ("HERB012", "Arjuna", "Terminalia arjuna", "Combretaceae",
             ["Cardioprotective", "Antihypertensive", "Antioxidant"]),
            ("HERB013", "Amla", "Phyllanthus emblica", "Phyllanthaceae",
             ["Antioxidant", "Hepatoprotective", "Vitamin C source"]),
            ("HERB014", "Guduchi", "Tinospora cordifolia", "Menispermaceae",
             ["Immunomodulator", "Antipyretic", "Hepatoprotective"]),
            ("HERB015", "Echinacea", "Echinacea purpurea", "Asteraceae",
             ["Immunostimulant", "Anti-inflammatory", "Antimicrobial"]),
        ]

        for pid, name, botanical, family, uses in herbs:
            self._herbs.append(
                HerbEntity(
                    plant_id=pid,
                    plant_name=name,
                    botanical_name=botanical,
                    family=family,
                    traditional_uses=uses,
                )
            )
            for use in uses:
                self._therapeutic_uses.append(
                    HerbTherapeuticUse(
                        plant_id=pid,
                        plant_name=name,
                        therapeutic_use=use,
                        evidence_type="Traditional",
                    )
                )

        # Phytochemicals found in these herbs
        phytochemicals = [
            ("PHYTO001", "Withanolide A", "Ashwagandha", ""),
            ("PHYTO002", "Withaferin A", "Ashwagandha", ""),
            ("PHYTO003", "Curcumin", "Turmeric", ""),
            ("PHYTO004", "Demethoxycurcumin", "Turmeric", ""),
            ("PHYTO005", "Hypericin", "St. John's Wort", ""),
            ("PHYTO006", "Hyperforin", "St. John's Wort", ""),
            ("PHYTO007", "Ginkgolide A", "Ginkgo", ""),
            ("PHYTO008", "Bilobalide", "Ginkgo", ""),
            ("PHYTO009", "Allicin", "Garlic", ""),
            ("PHYTO010", "Ajoene", "Garlic", ""),
            ("PHYTO011", "Gingerol", "Ginger", ""),
            ("PHYTO012", "Shogaol", "Ginger", ""),
            ("PHYTO013", "Azadirachtin", "Neem", ""),
            ("PHYTO014", "Nimbolide", "Neem", ""),
            ("PHYTO015", "Eugenol", "Tulsi", ""),
            ("PHYTO016", "Ursolic acid", "Tulsi", ""),
            ("PHYTO017", "Bacoside A", "Brahmi", ""),
            ("PHYTO018", "Bacosine", "Brahmi", ""),
            ("PHYTO019", "Shatavarin", "Shatavari", ""),
            ("PHYTO020", "Guggulsterone", "Guggul", ""),
            ("PHYTO021", "Arjunolic acid", "Arjuna", ""),
            ("PHYTO022", "Gallic acid", "Amla", ""),
            ("PHYTO023", "Berberine", "Guduchi", ""),
            ("PHYTO024", "Echinacoside", "Echinacea", ""),
        ]

        # Map herb names to IDs for edges
        herb_name_to_id = {h.plant_name: h.plant_id for h in self._herbs}

        for cid, name, herb_name, smiles in phytochemicals:
            self._phytochemicals.append(
                Phytochemical(
                    compound_id=cid,
                    compound_name=name,
                    smiles=smiles,
                )
            )
            herb_id = herb_name_to_id.get(herb_name, "")
            if herb_id:
                self._herb_compound_edges.append((herb_id, cid))

        logger.info(
            f"Generated {len(self._herbs)} herbs, "
            f"{len(self._phytochemicals)} phytochemicals, "
            f"{len(self._herb_compound_edges)} herb-compound edges"
        )

    def load_herbs(self) -> pd.DataFrame:
        """Load all herbs as a DataFrame."""
        self._load()
        return pd.DataFrame(
            [
                {
                    "plant_id": h.plant_id,
                    "plant_name": h.plant_name,
                    "botanical_name": h.botanical_name,
                    "family": h.family,
                    "traditional_uses": h.traditional_uses,
                }
                for h in self._herbs
            ]
        )

    def load_phytochemicals(self) -> pd.DataFrame:
        """Load all phytochemicals as a DataFrame."""
        self._load()
        return pd.DataFrame(
            [
                {
                    "compound_id": p.compound_id,
                    "compound_name": p.compound_name,
                    "smiles": p.smiles,
                    "molecular_weight": p.molecular_weight,
                    "pubchem_cid": p.pubchem_cid,
                }
                for p in self._phytochemicals
            ]
        )

    def load_therapeutic_uses(self) -> pd.DataFrame:
        """Load herb-therapeutic use associations."""
        self._load()
        return pd.DataFrame(
            [
                {
                    "plant_id": t.plant_id,
                    "plant_name": t.plant_name,
                    "therapeutic_use": t.therapeutic_use,
                    "evidence_type": t.evidence_type,
                }
                for t in self._therapeutic_uses
            ]
        )

    def load_herb_compound_edges(self) -> list[tuple[str, str]]:
        """Load herb → compound edges for knowledge graph construction."""
        self._load()
        return self._herb_compound_edges
