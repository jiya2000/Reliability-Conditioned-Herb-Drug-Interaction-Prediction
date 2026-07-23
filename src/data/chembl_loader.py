"""
ChEMBL Data Loader

Loads bioactivity data and molecular structures from ChEMBL.
Supports both SQLite local database and REST API access.

Data source: https://www.ebi.ac.uk/chembl/
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class ChEMBLCompound:
    """A chemical compound from ChEMBL."""

    chembl_id: str
    pref_name: str = ""
    molecule_type: str = ""
    max_phase: int = 0  # 0-4 (4 = approved drug)
    smiles: str = ""
    inchi_key: str = ""
    molecular_weight: float = 0.0
    alogp: float = 0.0


@dataclass
class ChEMBLActivity:
    """A bioactivity measurement from ChEMBL."""

    activity_id: str
    chembl_id: str  # Compound
    target_chembl_id: str  # Target
    assay_chembl_id: str
    standard_type: str = ""  # IC50, Ki, EC50, etc.
    standard_value: Optional[float] = None
    standard_units: str = ""
    pchembl_value: Optional[float] = None  # -log10(activity)
    activity_comment: str = ""


@dataclass
class ChEMBLTarget:
    """A biological target from ChEMBL."""

    target_chembl_id: str
    pref_name: str = ""
    target_type: str = ""  # SINGLE PROTEIN, PROTEIN COMPLEX, etc.
    organism: str = ""
    uniprot_ids: list[str] = field(default_factory=list)


class ChEMBLLoader:
    """
    Loads compound, target, and bioactivity data from ChEMBL.

    Supports:
    - Local SQLite database (preferred for bulk access)
    - REST API fallback for individual queries
    - Synthetic data for development without ChEMBL access

    Usage:
        loader = ChEMBLLoader("data/raw/chembl.sqlite")
        compounds_df = loader.load_compounds()
        activities_df = loader.load_activities()
        targets_df = loader.load_targets()
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = Path(db_path) if db_path else None
        self._compounds: list[ChEMBLCompound] = []
        self._activities: list[ChEMBLActivity] = []
        self._targets: list[ChEMBLTarget] = []
        self._parsed = False

    def _load(self) -> None:
        """Load data from SQLite or generate synthetic data."""
        if self._parsed:
            return

        if self.db_path and self.db_path.exists():
            self._load_from_sqlite()
        else:
            logger.warning(
                f"ChEMBL database not found at {self.db_path}. "
                "Using synthetic data. Download from: "
                "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/"
            )
            self._generate_synthetic_data()

        self._parsed = True

    def _load_from_sqlite(self) -> None:
        """Load data from ChEMBL SQLite database."""
        import sqlite3

        logger.info(f"Loading ChEMBL from SQLite: {self.db_path}")
        conn = sqlite3.connect(str(self.db_path))

        try:
            # Load compounds
            compound_query = """
                SELECT
                    md.chembl_id,
                    md.pref_name,
                    md.molecule_type,
                    md.max_phase,
                    cs.canonical_smiles,
                    cs.standard_inchi_key,
                    cp.mw_freebase,
                    cp.alogp
                FROM molecule_dictionary md
                LEFT JOIN compound_structures cs ON md.molregno = cs.molregno
                LEFT JOIN compound_properties cp ON md.molregno = cp.molregno
                WHERE md.max_phase >= 1
                LIMIT 10000
            """
            df = pd.read_sql_query(compound_query, conn)
            for _, row in df.iterrows():
                self._compounds.append(
                    ChEMBLCompound(
                        chembl_id=row["chembl_id"],
                        pref_name=row.get("pref_name", "") or "",
                        molecule_type=row.get("molecule_type", "") or "",
                        max_phase=int(row.get("max_phase", 0) or 0),
                        smiles=row.get("canonical_smiles", "") or "",
                        inchi_key=row.get("standard_inchi_key", "") or "",
                        molecular_weight=float(
                            row.get("mw_freebase", 0) or 0
                        ),
                        alogp=float(row.get("alogp", 0) or 0),
                    )
                )

            # Load activities (drug-target interactions with quantitative data)
            activity_query = """
                SELECT
                    a.activity_id,
                    md.chembl_id AS compound_chembl_id,
                    td.chembl_id AS target_chembl_id,
                    ad.chembl_id AS assay_chembl_id,
                    a.standard_type,
                    a.standard_value,
                    a.standard_units,
                    a.pchembl_value,
                    a.activity_comment
                FROM activities a
                JOIN molecule_dictionary md ON a.molregno = md.molregno
                JOIN assays ad ON a.assay_id = ad.assay_id
                JOIN target_dictionary td ON ad.tid = td.tid
                WHERE a.standard_type IN ('IC50', 'Ki', 'EC50', 'Kd')
                AND a.standard_value IS NOT NULL
                AND a.pchembl_value >= 5
                LIMIT 50000
            """
            df = pd.read_sql_query(activity_query, conn)
            for _, row in df.iterrows():
                self._activities.append(
                    ChEMBLActivity(
                        activity_id=str(row["activity_id"]),
                        chembl_id=row["compound_chembl_id"],
                        target_chembl_id=row["target_chembl_id"],
                        assay_chembl_id=row["assay_chembl_id"],
                        standard_type=row.get("standard_type", "") or "",
                        standard_value=row.get("standard_value"),
                        standard_units=row.get("standard_units", "") or "",
                        pchembl_value=row.get("pchembl_value"),
                        activity_comment=row.get("activity_comment", "") or "",
                    )
                )

            # Load targets
            target_query = """
                SELECT
                    td.chembl_id,
                    td.pref_name,
                    td.target_type,
                    td.organism
                FROM target_dictionary td
                WHERE td.target_type = 'SINGLE PROTEIN'
                AND td.organism = 'Homo sapiens'
                LIMIT 5000
            """
            df = pd.read_sql_query(target_query, conn)
            for _, row in df.iterrows():
                self._targets.append(
                    ChEMBLTarget(
                        target_chembl_id=row["chembl_id"],
                        pref_name=row.get("pref_name", "") or "",
                        target_type=row.get("target_type", "") or "",
                        organism=row.get("organism", "") or "",
                    )
                )

        finally:
            conn.close()

        logger.info(
            f"Loaded {len(self._compounds)} compounds, "
            f"{len(self._activities)} activities, "
            f"{len(self._targets)} targets from ChEMBL"
        )

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic ChEMBL-like data for development."""
        logger.info("Generating synthetic ChEMBL data...")

        compounds = [
            ("CHEMBL25", "Aspirin", "Small molecule", 4, "CC(=O)OC1=CC=CC=C1C(=O)O", 180.16),
            ("CHEMBL112", "Ibuprofen", "Small molecule", 4, "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 206.28),
            ("CHEMBL661", "Warfarin", "Small molecule", 4, "", 308.33),
            ("CHEMBL1200689", "Metformin", "Small molecule", 4, "CN(C)C(=N)NC(=N)N", 129.16),
            ("CHEMBL1751", "Digoxin", "Small molecule", 4, "", 780.94),
            ("CHEMBL1200983", "Simvastatin", "Small molecule", 4, "", 418.57),
            ("CHEMBL44", "Phenytoin", "Small molecule", 4, "", 252.27),
            ("CHEMBL108", "Carbamazepine", "Small molecule", 4, "", 236.27),
            ("CHEMBL1200657", "Lithium carbonate", "Small molecule", 4, "", 73.89),
            ("CHEMBL1642", "Curcumin", "Small molecule", 0, "", 368.38),
            ("CHEMBL159", "Quercetin", "Small molecule", 0, "", 302.24),
            ("CHEMBL50", "Resveratrol", "Small molecule", 0, "", 228.24),
            ("CHEMBL1236120", "Berberine", "Small molecule", 0, "", 336.36),
            ("CHEMBL1279", "Piperine", "Small molecule", 0, "", 285.34),
            ("CHEMBL88", "Caffeine", "Small molecule", 0, "", 194.19),
        ]

        for cid, name, mtype, phase, smiles, mw in compounds:
            self._compounds.append(
                ChEMBLCompound(
                    chembl_id=cid,
                    pref_name=name,
                    molecule_type=mtype,
                    max_phase=phase,
                    smiles=smiles,
                    molecular_weight=mw,
                )
            )

        targets = [
            ("CHEMBL220", "Acetylcholinesterase", "SINGLE PROTEIN", "Homo sapiens"),
            ("CHEMBL217", "Cyclooxygenase-2", "SINGLE PROTEIN", "Homo sapiens"),
            ("CHEMBL3594", "CYP3A4", "SINGLE PROTEIN", "Homo sapiens"),
            ("CHEMBL3397", "CYP2D6", "SINGLE PROTEIN", "Homo sapiens"),
            ("CHEMBL340", "CYP2C9", "SINGLE PROTEIN", "Homo sapiens"),
            ("CHEMBL2111389", "P-glycoprotein", "SINGLE PROTEIN", "Homo sapiens"),
            ("CHEMBL2093869", "HMG-CoA reductase", "SINGLE PROTEIN", "Homo sapiens"),
        ]

        for tid, name, ttype, org in targets:
            self._targets.append(
                ChEMBLTarget(
                    target_chembl_id=tid,
                    pref_name=name,
                    target_type=ttype,
                    organism=org,
                )
            )

        activities = [
            ("1", "CHEMBL25", "CHEMBL217", "ASSAY1", "IC50", 1500.0, "nM", 5.82),
            ("2", "CHEMBL112", "CHEMBL217", "ASSAY2", "IC50", 2100.0, "nM", 5.68),
            ("3", "CHEMBL661", "CHEMBL340", "ASSAY3", "Ki", 320.0, "nM", 6.49),
            ("4", "CHEMBL1642", "CHEMBL3594", "ASSAY4", "IC50", 40000.0, "nM", 4.40),
            ("5", "CHEMBL159", "CHEMBL3594", "ASSAY5", "IC50", 10000.0, "nM", 5.00),
            ("6", "CHEMBL50", "CHEMBL3594", "ASSAY6", "IC50", 25000.0, "nM", 4.60),
            ("7", "CHEMBL1236120", "CHEMBL3594", "ASSAY7", "IC50", 8000.0, "nM", 5.10),
            ("8", "CHEMBL1236120", "CHEMBL2111389", "ASSAY8", "IC50", 15000.0, "nM", 4.82),
        ]

        for aid, cid, tid, assay, stype, sval, sunits, pval in activities:
            self._activities.append(
                ChEMBLActivity(
                    activity_id=aid,
                    chembl_id=cid,
                    target_chembl_id=tid,
                    assay_chembl_id=assay,
                    standard_type=stype,
                    standard_value=sval,
                    standard_units=sunits,
                    pchembl_value=pval,
                )
            )

        logger.info(
            f"Generated {len(self._compounds)} synthetic compounds, "
            f"{len(self._activities)} activities, "
            f"{len(self._targets)} targets"
        )

    def load_compounds(self) -> pd.DataFrame:
        """Load all compounds as a DataFrame."""
        self._load()
        return pd.DataFrame(
            [
                {
                    "chembl_id": c.chembl_id,
                    "pref_name": c.pref_name,
                    "molecule_type": c.molecule_type,
                    "max_phase": c.max_phase,
                    "smiles": c.smiles,
                    "inchi_key": c.inchi_key,
                    "molecular_weight": c.molecular_weight,
                    "alogp": c.alogp,
                }
                for c in self._compounds
            ]
        )

    def load_activities(self) -> pd.DataFrame:
        """Load all bioactivity data as a DataFrame."""
        self._load()
        return pd.DataFrame(
            [
                {
                    "activity_id": a.activity_id,
                    "chembl_id": a.chembl_id,
                    "target_chembl_id": a.target_chembl_id,
                    "assay_chembl_id": a.assay_chembl_id,
                    "standard_type": a.standard_type,
                    "standard_value": a.standard_value,
                    "standard_units": a.standard_units,
                    "pchembl_value": a.pchembl_value,
                }
                for a in self._activities
            ]
        )

    def load_targets(self) -> pd.DataFrame:
        """Load all targets as a DataFrame."""
        self._load()
        return pd.DataFrame(
            [
                {
                    "target_chembl_id": t.target_chembl_id,
                    "pref_name": t.pref_name,
                    "target_type": t.target_type,
                    "organism": t.organism,
                }
                for t in self._targets
            ]
        )
