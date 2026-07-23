"""
DrugBank XML Loader

Parses DrugBank XML (full database download) to extract:
- Drug entities (name, DrugBank ID, type, description, SMILES, InChIKey)
- Drug-drug interactions (pairs with description)
- Drug targets (UniProt IDs, gene names, actions)
- Drug enzymes and transporters

DrugBank XML schema reference: https://go.drugbank.com/docs/drugbank-xml
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

# DrugBank XML namespace
NS = "{http://www.drugbank.ca}"


@dataclass
class DrugEntity:
    """Represents a single drug entry from DrugBank."""

    drugbank_id: str
    name: str
    drug_type: str  # "small molecule", "biotech", etc.
    description: str = ""
    cas_number: str = ""
    smiles: str = ""
    inchikey: str = ""
    molecular_formula: str = ""
    molecular_weight: float = 0.0
    categories: list[str] = field(default_factory=list)
    atc_codes: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)


@dataclass
class DrugInteraction:
    """Represents a drug-drug interaction pair."""

    drug1_id: str
    drug2_id: str
    drug1_name: str
    drug2_name: str
    description: str = ""


@dataclass
class DrugTarget:
    """Represents a drug-target association."""

    drug_id: str
    drug_name: str
    target_name: str
    uniprot_id: str = ""
    gene_name: str = ""
    actions: list[str] = field(default_factory=list)
    organism: str = "Humans"


class DrugBankLoader:
    """
    Loads and parses DrugBank XML database.

    Usage:
        loader = DrugBankLoader("data/raw/drugbank.xml")
        drugs_df = loader.load_drugs()
        interactions_df = loader.load_interactions()
        targets_df = loader.load_targets()
    """

    def __init__(self, xml_path: str | Path):
        self.xml_path = Path(xml_path)
        self._drugs: list[DrugEntity] = []
        self._interactions: list[DrugInteraction] = []
        self._targets: list[DrugTarget] = []
        self._parsed = False

    def _parse(self) -> None:
        """Parse the full DrugBank XML file."""
        if self._parsed:
            return

        if not self.xml_path.exists():
            logger.warning(
                f"DrugBank XML not found at {self.xml_path}. "
                "Download from https://go.drugbank.com/releases/latest"
            )
            self._generate_synthetic_data()
            return

        logger.info(f"Parsing DrugBank XML: {self.xml_path}")

        try:
            tree = ET.iterparse(str(self.xml_path), events=("end",))
            for event, elem in tqdm(tree, desc="Parsing DrugBank"):
                if elem.tag == f"{NS}drug":
                    self._parse_drug_element(elem)
                    elem.clear()  # Free memory for large XML
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}. Using synthetic data.")
            self._generate_synthetic_data()
            return

        self._parsed = True
        logger.info(
            f"Loaded {len(self._drugs)} drugs, "
            f"{len(self._interactions)} interactions, "
            f"{len(self._targets)} targets"
        )

    def _parse_drug_element(self, drug_elem: ET.Element) -> None:
        """Extract drug entity, interactions, and targets from a <drug> element."""
        drug_type = drug_elem.attrib.get("type", "unknown")

        # Primary DrugBank ID
        db_id_elem = drug_elem.find(f"{NS}drugbank-id[@primary='true']")
        if db_id_elem is None:
            db_id_elem = drug_elem.find(f"{NS}drugbank-id")
        if db_id_elem is None:
            return
        drugbank_id = db_id_elem.text or ""

        name = self._get_text(drug_elem, f"{NS}name")

        # Chemical properties
        smiles = ""
        inchikey = ""
        mol_formula = ""
        mol_weight = 0.0

        props = drug_elem.find(f"{NS}calculated-properties")
        if props is not None:
            for prop in props.findall(f"{NS}property"):
                kind = self._get_text(prop, f"{NS}kind")
                value = self._get_text(prop, f"{NS}value")
                if kind == "SMILES":
                    smiles = value
                elif kind == "InChIKey":
                    inchikey = value
                elif kind == "Molecular Formula":
                    mol_formula = value
                elif kind == "Molecular Weight":
                    try:
                        mol_weight = float(value)
                    except (ValueError, TypeError):
                        pass

        # Categories
        categories = []
        cats_elem = drug_elem.find(f"{NS}categories")
        if cats_elem is not None:
            for cat in cats_elem.findall(f"{NS}category"):
                cat_name = self._get_text(cat, f"{NS}category")
                if cat_name:
                    categories.append(cat_name)

        # ATC codes
        atc_codes = []
        atc_elem = drug_elem.find(f"{NS}atc-codes")
        if atc_elem is not None:
            for atc in atc_elem.findall(f"{NS}atc-code"):
                code = atc.attrib.get("code", "")
                if code:
                    atc_codes.append(code)

        drug = DrugEntity(
            drugbank_id=drugbank_id,
            name=name,
            drug_type=drug_type,
            description=self._get_text(drug_elem, f"{NS}description"),
            cas_number=self._get_text(drug_elem, f"{NS}cas-number"),
            smiles=smiles,
            inchikey=inchikey,
            molecular_formula=mol_formula,
            molecular_weight=mol_weight,
            categories=categories,
            atc_codes=atc_codes,
        )
        self._drugs.append(drug)

        # Drug-drug interactions
        interactions_elem = drug_elem.find(f"{NS}drug-interactions")
        if interactions_elem is not None:
            for interaction in interactions_elem.findall(
                f"{NS}drug-interaction"
            ):
                partner_id = self._get_text(interaction, f"{NS}drugbank-id")
                partner_name = self._get_text(interaction, f"{NS}name")
                desc = self._get_text(interaction, f"{NS}description")
                self._interactions.append(
                    DrugInteraction(
                        drug1_id=drugbank_id,
                        drug2_id=partner_id,
                        drug1_name=name,
                        drug2_name=partner_name,
                        description=desc,
                    )
                )

        # Drug targets
        targets_elem = drug_elem.find(f"{NS}targets")
        if targets_elem is not None:
            for target in targets_elem.findall(f"{NS}target"):
                target_name = self._get_text(target, f"{NS}name")
                polypeptide = target.find(f"{NS}polypeptide")
                uniprot_id = ""
                gene_name = ""
                if polypeptide is not None:
                    uniprot_id = polypeptide.attrib.get("id", "")
                    gene_name = self._get_text(
                        polypeptide, f"{NS}gene-name"
                    )

                actions = []
                actions_elem = target.find(f"{NS}actions")
                if actions_elem is not None:
                    for action in actions_elem.findall(f"{NS}action"):
                        if action.text:
                            actions.append(action.text)

                self._targets.append(
                    DrugTarget(
                        drug_id=drugbank_id,
                        drug_name=name,
                        target_name=target_name,
                        uniprot_id=uniprot_id,
                        gene_name=gene_name,
                        actions=actions,
                    )
                )

    @staticmethod
    def _get_text(elem: ET.Element, path: str) -> str:
        """Safely get text from an XML element."""
        child = elem.find(path)
        return child.text.strip() if child is not None and child.text else ""

    def _generate_synthetic_data(self) -> None:
        """Generate synthetic data for development/testing when DrugBank XML is unavailable."""
        logger.info("Generating synthetic DrugBank data for development...")

        # Synthetic drugs (common drugs used in herb-drug interaction studies)
        drug_data = [
            ("DB00001", "Warfarin", "small molecule", "CC(=O)CC1C2=CC=CC=C2OC1C1=CC=CC=C1O"),
            ("DB00002", "Metformin", "small molecule", "CN(C)C(=N)NC(=N)N"),
            ("DB00003", "Digoxin", "small molecule", ""),
            ("DB00004", "Cyclosporine", "small molecule", ""),
            ("DB00005", "Simvastatin", "small molecule", ""),
            ("DB00006", "Methotrexate", "small molecule", ""),
            ("DB00007", "Phenytoin", "small molecule", ""),
            ("DB00008", "Carbamazepine", "small molecule", ""),
            ("DB00009", "Lithium", "small molecule", ""),
            ("DB00010", "Ibuprofen", "small molecule", "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"),
            ("DB00011", "Aspirin", "small molecule", "CC(=O)OC1=CC=CC=C1C(=O)O"),
            ("DB00012", "Acetaminophen", "small molecule", "CC(=O)NC1=CC=C(O)C=C1"),
            ("DB00013", "Amoxicillin", "small molecule", ""),
            ("DB00014", "Omeprazole", "small molecule", ""),
            ("DB00015", "Atorvastatin", "small molecule", ""),
            ("DB00016", "Clopidogrel", "small molecule", ""),
            ("DB00017", "Fluoxetine", "small molecule", ""),
            ("DB00018", "Sertraline", "small molecule", ""),
            ("DB00019", "Lisinopril", "small molecule", ""),
            ("DB00020", "Amlodipine", "small molecule", ""),
        ]

        for db_id, name, dtype, smiles in drug_data:
            self._drugs.append(
                DrugEntity(
                    drugbank_id=db_id,
                    name=name,
                    drug_type=dtype,
                    description=f"Synthetic entry for {name}",
                    smiles=smiles,
                )
            )

        # Synthetic interactions
        interaction_pairs = [
            ("DB00001", "DB00002", "Warfarin", "Metformin", "May increase anticoagulant effect"),
            ("DB00001", "DB00005", "Warfarin", "Simvastatin", "Increased bleeding risk"),
            ("DB00003", "DB00010", "Digoxin", "Ibuprofen", "NSAIDs may increase digoxin levels"),
            ("DB00004", "DB00005", "Cyclosporine", "Simvastatin", "Increased risk of myopathy"),
            ("DB00007", "DB00008", "Phenytoin", "Carbamazepine", "Mutual enzyme induction"),
            ("DB00001", "DB00011", "Warfarin", "Aspirin", "Greatly increased bleeding risk"),
            ("DB00014", "DB00016", "Omeprazole", "Clopidogrel", "Reduced clopidogrel activation"),
            ("DB00017", "DB00018", "Fluoxetine", "Sertraline", "Serotonin syndrome risk"),
        ]

        for d1_id, d2_id, d1_name, d2_name, desc in interaction_pairs:
            self._interactions.append(
                DrugInteraction(
                    drug1_id=d1_id,
                    drug2_id=d2_id,
                    drug1_name=d1_name,
                    drug2_name=d2_name,
                    description=desc,
                )
            )

        # Synthetic targets
        target_data = [
            ("DB00001", "Warfarin", "Vitamin K epoxide reductase", "P23458", "VKORC1", ["inhibitor"]),
            ("DB00001", "Warfarin", "Cytochrome P450 2C9", "P11712", "CYP2C9", ["substrate"]),
            ("DB00003", "Digoxin", "Na+/K+ ATPase alpha-1", "P05023", "ATP1A1", ["inhibitor"]),
            ("DB00005", "Simvastatin", "HMG-CoA reductase", "P04035", "HMGCR", ["inhibitor"]),
            ("DB00010", "Ibuprofen", "Cyclooxygenase-2", "P35354", "PTGS2", ["inhibitor"]),
        ]

        for drug_id, drug_name, target_name, uniprot, gene, actions in target_data:
            self._targets.append(
                DrugTarget(
                    drug_id=drug_id,
                    drug_name=drug_name,
                    target_name=target_name,
                    uniprot_id=uniprot,
                    gene_name=gene,
                    actions=actions,
                )
            )

        self._parsed = True
        logger.info(
            f"Generated {len(self._drugs)} synthetic drugs, "
            f"{len(self._interactions)} interactions, "
            f"{len(self._targets)} targets"
        )

    def load_drugs(self) -> pd.DataFrame:
        """Load all drug entities as a DataFrame."""
        self._parse()
        records = []
        for d in self._drugs:
            records.append(
                {
                    "drugbank_id": d.drugbank_id,
                    "name": d.name,
                    "drug_type": d.drug_type,
                    "description": d.description,
                    "cas_number": d.cas_number,
                    "smiles": d.smiles,
                    "inchikey": d.inchikey,
                    "molecular_formula": d.molecular_formula,
                    "molecular_weight": d.molecular_weight,
                    "categories": d.categories,
                    "atc_codes": d.atc_codes,
                }
            )
        return pd.DataFrame(records)

    def load_interactions(self) -> pd.DataFrame:
        """Load all drug-drug interaction pairs as a DataFrame."""
        self._parse()
        records = [
            {
                "drug1_id": i.drug1_id,
                "drug2_id": i.drug2_id,
                "drug1_name": i.drug1_name,
                "drug2_name": i.drug2_name,
                "description": i.description,
            }
            for i in self._interactions
        ]
        return pd.DataFrame(records)

    def load_targets(self) -> pd.DataFrame:
        """Load all drug-target associations as a DataFrame."""
        self._parse()
        records = [
            {
                "drug_id": t.drug_id,
                "drug_name": t.drug_name,
                "target_name": t.target_name,
                "uniprot_id": t.uniprot_id,
                "gene_name": t.gene_name,
                "actions": t.actions,
                "organism": t.organism,
            }
            for t in self._targets
        ]
        return pd.DataFrame(records)

    def get_drug_by_id(self, drugbank_id: str) -> Optional[DrugEntity]:
        """Look up a single drug by its DrugBank ID."""
        self._parse()
        for d in self._drugs:
            if d.drugbank_id == drugbank_id:
                return d
        return None

    def get_interaction_partners(self, drugbank_id: str) -> list[str]:
        """Get all interaction partner IDs for a given drug."""
        self._parse()
        partners = []
        for i in self._interactions:
            if i.drug1_id == drugbank_id:
                partners.append(i.drug2_id)
            elif i.drug2_id == drugbank_id:
                partners.append(i.drug1_id)
        return partners
