"""Versioned assay endpoint, structure grouping and auditable role assignment."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.data.chembl_dataset import verify_dataset

PROTOCOL = {
    'version': 'assay-pic50/v1',
    'endpoint': 'pIC50 = 9 - log10(IC50 in nM); larger means greater assay potency',
    'features': 'RDKit Morgan radius 2, 2048 bits, chirality disabled',
    'roles': {'train': .60, 'validation': .15, 'calibration': .10, 'test': .15},
    'aggregation': 'median within document, then median across documents per structure-target',
    'dispersion': 'median absolute deviation of all retained log-potencies per structure-target',
    'weight': '1 / (1 + training-pair MAD); repeat consistency, not clinical reliability',
    'screen': 'exact human IC50 nM; A/B assays; inhibition wording; exclude named nonhuman species',
    'clinical_task_a_eligible': False,
    'clinical_task_b_eligible': False,
}
FOREIGN_SPECIES = re.compile(
    r'\b(rat|rats|mouse|mice|murine|rabbit|guinea.?pig|canine|dog|dogs|'
    r'monkey|monkeys|porcine|bovine|hamster|zebrafish)\b', re.I)


def canonical_structure(smiles):
    """Group salts, stereoisomers and equivalent SMILES conservatively.

    Largest carbon-containing fragment, canonical nonisomeric SMILES. No
    tautomer/charge normalization is claimed. Counterions are discarded.
    """
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    organic = [m for m in fragments if any(a.GetAtomicNum() == 6 for a in m.GetAtoms())]
    if not organic:
        return None
    parent = max(organic, key=lambda m: (m.GetNumHeavyAtoms(), Chem.MolToSmiles(m)))
    return Chem.MolToSmiles(parent, isomericSmiles=False)


def scaffold_key(smiles):
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    # Acyclic compounds all share one conservative scaffold group.
    return scaffold or '__ACYCLIC__'


def fingerprints(smiles):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    result = np.empty((len(smiles), 2048), dtype=np.uint8)
    for i, value in enumerate(smiles):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError('Invalid canonical structure')
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), result[i])
    return result


def load_records(directory):
    directory = Path(directory)
    manifest = verify_dataset(directory)
    frame = pd.read_csv(directory / 'assay_research_candidates.csv', keep_default_na=False)
    reasons = []
    structures = {s: canonical_structure(s) for s in sorted(set(frame.canonical_smiles))}
    frame['structure'] = frame.canonical_smiles.map(structures)
    for row in frame.itertuples():
        reason = []
        if (row.standard_type != 'IC50' or row.standard_relation != '=' or
                row.standard_units != 'nM' or row.organism != 'Homo sapiens'):
            reason.append('endpoint_mismatch')
        if row.assay_type not in {'A', 'B'}:
            reason.append('unsupported_assay_type')
        if not re.search(r'\binhibit', row.assay_description, re.I):
            reason.append('inhibition_not_explicit')
        if FOREIGN_SPECIES.search(row.assay_description):
            reason.append('nonhuman_description')
        if row.structure is None:
            reason.append('invalid_structure')
        try:
            value = float(row.standard_value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError()
            expected = 9 - np.log10(value)
            if abs(float(row.pchembl_value) - expected) > .06:
                reason.append('pchembl_inconsistent')
        except (TypeError, ValueError):
            reason.append('invalid_potency')
        if not row.document_chembl_id:
            reason.append('missing_document')
        reasons.append(';'.join(reason))
    frame['exclusion_reason'] = reasons
    excluded = frame[frame.exclusion_reason != ''].copy()
    retained = frame[frame.exclusion_reason == ''].copy()
    retained['pic50'] = 9 - np.log10(retained.standard_value.astype(float))
    retained['scaffold'] = retained.structure.map(
        {s: scaffold_key(s) for s in set(retained.structure)})
    return retained, excluded, manifest


def assign_roles(records, mode='scaffold', seed=20261030):
    """Assign groups before aggregation; source split purges cross-role structures.

    Test > calibration > validation > train priority is fixed without outcomes.
    A source split can lose many rows; the audit records all purged activity IDs.
    """
    columns = {'compound': 'structure', 'scaffold': 'scaffold', 'source': 'document_chembl_id'}
    if mode not in columns:
        raise ValueError('Unknown split mode')
    column = columns[mode]
    groups = np.array(sorted(records[column].unique()), dtype=object)
    if len(groups) < 20:
        raise ValueError('At least 20 independent split groups required')
    np.random.default_rng(seed).shuffle(groups)
    a, b, c = (int(len(groups) * x) for x in (.60, .75, .85))
    mapping = {g: role for role, values in zip(
        PROTOCOL['roles'], np.split(groups, [a, b, c])) for g in values}
    result = records.copy()
    result['role'] = result[column].map(mapping)
    purged = result.iloc[:0].copy()
    if mode == 'source':
        priority = {r: i for i, r in enumerate(PROTOCOL['roles'])}
        rank = result.role.map(priority)
        highest = rank.groupby(result.structure).transform('max')
        purged = result[rank != highest].copy()
        result = result[rank == highest].copy()
    audit = {'mode': mode, 'seed': seed, 'group_column': column,
             'purged_activity_ids': purged.activity_id.tolist(), 'overlaps': {}}
    roles = list(PROTOCOL['roles'])
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            pair = {}
            for name in ['structure', 'scaffold', 'document_chembl_id', 'assay_chembl_id']:
                overlap = set(result.loc[result.role == left, name]) & set(result.loc[result.role == right, name])
                pair[name] = len(overlap)
            if pair['structure'] or pair[column]:
                raise ValueError('Forbidden role overlap')
            audit['overlaps'][left + ':' + right] = pair
    return result, purged, audit


def aggregate(records):
    rows = []
    for (structure, target, role), group in records.groupby(['structure', 'target_symbol', 'role'], sort=True):
        document_values = group.groupby('document_chembl_id').pic50.median()
        values = group.pic50.to_numpy()
        median = float(document_values.median())
        mad = float(np.median(np.abs(values - np.median(values))))
        rows.append(dict(sample_id=hashlib.sha256((structure + '|' + target).encode()).hexdigest()[:24],
                         structure=structure, target=target, role=role, pic50=median,
                         mad=mad, spread=float(np.ptp(values)), weight=1 / (1 + mad),
                         measurements=len(group), documents=len(document_values),
                         activity_ids=json.dumps(sorted(group.activity_id.astype(int).tolist())),
                         document_ids=json.dumps(sorted(group.document_chembl_id.unique())),
                         molecule_ids=json.dumps(sorted(group.molecule_chembl_id.unique()))))
    return pd.DataFrame(rows)
