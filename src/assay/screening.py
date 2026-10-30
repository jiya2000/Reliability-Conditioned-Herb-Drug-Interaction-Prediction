"""Evidence-linked herb–drug research screening, never a clinical classifier."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path

from src.assay.predict import predict_smiles


def load_catalog(root):
    path = Path(root) / 'data/demo/public_evidence.json'
    catalog = json.loads(path.read_text())
    if catalog.get('schema') != 'hdi.demonstration-evidence/v1':
        raise ValueError('Unsupported demonstration evidence')
    catalog['sha256'] = sha256(path.read_bytes()).hexdigest()
    return catalog


def summarize_evidence(studies, herb, drug):
    selected = [deepcopy(row) for row in studies if row['herb'] == herb and row['drug'] == drug]
    unique = {}
    for row in selected:
        key = (row['pmid'], row['herb'], row['drug'], row['endpoint'])
        if key in unique and unique[key] != row:
            raise ValueError('Conflicting extractions for the same publication and endpoint')
        unique[key] = row
    rows = list(unique.values())
    directions = {row['direction'] for row in rows}
    if not rows:
        finding = 'No human trial in this demonstration catalog'
    elif directions == {'decreased_exposure'}:
        finding = 'Human studies report decreased drug exposure'
    elif directions == {'no_change_reported'}:
        finding = 'No PK change reported for the studied preparation'
    else:
        finding = 'Different study findings; inspect preparation and context'
    return {'finding': finding, 'raw_records': len(selected), 'unique_publications': len({r['pmid'] for r in rows}),
            'duplicate_records_removed': len(selected) - len(rows), 'studies': rows,
            'independence': 'Unique publication IDs do not prove independent cohorts or consensus.',
            'review_status': 'Abstract-level source check; independent dual review pending'}


def screen_pair(catalog, bundle, herb, drug, context='published', exposure_uM=None):
    if herb not in catalog['herbs'] or drug not in catalog['drugs']:
        raise ValueError('Pair outside supported demonstration catalog')
    if context not in {'published', 'different_or_unknown'}:
        raise ValueError('Unknown context policy')
    if exposure_uM is not None and (not math.isfinite(exposure_uM) or exposure_uM < 0):
        raise ValueError('Exposure must be finite and nonnegative')
    herb_info, drug_info = catalog['herbs'][herb], catalog['drugs'][drug]
    evidence = summarize_evidence(catalog['studies'], herb, drug)
    estimates = predict_smiles(bundle, herb_info['smiles']) if herb_info['smiles'] and bundle else None
    relevant = []
    if estimates:
        for row in estimates['predictions']:
            if row['target'] not in drug_info['targets']:
                continue
            row = deepcopy(row)
            # This is a sensitivity ratio, not receptor occupancy, an M12
            # regulatory threshold, an AUC prediction or an interaction score.
            if exposure_uM is not None and herb_info['mechanism'] != 'induction':
                row['illustrative_exposure_over_ic50'] = exposure_uM / row['implied_ic50_uM']
                row['ratio_lower'] = exposure_uM / (10 ** (6 - row['lower_pic50']))
                row['ratio_upper'] = exposure_uM / (10 ** (6 - row['upper_pic50']))
            relevant.append(row)
    missing = ['Preparation-specific constituent exposure and composition',
               'Independent clinical reference outcomes and validation',
               'Victim-drug quantitative clearance/transport contributions']
    if herb_info['mechanism'] == 'induction':
        missing.append('Induction parameters: inhibition IC50 cannot establish induction')
    if not evidence['studies']:
        missing.append('Human evidence for this pair is absent from the small demonstration catalog')
    if context == 'different_or_unknown':
        missing.append('Requested formulation, population or dose differs from the cited studies')
    return {'schema': 'hdi.research-screen/v1', 'herb': herb, 'drug': drug,
            'clinical_prediction': 'ABSTAIN — clinical prediction not validated',
            'evidence': evidence, 'context': context,
            'constituent': herb_info['constituent'],
            'mechanism': herb_info['mechanism'], 'mechanism_note': herb_info['mechanism_note'],
            'target_pathways': drug_info['targets'], 'assay_estimates': relevant,
            'full_assay_profile': estimates, 'assumed_unbound_exposure_uM': exposure_uM,
            'exposure_is_measured': False, 'missing_requirements': missing,
            'catalog_sha256': catalog['sha256'],
            'sources': [herb_info['mapping_source'], herb_info['structure_source'], drug_info['source']],
            'meaning': 'Published findings and model assay estimates are displayed separately; neither establishes patient-specific safety.'}
