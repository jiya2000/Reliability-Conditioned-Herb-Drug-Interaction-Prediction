"""Normalize a captured ChEMBL page for review, never canonical herb features."""
import math


def review_rows(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get('activities'), list):
        raise ValueError('Expected ChEMBL activity response')
    rows, seen = [], set()
    for row in payload['activities']:
        identity = row.get('activity_id')
        if not isinstance(identity, int) or identity in seen:
            raise ValueError('Missing or duplicate activity_id')
        seen.add(identity)
        value = row.get('standard_value')
        unit = row.get('standard_units')
        reason = []
        numeric = None
        try:
            numeric = float(value)
            if isinstance(value, bool) or not math.isfinite(numeric) or numeric <= 0:
                raise ValueError()
        except (TypeError, ValueError):
            numeric = None
            reason.append('missing_or_invalid_value')
        if unit not in {'nM','uM'}:
            reason.append('unsupported_units')
        if row.get('standard_relation') != '=':
            reason.append('censored_or_unspecified_relation')
        if row.get('standard_type') not in {'IC50','Ki'}:
            reason.append('unsupported_measurement_type')
        if row.get('target_organism') != 'Homo sapiens':
            reason.append('not_human_target')
        if row.get('data_validity_comment'):
            reason.append('source_validity_comment_requires_review')
        if row.get('potential_duplicate'):
            reason.append('potential_duplicate')
        if not row.get('assay_chembl_id') or not row.get('document_chembl_id') or not row.get('target_chembl_id'):
            reason.append('incomplete_provenance')
        # Preserve censoring; only exact quantities receive an exact unit conversion.
        normalized = (numeric / 1000 if unit == 'nM' else numeric) if not reason else None
        rows.append(dict(activity_id=identity, molecule_chembl_id=row.get('molecule_chembl_id'),
            molecule_name=row.get('molecule_pref_name'), target_chembl_id=row.get('target_chembl_id'),
            target_name=row.get('target_pref_name'), organism=row.get('target_organism'),
            assay_chembl_id=row.get('assay_chembl_id'), document_chembl_id=row.get('document_chembl_id'),
            source_database_id=row.get('src_id'), standard_type=row.get('standard_type'),
            standard_relation=row.get('standard_relation'), standard_value=value, standard_units=unit,
            exact_value_uM=normalized, review_flags=';'.join(reason),
            status='pending_compound_assay_review', task_a_eligible=False,
            limitation='Compound assay only; no herb mapping, exposure interpretation or substrate fraction inferred.'))
    return rows
