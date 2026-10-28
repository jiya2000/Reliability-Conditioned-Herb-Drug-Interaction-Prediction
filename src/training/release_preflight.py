"""Read-only dataset release eligibility contract. See docs/dataset_release.md."""
from __future__ import annotations

import csv
from collections.abc import Mapping
import hashlib
import io
import json
import math
from datetime import date
import re
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.evidence_schema import validate_consensus_labels_table
from src.data.mechanistic_curation import (
    validate_curation_dataframe, is_curation_verified, extract_curated_feature_matrix,
)
from src.training.evaluation_protocol import validate_partitions, canonical_manifest_json

VERSIONS = {'release_manifest': '2', 'preflight': '2', 'target_policy': '1', 'consensus': '1',
            'mechanistic': '1', 'evidence': '1', 'evaluation_protocol': '1'}
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_JSON_DEPTH = 64
PRIVATE_DIRECTORY = 'annotation_batch_24_seed20261029'
LIMITATION = ('Eligibility checks recorded attestations and structural prerequisites only; '
              'software cannot authenticate sources, reviewer independence, or scientific validity. '
              'Class support is a minimum contract, not statistical sample adequacy.')


def _path(path):
    path = Path(path)
    if PRIVATE_DIRECTORY in path.parts or PRIVATE_DIRECTORY in path.resolve().parts:
        raise ValueError('Private annotation directory is excluded from preflight access')
    return path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def _snapshot(path):
    with _path(path).open('rb') as handle:
        snapshot = handle.read(MAX_INPUT_BYTES + 1)
    if len(snapshot) > MAX_INPUT_BYTES:
        raise ValueError('Input exceeds 32 MiB limit')
    return snapshot


def _csv_snapshot(snapshot):
    text = snapshot.decode('utf-8-sig')
    try:
        header = next(csv.reader(io.StringIO(text), strict=True), [])
        if len(header) != len(set(header)):
            raise ValueError('Duplicate CSV headers')
        return pd.read_csv(io.StringIO(text))
    except csv.Error:
        raise ValueError('CSV parsing failed: malformed header or field exceeds parser limit') from None
    except pd.errors.ParserError:
        raise ValueError('CSV parsing failed: malformed table') from None


def _reject_constant(value):
    raise ValueError('Nonstandard JSON constant is forbidden: ' + value)


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Nonfinite JSON number is forbidden')
    return number


def _manifest_snapshot(snapshot):
    text = snapshot.decode('utf-8-sig')
    # Bound nesting before the decoder recurses; ignore delimiters in strings.
    depth, quoted, escaped = 0, False, False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '[{':
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError('JSON nesting exceeds 64 levels')
        elif char in ']}':
            depth -= 1
    try:
        return json.loads(text, object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant, parse_float=_finite_float)
    except RecursionError:
        raise ValueError('JSON nesting exceeds decoder limit') from None


def _validate_source_registry(sources):
    if not isinstance(sources, dict):
        raise ValueError('Source registry must be a mapping')
    ids, citations = set(), set()
    for ref, source in sorted(sources.items()):
        _identifier(ref)
        if not isinstance(source, dict):
            raise ValueError('Every source registry entry must be an object')
        required = ('source_id', 'source_name', 'source_url_or_citation',
                    'retrieval_date', 'verification_status', 'approval_ref')
        if any(k not in source for k in required):
            raise ValueError('Source registry entry is missing required fields')
        _identifier(source['source_id'])
        _identifier(source['approval_ref'])
        if not all(_present(source[k]) for k in ('source_name', 'source_url_or_citation', 'retrieval_date')):
            raise ValueError('Source registry entry requires source metadata')
        try:
            if date.fromisoformat(source['retrieval_date']).isoformat() != source['retrieval_date']:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError('Source registry retrieval_date must be YYYY-MM-DD') from None
        if source['verification_status'] != 'approved':
            raise ValueError('Every source registry entry requires recorded verification approval')
        source_id = source['source_id']
        citation = source['source_url_or_citation']
        if source_id in ids or citation in citations:
            raise ValueError('Ambiguous source registry alias: reuse one canonical registry entry per source ID/citation')
        ids.add(source_id)
        citations.add(citation)


def _identifier(value):
    if (not _present(value) or value != value.strip() or
            any(c.isspace() for c in value)):
        raise ValueError('Require canonical nonempty identifier without whitespace')
    return value


def _records(value, kind):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ValueError(kind + ' must be a list/tuple of mapping records')
    result = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError(kind + ' rows must be mappings')
        row = dict(row)
        if not {'herb_id', 'drug_id'} <= row.keys():
            raise ValueError(kind + ' requires explicit herb_id and drug_id fields')
        for field, prefix in [('herb_id', 'HERB_'), ('drug_id', 'DRUG_')]:
            if kind == 'topology' and row.get('herb_id') is None and row.get('drug_id') is None:
                continue
            v = _identifier(row.get(field))
            if re.fullmatch(prefix + r'[A-Z0-9]+(?:_[A-Z0-9]+)*', v) is None:
                raise ValueError('Malformed ' + field)
        if kind == 'assignments':
            _identifier(row.get('partition'))
        else:
            for field in ('study_id', 'source_id'):
                _identifier(row.get(field))
            cited = row.get('cited_study_ids')
            if not isinstance(cited, list):
                raise ValueError('cited_study_ids must be a list')
            row['cited_study_ids'] = [_identifier(v) for v in cited]
        result.append(row)
    return result


def _present(value):
    return (isinstance(value, str) and bool(value.strip()) and
            value.strip().lower() not in {'unknown', 'unverified', 'tbd', 'todo', 'n/a', 'nan', 'none'})


def preflight_release(manifest_path=None, **kwargs):
    """Read path inputs once, then run the Phase 2 contract."""
    return _preflight_release(manifest_path, **kwargs)


def preflight_release_snapshots(snapshots, **kwargs):
    """Run the same contract over immutable bytes; never access input paths.

    Keys are consensus, mechanistic and manifest. None denotes missing input.
    Hashing and parsing both consume these exact bytes.
    """
    captured = dict(snapshots)
    def load(name):
        value = captured.get(name)
        if value is None:
            raise FileNotFoundError(name)
        if not isinstance(value, bytes):
            raise ValueError('Snapshot must be immutable bytes')
        if len(value) > MAX_INPUT_BYTES:
            raise ValueError('Input exceeds 32 MiB limit')
        return value
    return _preflight_release('manifest', consensus_path='consensus',
                              mechanistic_path='mechanistic', _load=load, **kwargs)


def _preflight_release(manifest_path=None, *, consensus_path='data/benchmark/consensus_labels.csv',
                      mechanistic_path='data/curation/mechanistic_features.csv', task,
                      protocol, regime, assignments=None, partition_roles=None, topology=None,
                      target_policy=None, metric_policy=None, _load=None):
    """Return JSON-safe deterministic checks, without writes, fitting or scoring.

    Missing prerequisites block; malformed/contradictory supplied records fail.
    Any blocked prerequisite makes the overall status blocked (failures remain
    visible). Contract success never grants evaluation eligibility: statistical
    adequacy and the actual Task B evidence gate remain outside this API. Inputs are explicit:
    no directory discovery, registry fetching, or annotation workbook access.
    """
    checks = []
    check_indexes = {}
    def add(name, status, reason):
        result = dict(prerequisite=name, status=status, reason=reason)
        if name in check_indexes:
            # A later diagnostic can strengthen, never hide, an earlier failure.
            index = check_indexes[name]
            rank = {'passed': 0, 'blocked': 1, 'failed': 2}
            if rank[status] >= rank[checks[index]['status']]:
                checks[index] = result
        else:
            check_indexes[name] = len(checks)
            checks.append(result)

    from src.training.evaluation_protocol import _REQUIRED
    valid_request = (isinstance(task, str) and task in {'A', 'B'} and
                     isinstance(protocol, str) and protocol in _REQUIRED and
                     isinstance(regime, str) and regime in {'pair', 'herb', 'drug'})
    add('request', 'passed' if valid_request else 'failed',
        'Require task A/B, documented protocol and pair/herb/drug regime')
    # Invalid objects must never reach comparisons, sets or JSON serialization.
    task = task if isinstance(task, str) else None
    protocol = protocol if isinstance(protocol, str) else None
    regime = regime if isinstance(regime, str) else None
    target_policy = target_policy if isinstance(target_policy, str) else None
    metric_policy = metric_policy if isinstance(metric_policy, str) else None
    policy_valid = (valid_request and target_policy == 'binary_consensus_v1' and
                    metric_policy == 'binary_discrimination_v1')
    add('target_metric_policy', 'passed' if policy_valid else 'failed',
        'Task A/B support explicit binary_consensus_v1 + binary_discrimination_v1 only; '
        'disputed decisions remain separate, never negative; calibration/coverage adequacy is not certified')
    records_valid = True
    for name, value in [('assignments', assignments), ('topology', topology)]:
        try:
            parsed = _records(value, name)
            if name == 'assignments':
                assignments = parsed
            else:
                topology = parsed
            if name == 'assignments' or task == 'B' or value is not None:
                add(name + '_arguments', 'passed' if value is not None else 'blocked',
                    'Canonical records checked' if value is not None else 'No records supplied')
        except (ValueError, TypeError) as exc:
            records_valid = False
            add(name + '_arguments', 'failed', str(exc))
            if name == 'assignments':
                assignments = None
            else:
                topology = None
    try:
        if partition_roles is not None:
            if not isinstance(partition_roles, Mapping):
                raise ValueError('partition_roles must be a mapping')
            partition_roles = dict(partition_roles)
            for k, v in partition_roles.items():
                _identifier(k)
                _identifier(v)
    except (ValueError, TypeError) as exc:
        records_valid = False
        partition_roles = None
        add('role_arguments', 'failed', str(exc))
    hashes = {}
    snapshots = {}
    for name, path in [('consensus', consensus_path), ('mechanistic', mechanistic_path),
                       ('manifest', manifest_path)]:
        if path is None:
            hashes[name] = None
            continue
        try:
            snapshots[name] = (_snapshot if _load is None else _load)(path)
            hashes[name] = hashlib.sha256(snapshots[name]).hexdigest()
        except FileNotFoundError:
            hashes[name] = None
            add(name + '_input', 'blocked', 'Supply the missing input file')
        except (OSError, ValueError, TypeError) as exc:
            hashes[name] = None
            add(name + '_input', 'failed', str(exc))

    manifest = None
    if hashes['manifest'] is None:
        add('release_approval', 'blocked', 'Supply a versioned, explicitly approved release manifest')
    else:
        try:
            manifest = _manifest_snapshot(snapshots['manifest'])
            if not isinstance(manifest, dict):
                raise ValueError('Manifest must be an object')
            if manifest.get('schema_versions') != VERSIONS:
                raise ValueError('Use the documented v2 schema_versions mapping')
            if manifest.get('release_version') is not None and not isinstance(manifest['release_version'], str):
                raise ValueError('release_version must be a canonical string')
            if not _present(manifest.get('release_version')):
                add('release_version', 'blocked', 'Supply a release version')
            else:
                _identifier(manifest['release_version'])
                add('release_version', 'passed', manifest['release_version'])
            status = manifest.get('approval_status')
            add('release_approval', 'passed' if status == 'approved' else
                'blocked' if status in (None, 'pending', 'unapproved') else 'failed',
                'Explicit release approval recorded' if status == 'approved' else
                'Obtain release approval; CSV validity is not approval')
            for name in ('consensus', 'mechanistic'):
                expected = manifest.get('input_sha256', {}).get(name)
                add(name + '_hash', 'blocked' if not expected or hashes[name] is None else
                    'passed' if expected == hashes[name] else 'failed',
                    'Exact bytes must match the approved input_sha256; reapprove changed inputs')
            if manifest.get('target_policy') != target_policy or manifest.get('metric_policy') != metric_policy:
                add('release_policy', 'failed', 'Release must explicitly bind requested target/metric policies')
            else:
                add('release_policy', 'passed', 'Requested policies match release approval')
            refs = manifest.get('approval_refs')
            if isinstance(refs, list):
                for ref in refs:
                    _identifier(ref)
            add('approval_refs', 'blocked' if refs is None or refs == [] else
                'passed' if isinstance(refs, list) and all(_present(r) for r in refs) else 'failed',
                'Supply recorded release approval references; do not invent approval')
            add('manifest_schema', 'passed', 'Supported manifest structure and versions')
        except (ValueError, TypeError, AttributeError) as exc:
            add('manifest_schema', 'failed', str(exc))
            manifest = None

    consensus = None
    if hashes['consensus'] is not None:
        try:
            consensus = _csv_snapshot(snapshots['consensus'])
            valid, errors = validate_consensus_labels_table(consensus)
            if not valid:
                raise ValueError('; '.join(errors))
            for r in consensus.to_dict('records'):
                _identifier(r['herb_id'])
                _identifier(r['drug_id'])
                if r['consensus_label'] not in {'interaction', 'no_interaction', 'disputed'}:
                    raise ValueError('Noncanonical consensus state; unresolved/inconclusive remain staging/evidence only')
            empty = consensus.empty
            add('consensus_schema', 'passed', 'Distinct consensus states retained without binary coercion')
            add('consensus_labels', 'blocked' if empty else 'passed',
                'Curate source-backed independent decisions' if empty else 'Consensus decisions available')
        except (ValueError, OSError) as exc:
            consensus = None
            add('consensus_schema', 'failed', str(exc))

    sources_valid = False
    if manifest is None:
        add('source_registry', 'blocked', 'Need a parsed manifest source registry')
    else:
        try:
            _validate_source_registry(manifest['source_registry'])
            sources_valid = True
            add('source_registry', 'passed', 'Every source entry validated; ambiguous IDs/citations rejected')
        except KeyError:
            add('source_registry', 'blocked', 'Supply source_registry')
        except (ValueError, TypeError, AttributeError) as exc:
            add('source_registry', 'failed', str(exc))
    if not sources_valid or manifest is None or consensus is None or consensus.empty:
        add('decision_provenance', 'blocked', 'Need manifest and populated valid consensus decisions')
    else:
        try:
            sources = manifest['source_registry']
            provenance = manifest['provenance_registry']
            if not isinstance(sources, dict) or not isinstance(provenance, dict):
                raise ValueError('Registries must be reference-to-record mappings')
            for registry in (sources, provenance):
                for ref in registry:
                    _identifier(ref)
            decisions = manifest['decisions']
            indexed = {}
            for decision in decisions:
                key = (_identifier(decision['herb_id']), _identifier(decision['drug_id']))
                if key in indexed:
                    raise ValueError('Duplicate approval decision')
                indexed[key] = decision
            if set(indexed) != set(zip(consensus.herb_id, consensus.drug_id)):
                raise ValueError('Approval decisions must match the exact consensus pair universe')
            for row in consensus.to_dict('records'):
                d = indexed[(row['herb_id'], row['drug_id'])]
                if d['consensus_label'] != row['consensus_label'] or d['approval_status'] != 'approved':
                    raise ValueError('Decision approval/label contradicts consensus')
                reviewers = d['reviewer_approval_refs']
                adjudicator = d['adjudicator_approval_ref']
                if not isinstance(reviewers, list):
                    raise ValueError('reviewer_approval_refs must be a list')
                for ref in reviewers + [adjudicator, d['source_ref'], d['provenance_ref'],
                                         d['independence_approval_ref']]:
                    _identifier(ref)
                if (not isinstance(reviewers, list) or len(reviewers) != 2 or
                        not all(_present(r) for r in reviewers + [adjudicator]) or
                        len(set(reviewers + [adjudicator])) != 3):
                    raise ValueError('Require two distinct reviewer approvals and separate adjudicator approval')
                att = d['attestations']
                for name in ('independent_reviewers', 'blinded_to_predictions',
                             'external_consensus_only', 'independent_of_evaluation_evidence'):
                    if att[name] is not True:
                        raise ValueError('Independence attestation must be true: ' + name)
                source = sources[d['source_ref']]
                if any(source[name] != row[name] for name in
                       ('source_name', 'source_url_or_citation', 'retrieval_date')):
                    raise ValueError('Source registry contradicts consensus provenance')
                _identifier(source['source_id'])
                _identifier(source['approval_ref'])
                if task == 'B' and topology is not None and any(
                        isinstance(t, Mapping) and t.get('source_id') == source['source_id']
                        for t in topology):
                    raise ValueError('Consensus source overlaps evaluation topology; independence contradicts records')
                if source['verification_status'] != 'approved' or not _present(source['approval_ref']):
                    raise ValueError('Source registry requires recorded verification approval')
                if not _present(provenance[d['provenance_ref']]['reference']):
                    raise ValueError('Supply provenance registry reference')
                if not _present(d['independence_approval_ref']):
                    raise ValueError('Supply independence approval reference')
            add('decision_provenance', 'passed', 'Recorded approvals and independence attestations consistent; not authenticated')
        except KeyError as exc:
            add('decision_provenance', 'blocked', 'Supply missing attestation/registry field: ' + str(exc))
        except (TypeError, ValueError, AttributeError) as exc:
            add('decision_provenance', 'failed', str(exc))

    mechanistic = None
    verified = False
    if hashes['mechanistic'] is not None:
        try:
            mechanistic = _csv_snapshot(snapshots['mechanistic'])
            validate_curation_dataframe(mechanistic)
            for entity in mechanistic['entity']:
                _identifier(entity)
            for target in mechanistic['target']:
                _identifier(target)
            verified, n_verified, n_unverified = is_curation_verified(mechanistic)
            add('mechanistic_verification', 'passed' if verified else 'blocked',
                f'{n_verified} verified, {n_unverified} unverified rows; obtain external curation approval')
        except (ValueError, OSError) as exc:
            add('mechanistic_verification', 'failed', str(exc))

    if mechanistic is not None and manifest is not None:
        try:
            registry = manifest['mechanistic_provenance']
            if set(registry) != set(mechanistic['source'].dropna().astype(str)):
                raise ValueError('Mechanistic provenance registry must cover exact source set')
            for ref, entry in registry.items():
                _identifier(ref)
                _identifier(entry['approval_ref'])
                if not all(_present(entry[k]) for k in ('reference', 'approval_ref')):
                    raise ValueError('Mechanistic provenance requires reference and approval_ref')
            add('mechanistic_provenance', 'passed', 'Source approval references recorded; not authenticated')
        except KeyError as exc:
            add('mechanistic_provenance', 'blocked', 'Supply mechanistic provenance field: ' + str(exc))
        except (TypeError, ValueError, AttributeError) as exc:
            add('mechanistic_provenance', 'failed', str(exc))
    else:
        add('mechanistic_provenance', 'blocked', 'Need release manifest and mechanistic table')

    counts = {}
    split = None
    if consensus is None or consensus.empty:
        add('partition_structure', 'blocked', 'Need populated valid consensus pair universe')
        add('class_support', 'blocked', 'Need populated valid consensus decisions')
    else:
        counts = {'pairs': len(consensus), 'herbs': int(consensus.herb_id.nunique()),
                  'drugs': int(consensus.drug_id.nunique()),
                  'labels': {k: int(v) for k, v in sorted(consensus.consensus_label.value_counts().items())}}
        if not valid_request or not records_valid or assignments is None or partition_roles is None or (task == 'B' and topology is None):
            add('partition_structure', 'blocked', 'Supply explicit assignments/roles and Task B audited topology')
        else:
            try:
                split = validate_partitions(consensus.to_dict('records'), assignments,
                    regime=regime, partition_roles=partition_roles, protocol=protocol,
                    task=task, topology=topology)
                add('partition_structure', 'passed', 'Requested partition structure feasible')
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                add('partition_structure', 'failed', str(exc))
        if split is None:
            add('class_support', 'blocked', 'Validate assignments before checking per-partition support')
        else:
            labels = {(r.herb_id, r.drug_id): r.consensus_label for r in consensus.itertuples()}
            required = {'interaction', 'no_interaction'}
            states = required | {'disputed'}
            support = {a['partition']: dict.fromkeys(sorted(states), 0)
                       for a in split['assignments']}
            for a in assignments:
                support[a['partition']][labels[(a['herb_id'], a['drug_id'])]] += 1
            counts['partition_class_support'] = dict(sorted(support.items()))
            missing = {p: sorted(k for k, n in v.items() if k in required and n == 0)
                       for p, v in sorted(support.items()) if any(v[k] == 0 for k in required)}
            add('class_support', 'failed' if missing or not policy_valid else 'passed',
                'Unsupported target/metric policy; class support cannot be certified' if not policy_valid else
                'Collect additional independent decisions for missing states: ' + str(missing)
                if missing else 'Binary class support checked; disputed retained outside binary targets')
            counts['per_role'] = _support_by_role(split, labels)
    add('statistical_support', 'blocked',
        'No repository contract establishes minimum independent sample sizes; '
        'counts and disjointness do not establish sufficient evaluation support')
    if task == 'B':
        add('task_b_evidence_release', 'blocked',
            'Topology-only preflight: actual study release and Task B validate_studies/validate_study_sources '
            'are not checked; full Task B eligibility is not established')
    if consensus is None or consensus.empty or not verified:
        add('feature_completeness', 'blocked', 'Need valid pairs and verified mechanistic evidence')
    else:
        features = extract_curated_feature_matrix(consensus, mechanistic)
        missing = int((~np.isfinite(features)).sum())
        counts['missing_feature_cells'] = missing
        counts['complete_feature_pairs'] = int(np.isfinite(features).all(axis=1).sum())
        add('feature_completeness', 'blocked' if missing else 'passed',
            f'{missing} missing feature cells across all nine targets plus summary; curate missing evidence, never impute no interaction')
    if task == 'B':
        covered = set() if topology is None else {
            (r.get('herb_id'), r.get('drug_id')) for r in topology if isinstance(r, Mapping)}
        pairs = set() if consensus is None else set(zip(consensus.herb_id, consensus.drug_id))
        add('evidence_coverage', 'passed' if pairs and pairs <= covered else 'blocked',
            'Task B requires declared study/source/citation topology for every consensus pair; authenticity not checked')
    else:
        add('evidence_coverage', 'passed' if verified else 'blocked',
            'Task A requires verified mechanistic evidence; completeness checked separately')
    contract_checks = [c for c in checks if c['prerequisite'] not in
                       {'statistical_support', 'task_b_evidence_release'}]
    contract_status = ('failed' if any(c['status'] == 'failed' for c in contract_checks) else
                       'blocked' if any(c['status'] == 'blocked' for c in contract_checks) else 'passed')
    status = ('blocked' if any(c['status'] == 'blocked' for c in checks) else
              'failed' if any(c['status'] == 'failed' for c in checks) else 'prerequisites_passed')
    return dict(status=status, eligible=False, contract_status=contract_status,
                scope='dataset_and_partition_prerequisites_only',
                target_policy=target_policy if isinstance(target_policy, str) else None,
                metric_policy=metric_policy if isinstance(metric_policy, str) else None,
                checks=checks, counts=counts,
                input_sha256=hashes, release_version=manifest.get('release_version') if manifest else None,
                task=task if isinstance(task, str) else None,
                protocol=protocol if isinstance(protocol, str) else None,
                regime=regime if isinstance(regime, str) else None, schema_versions=dict(VERSIONS),
                software_versions={'package': 'hdi-prediction 0.1.0', 'python': platform.python_version(),
                                   'numpy': np.__version__, 'pandas': pd.__version__},
                partition_manifest=split, limitation=LIMITATION)


def _support_by_role(split, labels):
    assignments = split['assignments']
    group_by_pair = {}
    for i, group in enumerate(split['constraint_components']):
        for kind, identifier in group:
            if kind == 'pair':
                group_by_pair[identifier] = i
    result = {}
    for partition, role in sorted(split['partition_roles'].items()):
        rows = [a for a in assignments if a['partition'] == partition]
        states = ('interaction', 'no_interaction', 'disputed')
        memberships = split['pair_evidence_membership']
        result[role] = dict(partition=partition, pairs=len(rows),
            herbs=len({a['herb_id'] for a in rows}), drugs=len({a['drug_id'] for a in rows}),
            constraint_groups=len({group_by_pair[a['pair_id']] for a in rows}),
            evidence_components=len({c for a in rows for c in memberships[a['pair_id']]}),
            class_pairs={s: sum(labels[(a['herb_id'], a['drug_id'])] == s for a in rows) for s in states},
            class_constraint_groups={s: len({group_by_pair[a['pair_id']] for a in rows
                if labels[(a['herb_id'], a['drug_id'])] == s}) for s in states},
            limitation='Groups encode declared regime/topology constraints, not proven statistical independence')
    return result


def report_json(report):
    return canonical_manifest_json(report)
