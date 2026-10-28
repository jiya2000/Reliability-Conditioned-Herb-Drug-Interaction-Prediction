"""Pure explicit partition contract; no splitting, fitting, calibration, or scoring.

Identifiers follow the consensus vocabulary without its normalization. Topology
uses Task B's undirected study/citation and shared-source semantics. Callers must
supply complete audited topology: an empty list explicitly means no studies;
missing/unrepresented relationships cannot be detected. Non-topology fields in
input records are ignored, including disputed labels, findings and raw text.

Protocols (each listed role requires its own nonempty partition):
- baseline: fit, assessment
- tuned: fit, tuning, assessment
- probability_calibrated: fit, tuning, probability_calibration, assessment
- conformal: fit, tuning, conformal_calibration, assessment
- full: fit, tuning, probability_calibration, conformal_calibration, assessment

Manifest hashes use SHA-256 over canonical JSON. topology_hash includes the pair
universe and directed citation records, even if a changed edge leaves connected
components unchanged. assignment_hash includes pair IDs and partition names;
roles, regime, task, protocol and optional seed are recorded separately. No clock,
filesystem access or environment-dependent state is included.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Literal, TypedDict

Regime = Literal['pair', 'herb', 'drug']
Role = Literal['fit', 'tuning', 'probability_calibration', 'conformal_calibration', 'assessment']
Protocol = Literal['baseline', 'tuned', 'probability_calibrated', 'conformal', 'full']


class Pair(TypedDict):
    herb_id: str
    drug_id: str


class Assignment(Pair):
    partition: str


class Topology(TypedDict):
    # Both pair IDs may be None for intermediate, unassigned provenance nodes.
    herb_id: str | None
    drug_id: str | None
    study_id: str
    source_id: str
    cited_study_ids: list[str]


_REQUIRED = {
    'baseline': {'fit', 'assessment'},
    'tuned': {'fit', 'tuning', 'assessment'},
    'probability_calibrated': {'fit', 'tuning', 'probability_calibration', 'assessment'},
    'conformal': {'fit', 'tuning', 'conformal_calibration', 'assessment'},
    'full': {'fit', 'tuning', 'probability_calibration', 'conformal_calibration', 'assessment'},
}


def canonical_manifest_json(manifest: Mapping) -> str:
    """Canonical UTF-8 JSON representation, also used for SHA-256 hashes."""
    return json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_manifest_json(value).encode('utf-8')).hexdigest()


def _identifier(value: object, field: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(c.isspace() for c in value)
            or value.lower() in {'unknown', 'unverified', 'n/a', 'na', 'tbd', 'todo'}):
        raise ValueError(f'Missing or noncanonical {field}: {value!r}; supply an audited identifier')
    return value


def _key(row: Mapping) -> tuple[str, str]:
    values = []
    for field, prefix in [('herb_id', 'HERB_'), ('drug_id', 'DRUG_')]:
        value = _identifier(row.get(field), field)
        if re.fullmatch(prefix + r'[A-Z0-9]+(?:_[A-Z0-9]+)*', value) is None:
            raise ValueError(f'Malformed {field}: {value!r}; use canonical {prefix} identifiers')
        values.append(value)
    return tuple(values)


class _Components:
    def __init__(self):
        self.parent = {}

    def root(self, node):
        self.parent.setdefault(node, node)
        root = node
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[node] != node:
            previous = self.parent[node]
            self.parent[node] = root
            node = previous
        return root

    def link(self, a, b):
        self.parent[self.root(a)] = self.root(b)

    def groups(self):
        groups = {}
        for node in self.parent:
            groups.setdefault(self.root(node), []).append(node)
        return sorted(sorted(nodes) for nodes in groups.values())


def validate_partitions(
    pairs: Iterable[Pair], assignments: Iterable[Assignment], *, regime: Regime,
    partition_roles: Mapping[str, Role], protocol: Protocol = 'full',
    task: Literal['A', 'B'] = 'A', topology: Iterable[Topology] | None = None,
    seed: int | None = None,
) -> dict:
    """Validate supplied assignments and return a deterministic v1 manifest.

    Each protocol requires exactly its documented roles (see ``_REQUIRED``), one
    nonempty partition per role. All roles are mutually separated under the
    selected regime; Task B additionally enforces evidence separation. ``conformal``
    omits probability calibration; ``full`` separates both calibration stages.
    ``seed`` records upstream assignment provenance only; no randomization occurs.
    Intermediate topology rows have both pair IDs explicitly None. All other
    topology pairs must belong to the pair universe. Duplicate publication rows
    for a pair/source fail, matching Task B's publication contract.

    Topology rows must be mappings; unsupported containers raise ValueError with
    guidance to supply mapping records. Missing optional non-topology metadata
    is allowed.

    Raises ValueError for invalid or infeasible contracts. Does not attest source
    authenticity, metadata completeness, class balance, sample adequacy, or any
    statistical independence/coverage guarantee. Pass DataFrames via
    ``frame.to_dict('records')``; inputs are never modified.
    """
    if regime not in {'pair', 'herb', 'drug'} or task not in {'A', 'B'}:
        raise ValueError('Require regime pair/herb/drug and task A/B')
    if protocol not in _REQUIRED:
        raise ValueError(f'Unknown protocol {protocol!r}')
    roles = dict(partition_roles)
    for partition in roles:
        _identifier(partition, 'partition')
    if (any(not isinstance(r, str) for r in roles.values())
            or set(roles.values()) != _REQUIRED[protocol]
            or len(roles) != len(_REQUIRED[protocol])):
        raise ValueError(f'Invalid partition roles or missing required partitions: {protocol} '
                         f'requires one distinct partition for each of {sorted(_REQUIRED[protocol])}')
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError('seed must be an integer or None')
    keys = [_key(row) for row in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate pair keys in pair universe')
    assigned = {}
    for row in assignments:
        key = _key(row)
        if key in assigned:
            raise ValueError(f'Duplicate pair assignment: {key}')
        partition = _identifier(row.get('partition'), 'partition')
        if partition not in roles:
            raise ValueError(f'Unknown assignment partition: {partition}')
        assigned[key] = partition
    if set(keys) != set(assigned):
        raise ValueError(f'Missing assignments: {sorted(set(keys) - set(assigned))}; '
                         f'extra assignments: {sorted(set(assigned) - set(keys))}')
    if set(assigned.values()) != set(roles):
        raise ValueError('Missing required nonempty partitions; supply more independent pairs')
    keys.sort()
    pair_ids = {key: 'pair:' + _hash(list(key)) for key in keys}
    evidence = _Components()
    records = []
    publications = set()
    study_pairs = {}
    if task == 'B' and topology is None:
        raise ValueError('Insufficient metadata: Task B requires explicit complete topology (including [] for zero studies)')
    for row in (() if topology is None else topology):
        if not isinstance(row, Mapping):
            raise ValueError('Topology rows must be mappings; supply dictionary records '
                             "(for DataFrames use frame.to_dict('records'))")
        if not {'herb_id', 'drug_id', 'study_id', 'source_id', 'cited_study_ids'} <= row.keys():
            raise ValueError('Insufficient topology metadata: require pair IDs, study_id, source_id, cited_study_ids')
        key = None if row['herb_id'] is None and row['drug_id'] is None else _key(row)
        if key is not None and key not in assigned:
            raise ValueError(f'Extra topology pair {key}; include it in the universe or explicitly use unassigned provenance IDs')
        study = _identifier(row['study_id'], 'study_id')
        source = _identifier(row['source_id'], 'source_id')
        cited = row['cited_study_ids']
        if not isinstance(cited, list):
            raise ValueError('cited_study_ids must be an explicit parsed list (empty means no represented citations)')
        cited = sorted(set(_identifier(c, 'cited_study_id') for c in cited))
        publication = (key, source)
        if publication in publications:
            raise ValueError(f'Duplicate topology publication for pair/source: {publication}')
        publications.add(publication)
        if key is not None:
            study_pairs.setdefault(study, set()).add(pair_ids[key])
        evidence.link(('study', study), ('source', source))
        for ancestor in cited:
            evidence.link(('study', study), ('study', ancestor))
        records.append(dict(pair_id=pair_ids[key] if key else None, study_id=study,
                            source_id=source, cited_study_ids=cited))
    records.sort(key=canonical_manifest_json)
    components = []
    membership = {pair_ids[k]: [] for k in keys}
    for nodes in evidence.groups():
        component_id = 'evidence:' + _hash(nodes)
        members = sorted({pair_id for kind, identifier in nodes if kind == 'study'
                          for pair_id in study_pairs.get(identifier, ())})
        components.append(dict(component_id=component_id, nodes=nodes, pair_ids=members))
        for member in members:
            membership[member].append(component_id)
    # Joining via pair nodes is essential: one pair can bridge multiple otherwise
    # disconnected citation components. Entity constraints join these further.
    constraints = _Components()
    for key in keys:
        node = ('pair', pair_ids[key])
        constraints.root(node)
        if regime != 'pair':
            constraints.link(node, (regime, key[0 if regime == 'herb' else 1]))
        if task == 'B':
            for component in membership[pair_ids[key]]:
                constraints.link(node, ('evidence', component))
    by_id = {pair_ids[k]: assigned[k] for k in keys}
    groups = constraints.groups()
    for nodes in groups:
        partitions = sorted({by_id[n[1]] for n in nodes if n[0] == 'pair'})
        if len(partitions) > 1:
            raise ValueError(f'Infeasible connected component: forbidden {regime}/evidence overlap '
                             f'across partitions {partitions}; keep component together or revise protocol/data. '
                             f'Members: {nodes}')
    assignment_rows = [dict(pair_id=pair_ids[k], herb_id=k[0], drug_id=k[1],
                            partition=assigned[k]) for k in keys]
    return dict(schema='hdi.partition-manifest/v1', task=task, regime=regime,
                protocol=protocol, partition_roles=dict(sorted(roles.items())), seed=seed,
                assignments=assignment_rows,
                entities={kind: sorted({k[i] for k in keys})
                          for i, kind in enumerate(['herb', 'drug'])},
                evidence_components=components, pair_evidence_membership=membership,
                constraint_components=groups,
                topology_hash=_hash(dict(pairs=[list(k) for k in keys], topology=records)),
                assignment_hash=_hash(assignment_rows))
