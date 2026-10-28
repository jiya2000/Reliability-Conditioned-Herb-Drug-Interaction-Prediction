"""Synthetic partition contracts only; no scientific validation or performance."""
import copy
import itertools

import pytest

from src.training.evaluation_protocol import validate_partitions, canonical_manifest_json

ROLES = {r: r for r in ['fit', 'tuning', 'probability_calibration',
                         'conformal_calibration', 'assessment']}


def inputs():
    pairs = [dict(herb_id=f'HERB_{i}', drug_id=f'DRUG_{i}') for i in range(5)]
    assignments = [dict(p, partition=r) for p, r in zip(pairs, ROLES)]
    return pairs, assignments


def validate(pairs=None, assignments=None, **kwargs):
    p, a = inputs()
    return validate_partitions(p if pairs is None else pairs,
                               a if assignments is None else assignments,
                               **({'regime': 'pair', 'partition_roles': ROLES} | kwargs))


def study(i, name=None, source=None, cited=None):
    return dict(herb_id=f'HERB_{i}', drug_id=f'DRUG_{i}',
                study_id=name or f'S{i}', source_id=source or f'PUB{i}',
                cited_study_ids=[] if cited is None else cited)


@pytest.mark.parametrize('regime', ['pair', 'herb', 'drug'])
def test_valid_regimes_zero_study_and_disputed_pairs(regime):
    p, a = inputs()
    p[0]['consensus_label'] = 'disputed'
    m = validate(p, a, regime=regime, task='B', topology=[])
    assert len(m['assignments']) == 5
    assert all(not x for x in m['pair_evidence_membership'].values())


@pytest.mark.parametrize('regime,field', [('herb', 'herb_id'), ('drug', 'drug_id')])
def test_entity_overlap_only_for_applicable_regime(regime, field):
    p, a = inputs()
    p[1][field] = p[0][field]
    a[1][field] = p[0][field]
    validate(p, a)  # Distinct pairs may share an entity in pair-disjoint mode.
    with pytest.raises(ValueError, match='Infeasible connected component'):
        validate(p, a, regime=regime)


def test_pair_overlap_duplicate_keys_and_incomplete_assignments():
    p, a = inputs()
    with pytest.raises(ValueError, match='Duplicate pair keys'):
        validate(p + [p[0]], a)
    with pytest.raises(ValueError, match='Duplicate pair assignment'):
        validate(p, a + [dict(a[0], partition='assessment')])
    with pytest.raises(ValueError, match='Missing assignments'):
        validate(p, a[:-1])
    with pytest.raises(ValueError, match='extra assignments'):
        validate(p[:-1], a)


@pytest.mark.parametrize('value', [' HERB_0', 'HERB_', 'herb_0', 'HERB_BAD ID', None, 1])
def test_malformed_identifiers_not_repaired(value):
    p, a = inputs()
    p[0]['herb_id'] = value
    with pytest.raises(ValueError, match='herb_id'):
        validate(p, a)


@pytest.mark.parametrize('topology', [
    [study(0, name='SHARED'), study(1, name='SHARED')],
    [study(0, source='SHARED'), study(1, source='SHARED')],
    [study(0, cited=['S1']), study(1)],
    [study(0, cited=['ABSENT']), study(1, cited=['ABSENT'])],
    [study(0, cited=['MID']), study(1, cited=['ANCESTOR']),
     dict(herb_id=None, drug_id=None, study_id='MID', source_id='MIDPUB',
          cited_study_ids=['ANCESTOR'])],
])
def test_shared_study_source_transitive_and_absent_ancestor_overlap(topology):
    with pytest.raises(ValueError, match='Infeasible connected component.*evidence overlap'):
        validate(task='B', topology=topology)


def test_pair_bridges_multiple_components_and_stays_together():
    p, a = inputs()
    p.append(dict(herb_id='HERB_5', drug_id='DRUG_5'))
    a.append(dict(p[-1], partition='fit'))
    topology = [study(0, name='LEFT', source='LP'),
                study(0, name='RIGHT', source='RP'),
                study(5, cited=['LEFT'])]
    m = validate(p, a, task='B', topology=topology)
    assert len(m['evidence_components']) == 2
    bridge = m['assignments'][0]['pair_id']
    assert len(m['pair_evidence_membership'][bridge]) == 2
    # A second pair attached to either component must share the bridge's partition.
    topology.append(study(1, cited=['RIGHT']))
    with pytest.raises(ValueError, match='Infeasible'):
        validate(p, a, task='B', topology=topology)


@pytest.mark.parametrize('left,right', list(itertools.combinations(ROLES, 2)))
def test_all_role_pairs_require_evidence_separation(left, right):
    indexes = list(ROLES)
    topology = [study(indexes.index(left), source='SHARED'),
                study(indexes.index(right), source='SHARED')]
    with pytest.raises(ValueError, match='Infeasible'):
        validate(task='B', topology=topology)


@pytest.mark.parametrize('protocol,roles', [
    ('baseline', ['fit', 'assessment']),
    ('tuned', ['fit', 'tuning', 'assessment']),
    ('probability_calibrated', ['fit', 'tuning', 'probability_calibration', 'assessment']),
    ('conformal', ['fit', 'tuning', 'conformal_calibration', 'assessment']),
    ('full', list(ROLES)),
])
def test_protocol_required_roles(protocol, roles):
    p, _ = inputs()
    p = p[:len(roles)]
    a = [dict(row, partition=f'p{i}') for i, row in enumerate(p)]
    mapping = {f'p{i}': r for i, r in enumerate(roles)}
    validate_partitions(p, a, regime='pair', partition_roles=mapping, protocol=protocol)
    with pytest.raises(ValueError, match='Invalid partition roles'):
        validate_partitions(p, a, regime='pair', partition_roles=dict(mapping, bad='fit'),
                            protocol=protocol)


def test_invalid_roles_empty_partitions_and_options():
    for roles in [dict(ROLES, fit='assessment'), dict(ROLES, fit='train'),
                  {k: v for k, v in ROLES.items() if k != 'tuning'}]:
        with pytest.raises(ValueError, match='Invalid partition roles'):
            validate(partition_roles=roles)
    p, a = inputs()
    a[0]['partition'] = 'assessment'
    with pytest.raises(ValueError, match='nonempty partitions'):
        validate(p, a)
    a[0]['partition'] = 'unlisted'
    with pytest.raises(ValueError, match='Unknown assignment partition'):
        validate(p, a)
    for kwargs in [dict(regime='random'), dict(task='C'), dict(protocol='unknown'), dict(seed=True)]:
        with pytest.raises(ValueError):
            validate(**kwargs)


def test_canonical_manifest_hashes_and_perturbation():
    p, a = inputs()
    topology = [study(0, cited=['Z', 'ABSENT']), study(1)]
    m = validate(p, a, task='B', topology=topology, seed=42)
    reordered = copy.deepcopy(topology[::-1])
    reordered[1]['cited_study_ids'].reverse()
    assert canonical_manifest_json(m) == canonical_manifest_json(
        validate(p[::-1], a[::-1], task='B', topology=reordered, seed=42))
    changed = copy.deepcopy(topology)
    changed[0]['cited_study_ids'].append('NEW')
    n = validate(p, a, task='B', topology=changed, seed=42)
    assert n['topology_hash'] != m['topology_hash']
    assert n['assignment_hash'] == m['assignment_hash']
    changed_assignments = copy.deepcopy(a)
    changed_assignments[0]['partition'], changed_assignments[1]['partition'] = (
        changed_assignments[1]['partition'], changed_assignments[0]['partition'])
    n = validate(p, changed_assignments, task='B', topology=topology, seed=42)
    assert n['assignment_hash'] != m['assignment_hash']
    assert n['topology_hash'] == m['topology_hash']
    for row in p + a + topology:
        row.update(outcome='positive', original_text='secret text', study_finding='negative',
                   prediction=0.99, consensus_label='disputed')
    assert validate(p, a, task='B', topology=topology, seed=42) == m


@pytest.mark.parametrize('topology', [None, [{}], [dict(study(0), study_id=' ')],
    [dict(study(0), source_id=None)], [dict(study(0), cited_study_ids='S1')],
    [dict(study(0), cited_study_ids=[None])], [study(8)], [study(0), study(0)]])
def test_insufficient_or_ambiguous_topology_fails(topology):
    with pytest.raises(ValueError):
        validate(task='B', topology=topology)


def test_infeasible_giant_component_fails_without_fabricating_assignments():
    p, a = inputs()
    original = copy.deepcopy(a)
    with pytest.raises(ValueError, match='Infeasible connected component'):
        validate(p, a, task='B', topology=[study(i, cited=['GLOBAL']) for i in range(5)])
    assert a == original


def test_no_fit_or_calibration_sentinel(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from src.models.conformal_abstention import ConformalTriStateEngine
    def forbidden(*args, **kwargs):
        pytest.fail('Partition validation attempted fitting/calibration')
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden)
    monkeypatch.setattr(ConformalTriStateEngine, 'calibrate', forbidden)
    validate(task='B', topology=[study(0)])


@pytest.mark.parametrize('field,value', [('study_id', 'NEWSTUDY'), ('source_id', 'NEWSOURCE'),
                                         ('cited_study_ids', ['NEWANCESTOR']),
                                         ('herb_id', 'HERB_1')])
def test_each_topology_field_affects_hash(field, value):
    topology = [study(0)]
    before = validate(task='B', topology=topology)
    changed = copy.deepcopy(topology)
    changed[0][field] = value
    if field == 'herb_id':
        changed[0]['drug_id'] = 'DRUG_1'
    after = validate(task='B', topology=changed)
    assert after['topology_hash'] != before['topology_hash']
    assert after['assignment_hash'] == before['assignment_hash']


class CountingStudy(str):
    """Count equality work, including linear searches through component nodes."""
    comparisons = 0
    __hash__ = str.__hash__

    def __eq__(self, other):
        type(self).comparisons += 1
        return super().__eq__(other)


@pytest.mark.parametrize('length', [600, 1200])
def test_fully_assigned_cyclic_chain_has_linear_membership_work(length):
    topology = [study(0, name=CountingStudy(f'CHAIN_{i}'), source=f'PUBCHAIN_{i}',
                      cited=[CountingStudy(f'CHAIN_{(i + 1) % length}')])
                for i in range(length)]
    CountingStudy.comparisons = 0
    m = validate(task='B', topology=topology)
    # A structural work bound, independent of machine speed. The previous
    # record-by-component list searches perform quadratic identifier comparisons.
    assert CountingStudy.comparisons < 30 * length
    assert len(m['evidence_components']) == 1
    component = m['evidence_components'][0]
    assert len(component['nodes']) == 2 * length
    pair_id = m['assignments'][0]['pair_id']
    assert component['pair_ids'] == [pair_id]
    assert m['pair_evidence_membership'][pair_id] == [component['component_id']]
    assert m == validate(task='B', topology=topology[::-1])
    assert m['assignments'] == validate()['assignments']


@pytest.mark.parametrize('row', [None, 1, [], ['study_id'], 'row', 1.5, True, set()])
def test_unsupported_topology_row_containers(row):
    with pytest.raises(ValueError, match='Topology rows must be mappings; supply dictionary records'):
        validate(task='B', topology=[row])


def test_mapping_topology_without_optional_metadata():
    from types import MappingProxyType
    assert validate(task='B', topology=[MappingProxyType(study(0))]) == validate(
        task='B', topology=[study(0)])


@pytest.mark.parametrize('shared_field', ['herb_id', 'drug_id'])
@pytest.mark.parametrize('regime', ['pair', 'herb', 'drug'])
def test_shared_entities_enforce_exactly_selected_constraint(shared_field, regime):
    p, a = inputs()
    p.append(dict(p[0], **{shared_field: p[1][shared_field]}))
    a.append(dict(p[-1], partition='fit'))
    # The new pair shares one entity within fit, the other across fit/tuning.
    constrained_field = {'herb': 'herb_id', 'drug': 'drug_id'}.get(regime)
    if constrained_field == shared_field:
        with pytest.raises(ValueError, match='Infeasible connected component'):
            validate(p, a, regime=regime)
    else:
        m = validate(p, a, regime=regime)
        assert len(m['assignments']) == len(p)


def test_redundant_citation_edge_changes_hash_without_changing_components():
    topology = [study(0, name='ROOT', cited=['MID']),
                dict(herb_id=None, drug_id=None, study_id='MID', source_id='MIDPUB',
                     cited_study_ids=['END'])]
    before = validate(task='B', topology=topology)
    topology[0]['cited_study_ids'].append('END')
    after = validate(task='B', topology=topology)
    assert after['topology_hash'] != before['topology_hash']
    for field in ['evidence_components', 'pair_evidence_membership',
                  'constraint_components', 'assignments', 'assignment_hash']:
        assert after[field] == before[field]
