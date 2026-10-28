"""Read-only Streamlit integration with the redesigned data/feature contracts.

No training, artifact deserialization, inference, or synthetic fallback. Publication
findings describe evidence; independent consensus is displayed separately.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.data.evidence_schema import (
    load_canonical_table, validate_dataframe, load_consensus_labels_table,
    extract_study_finding, StudyFinding,
)
from src.data.mechanistic_curation import CURATION_COLUMNS, load_mechanistic_curation, is_curation_verified
from src.features.task_a_contract import build_task_a_features, validate_task_a_features

KEYS = ['herb_id', 'drug_id']


@dataclass
class ResearchState:
    consensus: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=KEYS + ['consensus_label']))
    studies: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=KEYS + ['outcome']))
    blockers: list[str] = field(default_factory=list)
    evidence_available: bool = False
    mechanisms: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=CURATION_COLUMNS))
    verified_mechanistic_rows: int | None = None
    unverified_mechanistic_rows: int | None = None


def load_research_state(root: Path) -> ResearchState:
    """Reuse existing validators, failing closed without caching stale readiness."""
    root = Path(root)
    state = ResearchState()
    try:
        state.consensus, empty = load_consensus_labels_table(root / 'data/benchmark/consensus_labels.csv')
        if empty:
            state.blockers.append('Independent consensus labels are empty; publication findings cannot replace them.')
    except (OSError, ValueError, RuntimeError) as exc:
        state.blockers.append(f'Consensus table unavailable or invalid: {exc}')
    try:
        raw = load_canonical_table(str(root / 'data/benchmark/verified_botanical_hdi.csv'))
        clean, errors = validate_dataframe(raw, strict_traceability=True, require_registered=True)
        if errors:
            raise ValueError(f'{len(errors)} evidence validation errors; first: {errors[0]}')
        state.studies = clean
        state.evidence_available = True
    except (OSError, ValueError, RuntimeError) as exc:
        state.blockers.append(f'Study evidence unavailable or invalid: {exc}')
    try:
        curation = load_mechanistic_curation(root / 'data/curation/mechanistic_features.csv')
        state.mechanisms = curation
        verified, n_verified, n_unverified = is_curation_verified(curation)
        state.verified_mechanistic_rows = n_verified
        state.unverified_mechanistic_rows = n_unverified
        if not verified:
            state.blockers.append('Mechanistic records remain unverified; Task A features and Task B priors are blocked.')
        else:
            pairs = pd.concat([state.consensus[KEYS], state.studies[KEYS]]).drop_duplicates().reset_index(drop=True)
            if pairs.empty:
                state.blockers.append('No candidate pairs are available to check mechanistic feature completeness.')
            else:
                validate_task_a_features(build_task_a_features(pairs, curation))
    except (OSError, ValueError, RuntimeError) as exc:
        state.blockers.append(f'Mechanistic inputs unavailable, invalid or incomplete: {exc}')
    if state.evidence_available and not {'study_finding', 'is_primary'} <= set(state.studies):
        state.blockers.append('Task B needs audited study_finding and primary/echo annotations; the study table is not inference-ready.')
    # Task A currently emits CV JSON; Task B returns an in-memory fitted object.
    # Neither provides a deployable inference bundle/loader. Do not fit in the UI.
    state.blockers.append('No persisted, provenance-checked Task A/Task B inference bundle and app loader are available.')
    return state


def describe_pair(state: ResearchState, herb_id: str, drug_id: str) -> dict:
    rows = state.studies.loc[state.studies.herb_id.eq(herb_id) & state.studies.drug_id.eq(drug_id)].copy()
    labels = state.consensus.loc[state.consensus.herb_id.eq(herb_id) & state.consensus.drug_id.eq(drug_id)]
    findings = rows.outcome.map(extract_study_finding)
    positive = int(findings.eq(StudyFinding.POSITIVE).sum())
    negative = int(findings.eq(StudyFinding.NEGATIVE).sum())
    explicit_conflict = int(rows.outcome.eq('conflicting').sum())
    return dict(studies=rows, consensus=None if labels.empty else labels.iloc[0].to_dict(),
                mechanisms=state.mechanisms.loc[state.mechanisms.entity.isin([herb_id, drug_id])].copy(),
                evidence_status=('unavailable' if not state.evidence_available else 'published' if len(rows) else 'no_studies'),
                positive_findings=positive, negative_findings=negative,
                inconclusive_findings=int(findings.eq(StudyFinding.NULL).sum()),
                explicitly_conflicting_findings=explicit_conflict,
                conflicting_findings=bool((positive and negative) or explicit_conflict))


def render_research_readiness(st, root: Path):
    state = load_research_state(root)
    st.title('Herb–drug interaction research')
    st.warning('Research readiness: predictions are unavailable. Real scientific predictions have not been validated.')
    st.caption('Read-only evidence inspection. No models are trained when this app starts.')
    st.subheader('Readiness blockers')
    for blocker in state.blockers:
        st.write(f'• {blocker}')
    st.write(f'Independent consensus rows: {len(state.consensus)}')
    if state.verified_mechanistic_rows is not None:
        st.write(f'Mechanistic provenance checks: {state.verified_mechanistic_rows} rows pass; '
                 f'{state.unverified_mechanistic_rows} rows fail.')
        st.caption('Structural provenance checks do not independently authenticate citations or clinical validity.')
    st.button('Scientific prediction unavailable', disabled=True)
    st.info('Uncertainty and model abstention are unavailable: no fitted calibration/conformal layer is integrated. '
            'Withholding predictions here is a readiness safeguard, not a model abstention result.')
    pairs = pd.concat([state.studies[KEYS], state.consensus[KEYS]]).drop_duplicates()
    if pairs.empty:
        st.info('No validated pair identifiers are available for evidence inspection.')
        return
    st.subheader('Inspect reported evidence')
    herb = st.selectbox('Herb identifier', sorted(pairs.herb_id.unique()), key='research_herb')
    drug = st.selectbox('Drug identifier', sorted(pairs.drug_id.unique()), key='research_drug')
    pair = describe_pair(state, herb, drug)
    consensus = pair['consensus']
    st.write('Independent consensus verdict: ' + ('unavailable' if consensus is None else consensus['consensus_label']))
    st.caption('interaction, no_interaction and disputed are external consensus categories. '
               'Disputed means conflicting consensus; absent labels and absent studies do not mean no_interaction.')
    if consensus is not None:
        st.dataframe(pd.DataFrame([consensus]), hide_index=True)
    if pair['evidence_status'] == 'unavailable':
        st.error('Evidence could not be validated; no claim of zero studies is made.')
    elif pair['evidence_status'] == 'no_studies':
        st.info('No studies for this pair in the loaded table. This is not evidence of no interaction. '
                'A mechanistic-prior prediction requires verified inputs and a fitted Task A model.')
    else:
        st.write(f"Reported findings: {pair['positive_findings']} interaction-present, "
                 f"{pair['negative_findings']} no-interaction-observed, "
                 f"{pair['inconclusive_findings']} inconclusive/unknown/conflicting.")
        if pair['conflicting_findings']:
            st.warning('Conflicting study findings are present. This does not assign a disputed consensus label.')
        columns = [c for c in ['source_id', 'study_id', 'outcome', 'evidence_type', 'species',
                               'route', 'preparation', 'dose', 'source_url_or_doi'] if c in pair['studies']]
        st.dataframe(pair['studies'][columns], hide_index=True)
    st.caption('Study counts are descriptive; no learned reliability weights, contradiction score, '
               'confidence percentage or clinical verdict is inferred from them.')
    st.subheader('Mechanistic curation records')
    st.caption('Recorded inputs, not verified mechanisms or model predictions. Source and citation fields '
               'are shown as recorded, not independently authenticated. not_tested and missing values '
               'do not establish no effect; a recorded status alone does not establish verification.')
    if pair['mechanisms'].empty:
        st.info('No loadable mechanistic records for these entities. No mechanism or absence of effect is inferred.')
    else:
        st.dataframe(pair['mechanisms'], hide_index=True)
