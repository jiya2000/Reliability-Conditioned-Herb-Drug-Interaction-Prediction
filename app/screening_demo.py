"""Presentation interface for evidence-linked herb–drug research screening."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.assay.predict import load_experiment
from src.assay.screening import load_catalog, screen_pair


@st.cache_resource(show_spinner='Checking and loading trained assay models…')
def cached_experiment(directory, report_bytes):
    # Report bytes are part of the cache identity; no fitting happens here.
    return load_experiment(directory)


def render_screening_demo(root):
    root = Path(root)
    st.title('Herb–Drug Interaction Research System')
    st.write('Trace a botanical constituent through a trained assay model, a drug pathway and published human evidence.')
    st.caption('Research demonstration · Assay ML + source traceability + mechanism checks + explicit abstention')
    catalog = load_catalog(root)
    directory = root / 'artifacts/assay/scaffold'
    try:
        bundle, report = cached_experiment(str(directory), (directory / 'report.json').read_bytes())
    except (OSError, ValueError, ImportError) as error:
        st.error(f'Trained assay model unavailable: {error}')
        st.code('venv/bin/python -m src.assay.experiment --output artifacts/assay/scaffold')
        return
    metrics = st.columns(3)
    metrics[0].metric('Assay targets', len(report['targets']))
    metrics[1].metric('Scaffold test MAE · pIC50', f"{report['selected_micro_test']['mae']:.3f}")
    metrics[2].metric('Held-out assay pairs', report['selected_micro_test']['n'])
    with st.sidebar:
        st.header('Screen a pair')
        herb = st.selectbox('Botanical', list(catalog['herbs']), key='screen_herb')
        drug = st.selectbox('Drug', list(catalog['drugs']), key='screen_drug')
        context_label = st.radio('Evidence context', ['Inspect published study conditions', 'Different or unknown preparation / dose'])
        add_duplicate = st.checkbox('Demonstrate duplicate-citation handling', help='Repeats an existing catalog record to test deduplication; adds no new evidence.')
        st.caption('Try St John’s wort + Digoxin, Echinacea + Digoxin, then Turmeric + Digoxin.')
    if add_duplicate:
        matches = [r for r in catalog['studies'] if r['herb'] == herb and r['drug'] == drug]
        if matches:
            catalog['studies'].append(dict(matches[0]))
    exposure = None
    if herb == 'Turmeric':
        with st.expander('Optional exposure sensitivity experiment'):
            st.caption('Assume a constituent concentration to inspect concentration/potency ratios. This is not a patient exposure estimate or clinical risk score.')
            if st.checkbox('Use an illustrative unbound concentration'):
                exposure = st.number_input('Assumed unbound constituent concentration (µM)', min_value=0., value=.01, format='%.4f')
    try:
        result = screen_pair(catalog, bundle, herb, drug,
                             'published' if context_label.startswith('Inspect') else 'different_or_unknown', exposure)
    except ValueError as error:
        st.error(str(error))
        return
    st.subheader(f'{herb} + {drug}')
    st.info(result['evidence']['finding'])
    st.caption('This is a retrieval of published findings, not a prediction of the clinical outcome.')
    if result['context'] != 'published':
        st.warning('The selected context differs: the cited study result cannot be carried over to this preparation or dose.')
    evidence_tab, model_tab, trace_tab, validation_tab = st.tabs(['Human evidence', 'Live molecular model', 'Explanation & audit', 'Model validation'])
    with evidence_tab:
        if not result['evidence']['studies']:
            st.warning('No matching study in this small catalog. This does not mean no interaction exists.')
        for row in result['evidence']['studies']:
            with st.container(border=True):
                st.markdown(f"**[PMID {row['pmid']}]({row['url']}) · {row['design']}**")
                st.write(row['summary'])
                st.caption(row['context'])
                if row['ratio'] is not None:
                    st.metric('Observed study exposure ratio · coadministration/comparator', f"{row['ratio']:.2f}")
                    st.caption(row['endpoint'] + ' · ' + row['ratio_basis'])
                    if row['ratio_ci']:
                        st.caption(f"Reported-ratio-derived 95% interval: {row['ratio_ci'][0]:.2f}–{row['ratio_ci'][1]:.2f}")
                st.caption(row['review_status'])
        st.caption(f"{result['evidence']['raw_records']} displayed input records → {result['evidence']['unique_publications']} unique publication IDs; "
                   f"{result['evidence']['duplicate_records_removed']} duplicate records removed. Cohort independence remains unverified.")
    with model_tab:
        if result['full_assay_profile']:
            st.write(f"Live inference for **{result['constituent']}**, a selected constituent of {herb}.")
            if result['mechanism'] == 'induction':
                st.warning(result['mechanism_note'] + ' These inhibition estimates do not explain or predict the induction-driven clinical finding.')
            frame = pd.DataFrame(result['full_assay_profile']['predictions'])
            st.bar_chart(frame.set_index('target')[['predicted_pic50']])
            st.dataframe(frame[['target', 'predicted_pic50', 'lower_pic50', 'upper_pic50', 'implied_ic50_uM', 'nearest_training_tanimoto', 'seen_in_training']], hide_index=True)
            st.caption('Five CYP enzymes + ABCB1 transporter. Larger pIC50 means stronger assay potency. Estimates are for the compound, not the whole herb.')
            st.write('Drug-linked pathway estimates')
            st.dataframe(result['assay_estimates'], hide_index=True)
            st.caption('Similarity is descriptive; training-compound predictions are illustrations, not fresh validation.')
            with st.expander('Structure and mapping sources'):
                st.code(catalog['herbs'][herb]['smiles'])
                st.write(catalog['herbs'][herb]['structure_note'])
                for source in result['sources']:
                    if source:
                        st.markdown(f'[Source]({source})')
        else:
            st.write('No representative constituent structure has been selected. The system shows the available human evidence without inventing molecular inputs.')
    with trace_tab:
        st.code('Botanical → source-linked constituent → molecular fingerprint → assay estimate\nDrug → cited pathway mapping\nPair → source-checked human findings → duplicate removal\nMechanism + context checks → explanation + clinical abstention')
        st.warning(result['clinical_prediction'])
        for missing in result['missing_requirements']:
            st.write('• ' + missing)
        st.caption('Distinctive implementation: evidence provenance, duplicate handling, mechanism mismatch detection and uncertainty are visible together. Scientific novelty has not been established.')
    with validation_tab:
        rows = []
        for target, item in report['targets'].items():
            score = item['test'][item['selected_model']]
            rows.append(dict(target=target, model=item['selected_model'], **score,
                             interval_coverage=item['interval']['test_coverage']))
        st.dataframe(rows, hide_index=True)
        other = root / 'results/assay_source.json'
        if other.exists():
            source_report = json.loads(other.read_text())
            st.metric('Publication-holdout MAE · pIC50', f"{source_report['selected_micro_test']['mae']:.3f}")
        st.caption('The source holdout performs worse. Several source-holdout R² values are negative; interval coverage is not guaranteed under domain shift. No clinical accuracy is reported.')
    st.download_button('Download this screening report', json.dumps(result, indent=2, allow_nan=False),
                       file_name='herb_drug_research_screen.json', mime='application/json')
