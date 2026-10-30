"""Explicitly separate assay results and molecular inference from clinical tasks."""
from pathlib import Path
import json


def render_assay_dashboard(st, root: Path):
    st.header('Official-data compound assay experiment')
    st.info('Predicts compound–target assay potency. Clinical herb–drug verdicts remain unavailable.')
    base = Path(root) / 'artifacts' / 'assay'
    available = sorted(p.parent for p in base.glob('*/report.json')) if base.exists() else []
    if not available:
        st.write('No completed experiment is installed. Train one with the command below.')
        st.code('venv/bin/python -m src.assay.experiment --output artifacts/assay/scaffold --split scaffold')
        return
    directory = st.selectbox('Assay experiment', available, format_func=lambda p: p.name)
    try:
        report = json.loads((directory / 'report.json').read_text())
        if report.get('complete') is not True or report.get('schema') != 'hdi.assay-results/v1':
            st.error('Experiment is incomplete.')
            return
        st.caption(f"Source: {report['source_version']} · Split: {report['config']['split']}")
        rows = []
        for target, result in report['targets'].items():
            score = result['test'][result['selected_model']]
            rows.append({'Target': target, 'Selected model': result['selected_model'],
                         'Test samples': score['n'], 'MAE (pIC50)': score['mae'],
                         'RMSE': score['rmse'], 'R²': score['r2'],
                         'Interval coverage': result['interval']['test_coverage']})
        st.dataframe(rows, use_container_width=True)
        with st.expander('Protocol, baselines and limitations'):
            st.json(report)
        st.download_button('Download experiment results', (directory / 'report.json').read_bytes(),
                           file_name=directory.name + '_report.json', mime='application/json')
        st.subheader('Explore a molecular structure')
        st.caption('Use SMILES for an individual compound. A herb name or drug pair is not a model input.')
        with st.form('assay_smiles'):
            smiles = st.text_input('Compound SMILES', value='CC(=O)Oc1ccccc1C(=O)O')
            submitted = st.form_submit_button('Estimate assay potency')
        if submitted:
            from src.assay.predict import load_experiment, predict_smiles
            bundle, _ = load_experiment(directory)
            result = predict_smiles(bundle, smiles)
            st.dataframe(result['predictions'], use_container_width=True)
            st.caption('Intervals are calibrated on held-out assays; coverage on a new chemical domain is unproven. '
                       'Tanimoto similarity is descriptive; no validated applicability threshold is claimed.')
    except (ValueError, OSError, KeyError, ImportError) as error:
        st.error(f'Assay experiment unavailable: {error}')
