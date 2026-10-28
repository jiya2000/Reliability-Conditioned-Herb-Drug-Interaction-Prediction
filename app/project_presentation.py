"""Read-only project presentation integrated into the existing Streamlit app."""
from pathlib import Path

TITLE = 'Reliability-Conditioned Herb–Drug Interaction Prediction'
DOCUMENTS = (
    ('Beginner explanation', 'explanation.md'),
    ('Research-paper draft', 'docs/research_paper_draft.md'),
    ('IDF-B review draft', 'docs/IDF-B_draft.md'),
    ('Evaluation and authorization contract', 'docs/evaluation_orchestrator.md'),
    ('Dataset release requirements', 'docs/dataset_release.md'),
    ('Task A feature contract', 'docs/task_a_feature_contract.md'),
)


def render_project_presentation(st, root: Path):
    markup = (Path(root) / 'app/assets/research_presentation.html').read_text(encoding='utf-8')
    if hasattr(st, 'html'):
        st.html(markup)
    else:  # Preserve compatibility with the existing Streamlit >=1.29 dependency.
        st.markdown(markup, unsafe_allow_html=True)
    st.subheader('Read the project documents', anchor='documents')
    st.caption('Local Markdown downloads. Drafts require scientific and institutional review.')
    for label, relative in DOCUMENTS:
        path = Path(root) / relative
        if path.is_file():
            st.download_button(label, path.read_bytes(), file_name=path.name,
                               mime='text/markdown', key='document_' + path.stem)
        else:
            st.warning(f'{label} is not available in this checkout.')
    st.caption('Research software only. No clinical recommendations or validated predictions are provided.')


def render_project_site(st, root: Path):
    view = st.selectbox('View', ['Project presentation', 'Evidence explorer'], key='research_view')
    if view == 'Evidence explorer':
        from app.research_status import render_research_readiness
        render_research_readiness(st, root)
    else:
        render_project_presentation(st, root)
