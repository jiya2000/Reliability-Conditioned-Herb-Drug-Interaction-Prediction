"""Presentation integration: real Streamlit rendering, no scientific inference."""
from html.parser import HTMLParser
from pathlib import Path

import pytest

from app.project_presentation import DOCUMENTS, TITLE

ROOT = Path(__file__).resolve().parent.parent


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = set(), []
        self.scripts = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        if tag == 'a':
            self.links.append(attrs.get('href', ''))
        self.scripts += tag == 'script'


def test_navigation_and_document_download_targets_exist():
    markup = (ROOT / 'app/assets/research_presentation.html').read_text()
    parsed = Links()
    parsed.feed(markup)
    assert parsed.links
    assert all(href.startswith('#') and href[1:] in parsed.ids | {'documents'} for href in parsed.links)
    assert parsed.scripts == 0
    assert TITLE in markup and 'prefers-reduced-motion' in markup and ':focus-visible' in markup
    assert 'consensus dataset is empty' in markup and 'mechanistic records remain unverified' in markup
    for _, relative in DOCUMENTS:
        assert (ROOT / relative).is_file(), relative


@pytest.mark.parametrize('entry', ['app/interactive_app.py', 'app/frontend.py'])
def test_real_streamlit_views_and_downloads_never_fit(entry, monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    from sklearn.pipeline import Pipeline
    import src.training.task_a
    import src.training.task_b

    def forbid(*args, **kwargs):
        pytest.fail('Presentation/explorer attempted scientific fitting or prediction')
    monkeypatch.setattr(Pipeline, 'fit', forbid)
    monkeypatch.setattr(Pipeline, 'predict_proba', forbid)
    monkeypatch.setattr(src.training.task_a, 'run_task_a', forbid)
    monkeypatch.setattr(src.training.task_b, 'fit_task_b', forbid)
    at = AppTest.from_file(str(ROOT / entry)).run(timeout=30)
    assert not at.exception
    assert at.selectbox(key='research_view').value == 'Project presentation'
    assert TITLE in at.get('html')[0].proto.body
    assert len(at.get('download_button')) == len(DOCUMENTS)
    assert any(s.value == 'Read the project documents' for s in at.subheader)
    at.selectbox(key='research_view').set_value('Evidence explorer').run(timeout=30)
    assert not at.exception
    assert any('Research readiness' in w.value for w in at.warning)
    assert any(button.disabled for button in at.button)
    at.selectbox(key='research_view').set_value('Project presentation').run(timeout=30)
    assert not at.exception and len(at.get('download_button')) == len(DOCUMENTS)
