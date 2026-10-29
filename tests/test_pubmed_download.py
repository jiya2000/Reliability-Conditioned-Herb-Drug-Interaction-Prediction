"""Synthetic PubMed parsing and network-boundary checks."""
import pytest
from src.data.pubmed_download import parse

XML=b'''<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID>
<Article><ArticleTitle>Synthetic <i>title</i></ArticleTitle><Abstract>
<AbstractText Label="RESULTS">Synthetic finding.</AbstractText></Abstract></Article>
</MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.0000/synthetic</ArticleId>
</ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>'''


def test_metadata_not_labels():
    row=parse(XML,{'123'})['123']
    assert row['title']=='Synthetic title'
    assert row['abstract_available'] is True
    assert row['dois']==['10.0000/synthetic']
    assert 'consensus_label' not in row and 'eligible' not in row


@pytest.mark.parametrize('xml,requested',[(b'<ERROR>failure</ERROR>',{'123'}),
    (XML,{'456'}),(XML.replace(b'</PubmedArticleSet>',XML.split(b'<PubmedArticleSet>')[1]),{'123'})])
def test_invalid_response_rejected(xml,requested):
    with pytest.raises(ValueError):parse(xml,requested)
