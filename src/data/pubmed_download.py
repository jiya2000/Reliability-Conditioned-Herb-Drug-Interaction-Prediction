"""Download authoritative PubMed records for audit, never interaction labels."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import re
from urllib.parse import urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from src.data.review_packet import _table

ENDPOINT = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'


def parse(raw, requested):
    root = ET.fromstring(raw)
    if root.tag != 'PubmedArticleSet' or root.find('.//ERROR') is not None:
        raise ValueError('Invalid PubMed response')
    records = {}
    for article in root.findall('PubmedArticle'):
        pmid = article.findtext('./MedlineCitation/PMID')
        if pmid not in requested or pmid in records:
            raise ValueError('Unexpected or duplicate PMID in response')
        title = article.find('./MedlineCitation/Article/ArticleTitle')
        dois = [v.text for v in article.findall('./PubmedData/ArticleIdList/ArticleId')
                if v.get('IdType') == 'doi' and v.text]
        abstracts = article.findall('./MedlineCitation/Article/Abstract/AbstractText')
        records[pmid] = dict(pmid=pmid, title=''.join(title.itertext()) if title is not None else '',
                             dois=dois, abstract_available=bool(abstracts),
                             publication_types=[v.text for v in article.findall('.//PublicationType')],
                             abstract_sections=[dict(label=v.get('Label',''),text=''.join(v.itertext()))
                                                for v in abstracts])
    return records


def download(input_path, output):
    raw = Path(input_path).read_bytes()
    table = _table(raw)
    if not {'source_id','source_url_or_doi'} <= set(table):
        raise ValueError('Input requires source_id and source_url_or_doi')
    pmids = sorted({v.removeprefix('PUBMED_') for v in table.source_id if re.fullmatch(r'PUBMED_[0-9]+',v)})
    if not pmids or len(pmids)>200:
        raise ValueError('Require 1–200 explicit unique PMIDs per audit request')
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    url=ENDPOINT+'?'+urlencode(dict(db='pubmed',id=','.join(pmids),retmode='xml',tool='hdi_source_audit'))
    with urlopen(url,timeout=60) as response:
        xml=response.read(16*1024*1024+1)
    if len(xml)>16*1024*1024:
        raise ValueError('PubMed response too large')
    records=parse(xml,set(pmids))
    findings=[]
    for index,row in table.iterrows():
        pmid=row.source_id.removeprefix('PUBMED_') if re.fullmatch(r'PUBMED_[0-9]+',row.source_id) else None
        record=records.get(pmid)
        reference=row.source_url_or_doi.strip()
        doi=re.sub(r'^https?://(?:dx\.)?doi\.org/','',reference,flags=re.I).lower()
        status=('non_pubmed_source_requires_separate_audit' if pmid is None else
                'pmid_not_returned' if record is None else
                'doi_matches_metadata_only' if doi in {v.lower() for v in record['dois']} else
                'doi_mismatch_or_uncomparable_requires_review')
        findings.append(dict(row=int(index)+1,source_id=row.source_id,record_status=status,
                             retrieved_title=record['title'] if record else None,
                             input_reference=reference,retrieved_dois=record['dois'] if record else []))
    report=dict(retrieved_at=datetime.now(timezone.utc).isoformat(),endpoint=ENDPOINT,
                input_sha256=sha256(raw).hexdigest(),xml_sha256=sha256(xml).hexdigest(),
                requested_pmids=len(pmids),returned_pmids=len(records),
                missing_pmids=sorted(set(pmids)-set(records)),findings=findings,
                eligible=False,limitations='Metadata retrieval is not scientific claim validation. '
                'DOI matching does not verify herb/drug mapping, outcome, study design or independence.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'pubmed.xml').write_bytes(xml)
    (output/'input_snapshot.csv').write_bytes(raw)
    (output/'publication_records.json').write_text(json.dumps(records,indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=download(args.input,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='findings'},indent=2))


if __name__=='__main__':
    main()
