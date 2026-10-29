"""Download a complete, explicitly filtered ChEMBL assay dataset for research.

This is a compound-assay dataset, not clinical herb-drug reference labels.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import urlopen
from urllib.error import HTTPError, URLError

import pandas as pd
from src.data.chembl_assay_review import review_rows

BASE = 'https://www.ebi.ac.uk'
API = BASE + '/chembl/api/data/'
TARGETS = {'CYP3A4':'P08684','CYP2D6':'P10635','CYP2C9':'P11712',
           'CYP2C19':'P33261','CYP1A2':'P05177','ABCB1':'P08183'}
FIELDS = ('activity_id','molecule_chembl_id','molecule_pref_name','canonical_smiles',
          'target_chembl_id','target_pref_name','target_organism','assay_chembl_id',
          'assay_description','assay_type','assay_variant_accession','assay_variant_mutation',
          'document_chembl_id','document_year','standard_type','standard_relation',
          'standard_value','standard_units','pchembl_value','data_validity_comment',
          'potential_duplicate','src_id')


def trusted_url(url):
    parts=urlsplit(url)
    if (parts.scheme!='https' or parts.netloc!='www.ebi.ac.uk' or
            not parts.path.startswith('/chembl/api/data/')):
        raise ValueError('Nonofficial ChEMBL pagination URL')
    return url


class Capture:
    def __init__(self, directory):
        self.directory=directory
        self.requests=[]

    def get(self,url):
        trusted_url(url)
        for attempt in range(3):
            try:
                time.sleep(0.35)
                with urlopen(url,timeout=60) as response:
                    trusted_url(response.url)
                    raw=response.read(16*1024*1024+1)
                break
            except (HTTPError,URLError,TimeoutError):
                if attempt==2:raise
                time.sleep(2*(attempt+1))
        if len(raw)>16*1024*1024:raise ValueError('Oversize API page')
        result=json.loads(raw)
        name=f'response_{len(self.requests):04d}.json'
        (self.directory/name).write_bytes(raw)
        self.requests.append(dict(url=url,path='raw/'+name,sha256=sha256(raw).hexdigest()))
        return result

    def pages(self,resource,query):
        url=API+resource+'.json?'+urlencode(dict(query,limit=1000))
        seen=set(); rows=[]; total=None
        while url:
            if url in seen:raise ValueError('Repeated pagination URL')
            seen.add(url)
            page=self.get(url)
            metadata=page['page_meta']
            if total is None:total=metadata['total_count']
            if total!=metadata['total_count']:raise ValueError('Source count changed during retrieval')
            rows.extend(page[resource+'s' if resource!='activity' else 'activities'])
            next_url=metadata['next']
            url=(BASE+next_url if next_url and next_url.startswith('/') else next_url)
        if len(rows)!=total:raise ValueError('Incomplete API pagination')
        return rows


def select_targets(rows):
    result={}
    for name,accession in TARGETS.items():
        candidates=[r for r in rows if r.get('organism')=='Homo sapiens' and
                    r.get('target_type')=='SINGLE PROTEIN' and
                    len(r.get('target_components',[]))==1 and
                    r['target_components'][0].get('accession')==accession]
        if len(candidates)!=1:raise ValueError(f'Ambiguous or missing target: {name}')
        result[name]=candidates[0]['target_chembl_id']
    return result


def normalize(activities,targets,documents):
    by_id={v:k for k,v in targets.items()}
    rows=review_rows({'activities':activities})
    source={r['activity_id']:r for r in activities}
    for row in rows:
        original=source[row['activity_id']]
        if row['target_chembl_id'] not in by_id:raise ValueError('Unexpected target')
        doc=documents.get(row['document_chembl_id'])
        if doc is None:raise ValueError('Missing source document metadata')
        row.update(target_symbol=by_id[row['target_chembl_id']],
                   canonical_smiles=original.get('canonical_smiles'),
                   assay_description=original.get('assay_description'),assay_type=original.get('assay_type'),
                   pchembl_value=original.get('pchembl_value'),document_doi=doc.get('doi'),
                   document_pmid=doc.get('pubmed_id'),document_title=doc.get('title'),
                   document_year=doc.get('year'),assay_url=API+'assay/'+row['assay_chembl_id']+'.json')
        flags=[row['review_flags']] if row['review_flags'] else []
        if not row['canonical_smiles']:flags.append('missing_structure')
        if original.get('assay_variant_mutation'):flags.append('mutant_assay_requires_review')
        if not doc.get('doi') and not doc.get('pubmed_id'):flags.append('document_without_doi_or_pmid')
        row['review_flags']=';'.join(flags)
        row['structurally_usable_for_assay_research']=not flags and row['exact_value_uM'] is not None
        # No source row is promoted to whole-herb features or clinical labels.
    return rows


def download(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'raw').mkdir()
    capture=Capture(output/'raw')
    # Until manifest.json exists, this is an incomplete download, never a dataset.
    status=capture.get(API+'status.json')
    target_rows=capture.pages('target',{'target_components__accession__in':','.join(TARGETS.values())})
    targets=select_targets(target_rows)
    activities=[]
    query=dict(standard_type='IC50',standard_relation='=',standard_units='nM',
               target_organism='Homo sapiens',assay_confidence_score=9,
               pchembl_value__isnull='false',only=','.join(FIELDS),order_by='activity_id')
    counts={}
    for name,target in targets.items():
        rows=capture.pages('activity',dict(query,target_chembl_id=target))
        if any(r['target_chembl_id']!=target for r in rows):raise ValueError('Target filter mismatch')
        activities.extend(rows);counts[name]=len(rows)
        print(f'{name}: {len(rows)} complete source records',flush=True)
    docs={}
    ids=sorted({r['document_chembl_id'] for r in activities})
    for start in range(0,len(ids),100):
        for row in capture.pages('document',{'document_chembl_id__in':','.join(ids[start:start+100])}):
            docs[row['document_chembl_id']]=row
    rows=normalize(activities,targets,docs)
    if capture.get(API+'status.json')!=status:raise ValueError('ChEMBL status changed during capture')
    frame=pd.DataFrame(rows)
    if frame.empty:raise ValueError('No assay records returned')
    frame.to_csv(output/'assay_records.csv',index=False)
    usable=frame[frame.structurally_usable_for_assay_research]
    usable.to_csv(output/'assay_research_candidates.csv',index=False)
    manifest=dict(schema='hdi.chembl-assay-source/v1',source_status=status,
                  retrieved_at=datetime.now(timezone.utc).isoformat(),filters=query,
                  target_accessions=TARGETS,target_chembl_ids=targets,counts_by_target=counts,
                  records=len(frame),unique_compounds=frame.molecule_chembl_id.nunique(),
                  source_documents=len(docs),structurally_usable_records=len(usable),
                  complete_for_declared_queries=True,task_a_eligible=False,task_b_eligible=False,
                  limitations='Exact IC50 compound assays, not whole-herb clinical labels, induction or clearance fractions. '
                  'Structural screening is not expert scientific approval; repeated experiments are retained, not independent samples.',
                  captures=capture.requests,outputs={name:sha256((output/name).read_bytes()).hexdigest()
                      for name in ['assay_records.csv','assay_research_candidates.csv']})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify_dataset(directory):
    """Detect changed/incomplete local captures; never confer scientific eligibility."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if (manifest.get('schema')!='hdi.chembl-assay-source/v1' or
            manifest.get('complete_for_declared_queries') is not True or
            manifest.get('task_a_eligible') is not False or
            manifest.get('task_b_eligible') is not False):
        raise ValueError('Invalid assay-source manifest')
    expected={'assay_records.csv','assay_research_candidates.csv'}
    if set(manifest['outputs'])!=expected:
        raise ValueError('Invalid assay-source outputs')
    for capture in manifest['captures']:
        path=Path(capture['path'])
        if len(path.parts)!=2 or path.parts[0]!='raw' or path.suffix!='.json':
            raise ValueError('Invalid capture path')
        trusted_url(capture['url'])
        if sha256((directory/path).read_bytes()).hexdigest()!=capture['sha256']:
            raise ValueError('Source capture changed')
    for name,digest in manifest['outputs'].items():
        if sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Dataset output changed')
    frame=pd.read_csv(directory/'assay_records.csv',keep_default_na=False)
    candidates=pd.read_csv(directory/'assay_research_candidates.csv',keep_default_na=False)
    if (len(frame)!=manifest['records'] or frame.activity_id.duplicated().any() or
            len(candidates)!=manifest['structurally_usable_records'] or
            not set(candidates.activity_id)<=set(frame.activity_id)):
        raise ValueError('Dataset identity/count mismatch')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=download(args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ['captures','filters']},indent=2))


if __name__=='__main__':main()
