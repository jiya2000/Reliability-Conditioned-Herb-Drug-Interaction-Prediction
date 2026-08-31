import xml.etree.ElementTree as ET
import random
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_drugbank(xml_path, limit_drugs=None):
    """
    Parse drugbank.xml iteratively.
    Extract drugs, their interactions, and targets.
    """
    logger.info(f"Parsing {xml_path}")
    
    # We only care about the drug namespace
    ns = '{http://www.drugbank.ca}'
    
    drugs = {}
    interactions = []
    
    context = ET.iterparse(xml_path, events=("end",))
    
    count = 0
    for event, elem in context:
        if elem.tag == f'{ns}drug':
            if 'type' in elem.attrib and elem.attrib['type'] == 'biotech':
                elem.clear()
                continue
                
            db_id = None
            for id_elem in elem.findall(f'{ns}drugbank-id'):
                if id_elem.attrib.get('primary'):
                    db_id = id_elem.text
                    break
            
            name = elem.findtext(f'{ns}name')
            if not db_id or not name:
                elem.clear()
                continue
                
            drugs[db_id] = name
            
            # Interactions
            interactions_elem = elem.find(f'{ns}drug-interactions')
            if interactions_elem is not None:
                for inter in interactions_elem.findall(f'{ns}drug-interaction'):
                    target_id = inter.findtext(f'{ns}drugbank-id')
                    desc = inter.findtext(f'{ns}description')
                    if target_id and desc:
                        interactions.append((db_id, target_id, desc))
            
            count += 1
            if limit_drugs and count >= limit_drugs:
                break
            
            # Clear memory
            elem.clear()
            
    logger.info(f"Parsed {len(drugs)} drugs and {len(interactions)} interactions.")
    return drugs, interactions

if __name__ == "__main__":
    path = "/Users/aditisharma/.cache/kagglehub/datasets/sergeguillemart/drugbank/versions/1/drugbank.xml"
    if os.path.exists(path):
        drugs, ints = parse_drugbank(path, limit_drugs=5000)
        
        # Save to processed
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/drugbank_parsed.json", "w") as f:
            json.dump({"drugs": drugs, "interactions": ints}, f)
        print("Done parsing and saved to data/processed/drugbank_parsed.json")
    else:
        print("Path not found:", path)
