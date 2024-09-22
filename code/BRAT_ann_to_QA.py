import os
import re
import random
import json
import argparse
import logging
from typing import List, Dict
from brat_records_class import BRAT_record
from utils import generate_unique_id, save_json

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert BRAT annotations to QA dataset")
    parser.add_argument("--source", default="../BRAT_annotations", help="Path to the source directory")
    parser.add_argument("--version", default="v2", help="Version of the test dataset")
    parser.add_argument("--savename", default="TE_QA_dataset.json", help="Saving name of the test dataset")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for unanswerable questions")
    return parser.parse_args()

def get_synonyms(target: str, synonyms_list: List[List[str]]) -> List[str]:
    """Find synonyms for a given target from a list of synonym groups."""
    for synonyms in synonyms_list:
        if target in synonyms:
            return synonyms
    return [target]

def process_entities(annotation_file: List[str]) -> Dict[str, List[str]]:
    """Process entities from the annotation file."""
    entities = [re.split(r'[\s\t]+', line.strip(), 4) for line in annotation_file if line.startswith("T")]
    return {entity[0]: entity[2:] for entity in entities}

def process_events(annotation_file: List[str]) -> List[Dict[str, str]]:
    """Process events from the annotation file."""
    events_split = [re.split(r'[\s\t]+', line.strip(), 4) for line in annotation_file if line.startswith("E")]
    events_list = []
    for event in events_split:
        event_dict = {}
        for kv in event:
            if ":" in kv:
                key, value = kv.split(":")
                event_dict[key] = value
        events_list.append(event_dict)
    return events_list

def process_relations(annotation_file: List[str], entities: Dict[str, List[str]]) -> List[List[str]]:
    """Process relations from the annotation file."""
    relations = [line for line in annotation_file if line.startswith("R")]
    synonyms_list = []
    for relation in relations:
        arg1, arg2 = relation.split("\t")[1].split(" ")[1:]
        arg1, arg2 = arg1.split(":")[1], arg2.split(":")[1]
        name1, name2 = entities[arg1][2], entities[arg2][2]
        for synonyms in synonyms_list:
            if name1 in synonyms or name2 in synonyms:
                synonyms.extend([name1, name2])
                break
        else:
            synonyms_list.append([name1, name2])
    return synonyms_list

def from_annotation_file_to_records(annotation_file: List[str], context: str) -> List[BRAT_record]:
    """Convert a BRAT annotation file to a list of BRAT_record objects."""
    entities = process_entities(annotation_file)
    events_list = process_events(annotation_file)
    synonyms_list = process_relations(annotation_file, entities)

    start_indices = {entity_vals[2]: entity_vals[0] for entity_vals in entities.values()}

    logging.debug(f"Entities: {entities}")
    logging.debug(f"Events: {events_list}")
    logging.debug(f"Synonyms: {synonyms_list}")
    
    records = []
    for event in events_list:
        try:
            value = entities[event['Value']][2]
            temp = entities[event['temp']][2]
            spec = entities[event['spec']][2]
            spec_synonyms = get_synonyms(spec, synonyms_list)
            cem = entities[event['cem']][2]
            cem_synonyms = get_synonyms(cem, synonyms_list)
            start_indices_for_record = {entry: int(start_indices[entry]) for entry in [value, temp] + spec_synonyms + cem_synonyms}
            
            record = BRAT_record(value, spec_synonyms, cem_synonyms, temp, context, start_indices_for_record)
            records.append(record)
        except KeyError as e:
            logging.error(f"KeyError when processing event: {event}. Error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error when processing event: {event}. Error: {e}")

    return records

def main():
    setup_logging()
    args = parse_arguments()

    random.seed(42)  # for reproducibility
    QA_test_dataset = {"data": [], "version": args.version}

    pattern = r'[–−—―]'

    for annotation_file_name in sorted([f for f in os.listdir(args.source) if f.endswith(".ann")]):
        annotation_file_path = os.path.join(args.source, annotation_file_name)
        logging.info(f"Processing file: {os.path.basename(annotation_file_path)}")

        try:
            with open(annotation_file_path, "r") as f:
                annotation_file = [re.sub(pattern, '-', line) for line in f]

            context_filename = annotation_file_path.replace(".ann", ".txt")
            with open(context_filename, "r") as f:
                context = f.read()

            records = from_annotation_file_to_records(annotation_file, context)
            
            if not records:
                logging.warning(f"No records found in {annotation_file_name}")
                continue

            entry_dict = {'title': annotation_file_name, 'paragraphs': [{'context': context, 'qas': []}]}
            
            for record in records:
                questions_set, answers_set, answerstarts_set = record.to_QA_for_test_dataset()

                for question, answers, answerstarts in zip(questions_set, answers_set, answerstarts_set):
                    uuid_ = generate_unique_id([q['id'] for q in entry_dict['paragraphs'][0]['qas']])
                    answers_entry = [{'text': answer, 'answer_start': start} for answer, start in zip(answers, answerstarts)]
                    entry_dict['paragraphs'][0]['qas'].append({'question': question, 'id': uuid_, 'answers': answers_entry})

                if args.version == 'v2':
                    try:
                        unanswerable_questions, _, _ = record.to_unanswerable_QA_for_test_dataset(context)
                        for question in (uq for uq in unanswerable_questions if random.random() <= args.threshold):
                            uuid_ = generate_unique_id([q['id'] for q in entry_dict['paragraphs'][0]['qas']])
                            entry_dict['paragraphs'][0]['qas'].append({'question': question, 'id': uuid_, 'answers': [{'text': "", 'answer_start': -1}]})
                    except Exception as e:
                        logging.error(f"Error generating unanswerable questions for {record.specifier} in {annotation_file_name}: {e}")

            QA_test_dataset["data"].append(entry_dict)

        except Exception as e:
            logging.error(f"Error processing file {annotation_file_name}: {e}")

    # Calculate and log statistics
    total_questions = sum(len(entry['paragraphs'][0]['qas']) for entry in QA_test_dataset['data'])
    answerable_questions = sum(len([q for q in entry['paragraphs'][0]['qas'] if q['answers'][0]['text']]) for entry in QA_test_dataset['data'])
    unanswerable_questions = total_questions - answerable_questions

    logging.info(f"Number of contexts: {len(QA_test_dataset['data'])}")
    logging.info(f"Total questions: {total_questions}")
    logging.info(f"Answerable questions: {answerable_questions}")
    logging.info(f"Unanswerable questions: {unanswerable_questions}")
    logging.info(f"Split: {answerable_questions/total_questions:.1%} answerable, {unanswerable_questions/total_questions:.1%} unanswerable")

    save_name = args.savename

    if os.path.exists(save_name):
        response = input(f"File {save_name} already exists. Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            logging.info("File not saved")
            return

    save_json(QA_test_dataset, save_name)
    logging.info(f"Test dataset saved to {save_name}")

if __name__ == "__main__":
    main()