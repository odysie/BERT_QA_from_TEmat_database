import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import os
import random
import argparse
import logging

from utils import recover_leading_spaces, remove_leading_and_trailing_spaces, load_json, save_json, find_key_by_value, generate_unique_id

"""
This script converts a thermoelectric database into a question-answering dataset.
It processes records from the database, generates answerable and unanswerable questions,
and creates a JSON file in SQuAD format.
"""

def setup_argparse():
    parser = argparse.ArgumentParser(description="Convert thermoelectric database to QA dataset")
    parser.add_argument("--input_csv", default="../TE_database/tedb_for_QA.csv", help="Path to input CSV file")
    parser.add_argument("--output_json", default="TE-CDE-QA.json", help="Path to output JSON file")
    parser.add_argument("--provisions_folder", default="provisions", help="Path to provisions folder")
    parser.add_argument("--version", choices=["v1", "v2"], default="v2", help="Version of the dataset")
    parser.add_argument("--unanswerable_percentage", type=float, default=0.5, help="Percentage of unanswerable questions")
    parser.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Logging level")
    return parser.parse_args()

def setup_logging(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

candidates_for_leading_space_addition = r"Kcm"
candidates_for_leading_space_removal = r"K/-%:×"
candidates_for_trailing_space_removal = r"~/-<>%:×"

random.seed(42) # for reproducibility

def clean_value_and_units(units_if_any):
    units_if_any = units_if_any[:-1] if units_if_any.endswith("(") else units_if_any
    return units_if_any.strip()


def resolve_ZT_dashes(record):
    return f" {record['raw_units']}" if record['raw_units'] != "-" else ""


def get_material_answer(record):
    return (record['compound_name']).strip()


def get_value_and_units_answer(record):
    # NB leading space if there are units
    units_if_any = resolve_ZT_dashes(record)
    units_if_any = clean_value_and_units(units_if_any)
    return f"{record['raw_value'].strip()} {units_if_any}"


def get_temperature_answer(record):
    if record['raw_temp_value'] != "-":
        temperature_answer = f"{record['raw_temp_value'].strip()} {record['raw_temp_units'].strip()}"
    else:
        temperature_answer = f"{record['raw_room_temperature'].strip()}"
    return temperature_answer[:-1] if temperature_answer.endswith("(") else temperature_answer


def pick_one_from_synonyms(synonyms):
    if not isinstance(synonyms, list):
        return synonyms
    
    # return the longest entry (but not more than 3 words) that doesn't start with 'room' or ends with ')'
    filtered_synonyms = [s for s in synonyms if not s.startswith('room') and not s.endswith(')') and not (len(s.split()) > 3)]
    if not filtered_synonyms:
        return random.choice(synonyms)
    
    return max(filtered_synonyms, key=len)


def find_all_possible_answer_starts(context, answer):
    start = 0
    answer_starts = []

    while True:
        start = context.find(answer, start)
        if start == -1:  # No more occurrences
            break
        answer_starts.append(start)
        start += 1  # Move past this match

    return answer_starts


def df_record_to_QAs(record):
    """
    Convert a database record to question-answer pairs.

    Args:
        record (pd.Series): A row from the thermoelectric database.

    Returns:
        tuple: A tuple containing two lists - questions and answers.
    """

    # every record is a single occurance, no synonyms

    units_if_any = resolve_ZT_dashes(record)
    units_if_any = clean_value_and_units(units_if_any)

    A1_valunits = get_value_and_units_answer(record)
    Q2_temperature = f"At what temperature was the value of {A1_valunits} recorded?"
    A2_temperature = get_temperature_answer(record)
    Q3_specifier = f"Which property was recorded to be {A1_valunits} at {A2_temperature}?"
    A3_selected_specifier = record['specifier']
    Q4_material = f"Which material was recorded to have a {A3_selected_specifier} of {A1_valunits} at {A2_temperature}?"
    A4_material = get_material_answer(record)

    questions = [Q2_temperature, Q3_specifier, Q4_material]
    answers = [A2_temperature, A3_selected_specifier, A4_material]

    return questions, answers
        

# specifier (property) -> compound_name -> raw_value + raw_units -> temperature

# load provisions here
provisions_folder = "provisions"
specifiers_set_per_model = load_json(os.path.join(provisions_folder, "specifiers_set_per_model.json"))
models_set = load_json(os.path.join(provisions_folder, "models_set.json"))
compounds_set = load_json(os.path.join(provisions_folder, "compounds_set.json"))
value_and_units_set = load_json(os.path.join(provisions_folder, "value_and_units_set.json"))
value_and_units_set_per_model = load_json(os.path.join(provisions_folder, "value_and_units_set_per_model.json"))
temperatures_not_room = load_json(os.path.join(provisions_folder, "temperatures_not_room.json"))

# randomly choose a different specifier, compound, value and units, or temperature kind
# if the chosen one is already in the context, call the function Recursively to choose another one
def get_unfindable_datapoint(datapoint, datapoint_kind, context):
    """
    Generate an unfindable datapoint for creating unanswerable questions.

    Args:
        datapoint (str): The original datapoint.
        datapoint_kind (str): The type of datapoint ('specifier', 'compound', 'value_and_units', or 'temperature').
        context (str): The context in which the datapoint should not be found.

    Returns:
        str: An unfindable datapoint of the specified kind.
    """

    # get a specifier from a different model (because a specifier from the same model should still have an answer?)
    if datapoint_kind == "specifier":
        try:
            model_from_specifier = [model for model in models_set if datapoint.strip() in specifiers_set_per_model[model]][0]
        except IndexError:
            print(f"Specifier >{datapoint}< not found in any model's specifiers. (check specifiers_set_per_model)")
            return "None"
        different_model = random.choice([m for m in models_set if m != model_from_specifier])
        different_specifier = random.choice(specifiers_set_per_model[different_model])
        if different_specifier in context:
            # print(f"Found specifier >{different_specifier}< (changed from {datapoint}) in '{context}' (should not be there)")
            get_unfindable_datapoint(datapoint, datapoint_kind, context)
        else:
            return different_specifier
        
    if datapoint_kind == "compound":
        different_compound = random.choice(compounds_set)
        if different_compound in context:
            # print(f"Found compound >{different_compound}< (changed from {datapoint}) in '{context}' (should not be there)")
            get_unfindable_datapoint(datapoint, datapoint_kind, context)
        else:
            return different_compound
        
    # get value and units from the same model (unlikely to coincide with the original one)
    if datapoint_kind == "value_and_units":
        model_from_value_and_units = find_key_by_value(datapoint, value_and_units_set_per_model)
        if not model_from_value_and_units:
            print(f"Value and units [{datapoint}] not found in any model's raw_values. Pick randomly.")
            model_from_value_and_units = random.choice(models_set)
        
        different_value_and_units = random.choice(value_and_units_set_per_model[model_from_value_and_units])
        if different_value_and_units in context:
            # print(f"Found valunits >{different_value_and_units}< (changed from {datapoint}) in '{context}' (should not be there)")
            # run again until we get a different one. Can get stuck in infinite loop?
            get_unfindable_datapoint(datapoint, datapoint_kind, context)
        else:
            return different_value_and_units
        
    if datapoint_kind == "temperature":
        # ignore the room temperature cases because they are too frequent (and it's hard to pick out their synonyms)
        different_temperature = random.choice(temperatures_not_room)
        if different_temperature in context:
            # print(f"Found temperature >{different_temperature}< (changed from {datapoint}) in '{context}' (should not be there)")
            get_unfindable_datapoint(datapoint, datapoint_kind, context)
        else:
            return different_temperature
    

# using a choice between which single datapoint to sabotage. Could also do it with possibility for multiple ones
def df_record_to_unanswerable_QAs(record):
    """
    Generate unanswerable questions from a database record.

    Args:
        record (pd.Series): A row from the thermoelectric database.

    Returns:
        tuple: A tuple containing two lists - unanswerable questions and empty answers.
    """
    context = record['context']
    
    # returns 3 unanswerable QAs per record
    unanswerable_questions = []

    # actual answers
    A1_valunits = get_value_and_units_answer(record)
    A2_temperature = get_temperature_answer(record)
    A3_selected_specifier = record['specifier']
    # material answer not needed, just remains unanswered

    # Q2: unfindable value and units -> looking for temperature
    unfindable_valunits = get_unfindable_datapoint(A1_valunits, "value_and_units", context)
    unanswerable_Q2_temperature = f"At what temperature was the value of {unfindable_valunits} recorded?"
    unanswerable_questions.append(unanswerable_Q2_temperature)
    # print(f"unanswerable_Q2_temperature: {unanswerable_Q2_temperature}")

    # Q3: unfindable value and units, or temperature -> looking for specifier
    # choose to sabotage valunits or temperature
    sabotage_choice = random.choice([1,2])
    if sabotage_choice == 1:
        unfindable_valunits = get_unfindable_datapoint(A1_valunits, "value_and_units", context)
        unanswerable_Q3_specifier = f"Which property was recorded to be {unfindable_valunits} at {A2_temperature}?"
    else:
        unfindable_temperature = get_unfindable_datapoint(A2_temperature, "temperature", context)
        unanswerable_Q3_specifier = f"Which property was recorded to be {A1_valunits} at {unfindable_temperature}?"
    unanswerable_questions.append(unanswerable_Q3_specifier)
    # print(f"unanswerable_Q3_specifier: {unanswerable_Q3_specifier}")

    # Q4: unfindable value and units, temperature, or specifier -> looking for material
    # choose to sabotage valunits, temperature, or specifier
    sabotage_choice = random.choice([1,2,3])
    if sabotage_choice == 1:
        unfindable_valunits = get_unfindable_datapoint(A1_valunits, "value_and_units", context)
        unanswerable_Q4_material = f"Which material was recorded to have a {A3_selected_specifier} of {unfindable_valunits} at {A2_temperature}?"
    elif sabotage_choice == 2:
        unfindable_temperature = get_unfindable_datapoint(A2_temperature, "temperature", context)
        unanswerable_Q4_material = f"Which material was recorded to have a {A3_selected_specifier} of {A1_valunits} at {unfindable_temperature}?"
    else:
        unfindable_specifier = get_unfindable_datapoint(A3_selected_specifier, "specifier", context)
        unanswerable_Q4_material = f"Which material was recorded to have a {unfindable_specifier} of {A1_valunits} at {A2_temperature}?"
    unanswerable_questions.append(unanswerable_Q4_material)
    # print(f"unanswerable_Q4_material: {unanswerable_Q4_material}")

    no_answers = ["" for _ in unanswerable_questions]
    return unanswerable_questions, no_answers


if __name__ == "__main__":
    """
    Main execution of the script. This section:
    1. Loads the thermoelectric database
    2. Processes each record to generate answerable and unanswerable questions
    3. Creates a SQuAD-format JSON file with the generated QA pairs
    4. Saves the dataset and provides statistics on the conversion process
    """
    args = setup_argparse()
    setup_logging(args.log_level)
    
    random.seed(42) # for reproducibility
    version = args.version
    percentage_of_unanswerable_questions = args.unanswerable_percentage
    
    QA_Dataset = {"data": [], "version": version}
    count_ = 0
    answers_starts_not_found = []
    answers_starts_not_found_per_property = {p: [] for p in models_set}
    number_of_answerable_questions = 0
    number_of_unanswerable_questions = 0
    number_of_multi_index_answers = 0

    logging.info(f"Number of unique compounds: {len(compounds_set)}")
    logging.info(f"Number of unique value and units: {len(value_and_units_set)}")  

    try:
        tedb = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        logging.error(f"Input CSV file not found: {args.input_csv}")
        raise
    except pd.errors.EmptyDataError:
        logging.error(f"Input CSV file is empty: {args.input_csv}")
        raise
                        
    for i in tqdm(range(len(tedb))):
        record = tedb.iloc[i]
        # this is not true. there exist duplicates. should I get a hash from the context?
        entry_dict = {'title': f'{"sentence" if version == "v1" else "paragraph"}{f"{i+1}".zfill(3)}',
                      'doi': record['doi'].replace('-', '/', 1).replace('.txt', '').replace('.html', '').replace('.xml', ''),
                      'paragraphs': []}
        
        # NB I only have one paragraph per entry (title)
        paragraphs = [{'context': "", 'qas': []}]
        
        context = record['context']

        # answerable questions:
        questions, answers = df_record_to_QAs(record)

        for question, answer in zip(questions, answers):

            # processed answers contains possible variations of the answer (e.g. with leading spaces)
            # but only one could be found in the context
            processed_answers = recover_leading_spaces(answer, candidates_for_leading_space_addition)
            processed_answers.extend(remove_leading_and_trailing_spaces(answer,
                                        candidates_for_leading_space_addition,
                                        candidates_for_leading_space_removal))
            
            found_answer = False

            for processed_answer in processed_answers:
                answer_start = context.find(processed_answer)
                if answer_start != -1:
                    # if one is found, try and find all the possible answer starts. Not optimal but the code was already set up for it
                    answer_start_options = find_all_possible_answer_starts(context, processed_answer)
                    if len(answer_start_options) > 1:
                        number_of_multi_index_answers += 1
                        # print(f"Multiple answer starts found for question: {question} with answer: {answer}")
                        # print(f"Context: {context}")
                        # print(f"Answer starts: {answer_start_options}")
                        # print()
                        answers_for_data = []
                        for answer_start_option in answer_start_options:
                            answers_for_data.append({'text': processed_answer, 'answer_start': answer_start_option})
                    else:
                        answers_for_data = [{'text': processed_answer, 'answer_start': answer_start}]

                    uuid_ = generate_unique_id([q['id'] for q in paragraphs[0]['qas']])
                    paragraphs[0]['qas'].append({'question': question, 'id': f"{uuid_}",
                                                'answers': answers_for_data,
                                                'is_impossible': False})
                    count_ += 1
                    found_answer = True
                    number_of_answerable_questions += 1
                    break

            if not found_answer:
                answers_starts_not_found.append([question, answer, context])
                answers_starts_not_found_per_property[record['model']].append([question, answer, context])

                # if record['model'] == 'ZT':
                    # ZT has the most problems, so print the failed cases
                    # print(f"Answer not found for question: {question} with answer: {answer}")
                    # print(f"Context: {context}")
                    # print()
                    

        # unanswerable questions (subset):
        if version == "v2":
            unanswerable_questions, no_answers = df_record_to_unanswerable_QAs(record)
            # random subset of unanswerable questions based on percentage (typically 50%)
            # Dynamicaly the percentage. Because unanswerable questions never fail (so always 3 per records available to choose from)
            # while answerable questions can fail if the answer_start isn't found within the context (so always fewer than 3. Currently 2.74)
            adjusted_percentage = percentage_of_unanswerable_questions * (number_of_answerable_questions - len(answers_starts_not_found)) / number_of_answerable_questions
            # manual adjustment
            # adjusted_percentage = 2.74 / 6
            random_subset_of_unanswerable_questions = (uq for uq in unanswerable_questions if random.random() <= adjusted_percentage)

            for question in random_subset_of_unanswerable_questions:
                uuid_ = generate_unique_id([q['id'] for q in paragraphs[0]['qas']])
                paragraphs[0]['qas'].append({'question': question, 'id': f"{uuid_}",
                                             'answers': [{'text': [], 'answer_start': []}],
                                             'is_impossible': True})
                number_of_unanswerable_questions += 1
                count_ += 1

        # add to dataset
        paragraphs[0]['context'] = context
        entry_dict['paragraphs'].extend(paragraphs)
        QA_Dataset['data'].append(entry_dict)

    # end of big loop ^
    # example entry
    logging.debug(f"Example entry: {QA_Dataset['data'][0]}")
    logging.info(f"Total number of questions: {count_}")
    logging.info(f"Failed answerable questions: {len(answers_starts_not_found)}. Percentage failed: {len(answers_starts_not_found)/number_of_answerable_questions*100.0:.2f}%")
    logging.info(f"Number of answerable questions: {number_of_answerable_questions}. Multiples of records: {number_of_answerable_questions/len(tedb):.2f}")
    logging.info(f"Number of unanswerable questions: {number_of_unanswerable_questions}. Multiples of records: {number_of_unanswerable_questions/len(tedb):.2f}")
    logging.info("Answers not found per property: %s", {p: len(answers_starts_not_found_per_property[p]) for p in answers_starts_not_found_per_property})
    logging.info(f"Number of multi-index answers: {number_of_multi_index_answers}")

    with open("not_found_answers_per_property.json", 'w') as f:
        json.dump(answers_starts_not_found_per_property, f, indent=4)

    # saving:
    save_name = args.output_json

    if not os.path.exists(save_name):
        save_json(QA_Dataset, save_name)
        logging.info(f"File saved: {save_name}")
        save_json(answers_starts_not_found, "not_found_answers.json")
    else:
        logging.warning(f"File {save_name} already exists")

        response = input("Do you want to overwrite the file? (y/n): ")

        if response.lower() == "y":
            save_json(QA_Dataset, save_name)
            save_json(answers_starts_not_found, "not_found_answers.json")
            logging.info("File overwritten")
        else:
            logging.info("File not saved")