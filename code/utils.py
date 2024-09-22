import re
import os
import json
import uuid
from typing import Any, Dict, List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_metadata(filepath: str) -> Dict[str, Any]:
    """
    Load metadata from a JSON file.
    Args:
        filepath (str): Path to the metadata JSON file.
    Returns:
        Dict[str, Any]: Loaded metadata dictionary.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Metadata file not found: {filepath}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in metadata file: {filepath}")
        raise

try:
    metadata = load_metadata(os.getenv('METADATA_PATH', 'provisions/metadata_dict.json'))
except (FileNotFoundError, json.JSONDecodeError):
    logging.warning("Failed to load metadata. Some functions may not work correctly.")
    metadata = {}


def convert_article_name_to_doi(article_name: str) -> str:
    """
    Convert an article name to a DOI format.
    Args:
        article_name (str): The name of the article.
    Returns:
        str: The article name converted to a DOI-like format.
    """
    article_name = article_name.replace(".html", "").replace(".xml", "").replace(".txt", "")
    article_name = article_name.replace("article-", "")
    return article_name.replace("-", "/", 1)


def get_metadata(doi: str, info: str) -> Optional[Any]:
    """
    Retrieve metadata information for a given DOI.
    Args:
        doi (str): The DOI of the article.
        info (str): The type of metadata information to retrieve.
    Returns:
        Optional[Any]: The requested metadata information, or None if not found.
    """
    doi = doi.replace("article-", "")
    return metadata.get(doi, {}).get(info)


def generate_unique_id(existing_ids: List[str]) -> str:
    """
    Generate a new unique custom ID.
    Args:
        existing_ids (List[str]): A list of existing IDs to check against.
    Returns:
        str: A new unique ID.
    """
    while True:
        new_id = uuid.uuid4().hex
        if new_id not in existing_ids:
            return new_id


def find_key_by_value(element: Any, dictionary: Dict[Any, List[Any]]) -> Optional[Any]:
    """
    Find the key in a dictionary for a given element in its values.
    Args:
        element: The element to search for in the dictionary values.
        dictionary (Dict[Any, List[Any]]): The dictionary to search in.
    Returns:
        Optional[Any]: The key if the element is found, None otherwise.
    """
    return next((key for key, values in dictionary.items() if element in values), None)


def decode_unicode(data: Any) -> Any:

    if isinstance(data, str):
        return data.encode('utf-8').decode('unicode_escape')
    elif isinstance(data, list):
        return [decode_unicode(item) for item in data]
    elif isinstance(data, dict):
        return {key: decode_unicode(value) for key, value in data.items()}
    return data


def load_or_create_dict(path: str) -> Dict[Any, Any]:

    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    if not os.path.exists(path):
        with open(path, 'w') as file:
            json.dump({}, file)
        return {}
    
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {path}")
        return {}


def load_json(file_path: str) -> Any:

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {file_path}")
        raise

def save_json(data: Any, filepath: str) -> None:

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    except IOError as e:
        logging.error(f"Error saving JSON file: {e}")
        raise


def load_contexts(file_path):
    """
    Load contexts from a SQuAD-like JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        list: A list of unique contexts.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    contexts = []
    original_contexts = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            original_contexts.append(context)
            if context not in contexts:
                contexts.append(context)
    
    print("Original number of contexts:", len(original_contexts))
    print("Number of unique contexts returned:", len(contexts))
    return contexts


def get_version_from_dataset_name(dataset_name):

    if dataset_name.endswith(".json"):
        dataset_name = dataset_name.replace(".json", "")   
    version = "v" + re.search(r'[vspcf](\d)$', dataset_name).group(1)
    print(f"Version deduced: {version}")
    return version

def recover_leading_spaces(text, chars, start=0, current="", results=None):
    """
    Recursively insert spaces before specified characters in a string.

    Args:
        text (str): The original text to process.
        chars (str): Characters before which spaces should be inserted.
        start (int): The current index in the text being processed.
        current (str): The current state of the processed text.
        results (list): A list to collect the combinations.

    Returns:
        list: A list of combinations with spaces inserted before specified characters.
    """
    if results is None:
        results = [text]
    
    if start >= len(text) or len(results) >= 10:
        if len(results) < 10:
            results.append(current)
        return results
    
    if text[start] in chars:
        if len(results) < 10:
            if start > 0:
                recover_leading_spaces(text, chars, start + 1, current + " " + text[start], results)
            if len(results) < 10:
                recover_leading_spaces(text, chars, start + 1, current + text[start], results)
    else:
        recover_leading_spaces(text, chars, start + 1, current + text[start], results)
    
    return results


def remove_leading_and_trailing_spaces(text, lead_remove, trail_remove):
    """
    Remove leading and trailing spaces around specified characters.

    Args:
        text (str): The original text to process.
        lead_remove (str): Characters to remove leading spaces from.
        trail_remove (str): Characters to remove trailing spaces from.

    Returns:
        list: A list of variations of the text with spaces removed.
    """
    texts = []

    for lead_char in list(lead_remove) + ['']:
        text = re.sub(f"\s{lead_char}", f"{lead_char}", text)
        for trail_char in list(trail_remove) + ['']:
            text = re.sub(f"{trail_char}\s", f"{trail_char}", text)
            if text not in texts:
                texts.append(text)
    
    return texts