import json

# Paths to the two JSON files
file1_path = "/Users/ody/Desktop/BERT/bert-pycharm/BERT_QA_from_TE_database/code/TE-CDE.json"  # File with DOI information
file2_path = "/Users/ody/Desktop/BERT/bert-pycharm/BERT_QA_from_TE_database/TE_QA_train-dev_datasets/TE-CDE+SQuADv2_mixed/train_mixed.json"  # File that needs to have DOI added
output_file2_path = file2_path.replace(".json", "_with_DOI.json")

# Load the first file and create a mapping from title to DOI.
with open(file1_path, 'r', encoding='utf-8') as f:
    data1 = json.load(f)

doi_mapping = {}
for entry in data1.get("data", []):
    title = entry.get("title")
    doi = entry.get("doi")
    if title and doi:
        doi_mapping[title] = doi

# Load the second file.
with open(file2_path, 'r', encoding='utf-8') as f:
    data2 = json.load(f)

# Iterate through each entry in the second file and add the DOI if available.
for entry in data2.get("data", []):
    title = entry.get("title")
    if title in doi_mapping:
        entry["doi"] = doi_mapping[title]
    else:
        # Optionally, handle entries with no matching DOI.
        print(f"No DOI found for title '{title}'.")

# Save the updated second file.
with open(output_file2_path, 'w', encoding='utf-8') as f:
    json.dump(data2, f, ensure_ascii=False, indent=4)

print(f"Updated file saved as '{output_file2_path}'.")
