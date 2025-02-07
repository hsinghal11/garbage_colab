import json
from sklearn.metrics import cohen_kappa_score

def flatten_annotations(annotations):
    """
    Flatten a list of annotations into a single list of labels.
    Handles multi-dimensional annotations by extracting only the label.
    """
    flatten_data = []
    for sentence in annotations:
        for label in sentence:
            if isinstance(label, dict) and "labels" in label:
                flatten_data.append(label["labels"])
            else:
                flatten_data.append(label)
    return flatten_data

def load_annotations(file_path):
    """
    Load annotations from a JSON file.
    Each entry should have a "pos_tags" key containing text and labels.
    Returns a structured list of sentences with their labels.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    result = []
    for entry in data:
        temp = []
        if "pos_tags" in entry:
            for val in entry["pos_tags"]:
                temp.append({"text": val["text"], "labels": val["labels"]})
        result.append(temp)
    return result

def calculate_cohen_kappa(file1, file2):
    """
    Calculate Cohen's kappa score between two annotation files.
    """
    # Load and flatten annotations
    annotations_a1 = flatten_annotations(load_annotations(file1))
    annotations_a2 = flatten_annotations(load_annotations(file2))

    # Ensure annotations have the same length
    if len(annotations_a1) != len(annotations_a2):
        raise ValueError("The annotations have different lengths.")

    # Check for empty annotations
    if not annotations_a1 or not annotations_a2:
        raise ValueError("One or both annotation files are empty.")

    # Calculate Cohen's kappa
    kappa_score = cohen_kappa_score(annotations_a1, annotations_a2)
    return kappa_score

# Example usage
file1 = "pos_1.json"
file2 = "pos_2.json"

try:
    kappa = calculate_cohen_kappa(file1, file2)
    print(f"Cohen's kappa: {kappa}")
except ValueError as e:
    print(f"Error: {e}")

if kappa < 0:
    interpretation = "Poor agreement"
elif kappa <= 0.20:
    interpretation = "Slight agreement"
elif kappa <= 0.40:
    interpretation = "Fair agreement"
elif kappa <= 0.60:
    interpretation = "Moderate agreement"
elif kappa <= 0.80:
    interpretation = "Substantial agreement"
else:
    interpretation = "Almost perfect agreement"
print(f"Cohen's Kappa: {kappa}, Interpretation: {interpretation}")
