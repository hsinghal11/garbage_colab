import json
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

def load_annotations(file_path):
    """
    Load annotations from a JSON file.
    Returns a dictionary mapping image IDs to labels.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    annotations = {}
    for entry in data:
        annotations[entry["id"]] = entry["label"]
    return annotations

def build_rating_matrix(annotators_data, labels):
    """
    Build the rating matrix for Fleiss' Kappa.
    Each row corresponds to an image, and each column to a label.
    The value at (i, j) is the count of annotators who chose label j for image i.
    """
    # Find all unique image IDs
    all_image_ids = set()
    for annotations in annotators_data:
        all_image_ids.update(annotations.keys())

    # Initialize the matrix
    label_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(all_image_ids), len(labels)), dtype=int)

    # Fill the matrix
    for row_idx, image_id in enumerate(sorted(all_image_ids)):
        for annotations in annotators_data:
            label = annotations.get(image_id)
            if label in label_index:
                col_idx = label_index[label]
                matrix[row_idx][col_idx] += 1

    return matrix

def main():
    # File paths for the annotator data
    files = ["images_1.json", "images_2.json", "images_3.json"]

    # Load annotations for each annotator
    annotators_data = [load_annotations(file) for file in files]

    # Define all possible labels
    labels = ["Trucks", "No Trucks"]

    # Build the rating matrix
    rating_matrix = build_rating_matrix(annotators_data, labels)

    # Calculate Fleiss' Kappa
    kappa = fleiss_kappa(rating_matrix)
    print("Fleiss' Kappa:", kappa)
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
    print(f"Fleiss' Kappa: {kappa}, Interpretation: {interpretation}")

if __name__ == "__main__":
    main()
