import pandas as pd
import argparse
from sklearn.metrics import precision_recall_fscore_support
import os
import sys

def load_goldstandard(file_path):
    """Load the gold standard data from a CSV file and return a data frame."""
    if os.path.isdir(file_path):
        files = os.listdir(file_path)
        if len(files) == 1:
            file_path = os.path.join(file_path, files[0])
        else:
            sys.stderr.write("Cannot load path with several files\n" + str(files))
            sys.exit(1)

    df = pd.read_csv(file_path)
    return df.set_index('id')


def load_predictions(file_path):
    """Load the predictions from a CSV file and return a data frame with ID and predictions columns."""
    if os.path.isdir(file_path):
        files = os.listdir(file_path)
        if len(files) == 1:
            file_path = os.path.join(file_path, files[0])
        else:
            sys.stderr.write("Cannot load path with several files\n" + str(files))
            sys.exit(1)

    df = pd.read_csv(file_path, header=0)

    # Check for duplicates in the 'id' column before setting it as the index
    duplicate_ids = df['id'][df['id'].duplicated()].unique()
    if len(duplicate_ids) > 0:
        print(f"Duplicate prediction entries for IDs: {', '.join(map(str, duplicate_ids))}")

    return df.set_index('id')['predicted_label'].to_dict()

def check_errors(goldstandard, predictions):
    """Check for missing predictions, duplicate entries, and unknown labels in the predictions."""
    errors = []
    goldstandard = goldstandard['label'].to_dict()
    missing_ids = set(goldstandard.keys()) - set(predictions.keys())
    if missing_ids:
        errors.append(f"Missing predictions for IDs: {missing_ids}")

    duplicate_ids = [id for id in predictions if list(predictions.keys()).count(id) > 1]
    if duplicate_ids:
        errors.append(f"Duplicate prediction entries for IDs: {set(duplicate_ids)}")

    unknown_labels = [label for label in predictions.values() if label not in {0, 1}]
    if unknown_labels:
        errors.append(f"Unknown labels found in predictions: {set(unknown_labels)}")

    unknown_keys = [key for key in predictions.keys() if key not in goldstandard.keys()]
    if unknown_keys:
        errors.append(f"Unknown keys found in predictions: {set(unknown_keys)}")

    if errors:
        print("Errors found in predictions:")
        for error in errors:
            print(error)
    else:
        print("No errors found in predictions.")


def evaluate(gold_df, predictions, output_folder=None):
    """Calculate and print or save precision, recall, and F1-score based on the gold standard and predictions."""
    goldstandard = gold_df['label'].to_dict()
    languages = gold_df['language'].to_dict()

    all_y_true = []
    all_y_pred = []
    results = []

    for lang in set(languages.values()):
        lang_ids = [i for i in goldstandard.keys() if languages[i] == lang]
        y_true = [goldstandard[i] for i in lang_ids if i in predictions]
        y_pred = [predictions[i] for i in lang_ids if i in predictions]

        if y_true and y_pred:
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            results.append(f'Precision-{lang}: {precision:.4f}\nRecall-{lang}: {recall:.4f}\nF1-{lang}: {f1:.4f}\n')

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    if all_y_true and all_y_pred:
        precision, recall, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='binary')
        results.append(f'Precision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}')

    result_text = "\n".join(results)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, "scores.txt"), "w") as f:
            f.write(result_text)

    print(result_text)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate predictions against gold standard.")
    parser.add_argument("--goldstandard_file", type=str, help="Path to the goldstandard CSV file")
    parser.add_argument("--predictions_file", type=str, help="Path to the predictions CSV file")
    parser.add_argument("--output_folder", type=str, default=None, help="Optional folder to save evaluation results")
    args = parser.parse_args()

    gold_df = load_goldstandard(args.goldstandard_file)
    predictions = load_predictions(args.predictions_file)

    check_errors(gold_df, predictions)
    evaluate(gold_df, predictions, args.output_folder)