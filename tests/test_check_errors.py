import pytest
import pandas as pd
from scoring import load_goldstandard, load_predictions, check_errors

def test_load_goldstandard(tmp_path):
    """
    Test that the load_goldstandard function correctly loads data from a CSV file
    and returns a DataFrame with the expected structure and values.
    """
    # Create a sample CSV file
    gold_file = tmp_path / "goldstandard.csv"
    gold_file.write_text("id,label\n1,1\n2,0\n3,1\n")

    df = load_goldstandard(str(gold_file))

    assert isinstance(df, pd.DataFrame)
    assert df.loc[1, 'label'] == 1
    assert df.loc[2, 'label'] == 0
    assert df.loc[3, 'label'] == 1


def test_load_predictions(tmp_path):
    """
    Test that the load_predictions function correctly loads prediction data from a CSV file
    and returns a dictionary mapping IDs to predicted labels.
    """
    # Create a sample CSV file for predictions
    pred_file = tmp_path / "predictions.csv"
    pred_file.write_text("id,predicted_label\n1,1\n2,0\n3,0\n")

    predictions = load_predictions(str(pred_file))

    assert isinstance(predictions, dict)
    assert predictions[1] == 1
    assert predictions[2] == 0
    assert predictions[3] == 0


def test_load_predictions_with_duplicates(tmp_path):
    """
    Test that the load_predictions function correctly identifies and reports duplicate IDs
    in the predictions file, while keeping the last occurrence of each duplicate in the result.
    """
    # Create a sample CSV file with duplicate IDs
    pred_file = tmp_path / "predictions_with_duplicates.csv"
    pred_file.write_text("id,predicted_label\n1,1\n2,0\n3,0\n3,1\n")  # Note duplicate ID 3

    # Capture stdout
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call function while stdout is being captured
    predictions = load_predictions(str(pred_file))

    # Restore stdout
    sys.stdout = sys.__stdout__

    # Check that the duplicate was detected
    assert "Duplicate prediction entries for IDs: 3" in captured_output.getvalue()

    # Check dictionary has the last value for the duplicate key
    assert predictions[3] == 1


def test_check_errors():
    """
    Test the check_errors function with valid predictions that match the gold standard.
    Verifies that no error messages are reported when all predictions are valid.
    """
    gold_df = pd.DataFrame({
        'id': ['de_0', 'en_0', 'fr_0'],
        'label': [0, 1, 0]
    })
    gold_df = gold_df.set_index('id')
    predictions = pd.DataFrame({
        'id': ['de_0', 'en_0', 'fr_0'],
        'predicted_label': [0, 1, 0]
    })
    predictions = predictions.set_index('id')['predicted_label'].to_dict()
    errors = []

    def mock_print(msg):  # Capture print statements
        errors.append(msg)

    # Monkey-patch print to capture output
    import builtins
    original_print = builtins.print
    builtins.print = mock_print

    check_errors(gold_df, predictions)

    builtins.print = original_print  # Restore original print

    assert "No errors found in predictions." in errors[0]  # Expect no error message


def test_check_errors_MissingPredictions():
    """
    Test that check_errors correctly identifies and reports when predictions
    are missing for some IDs that are present in the gold standard.
    """
    #Test for missing predictions; i.e., when the number of predictions is less than the number of goldstandard labels
    gold_df = pd.DataFrame({
        'id': ['de_0', 'en_0', 'fr_0'],
        'label': [0, 1, 0]
    })
    gold_df = gold_df.set_index('id')
    predictions = pd.DataFrame({
        'id': ['de_0', 'en_0', ],
        'predicted_label': [0, 1]
    })
    predictions = predictions.set_index('id')['predicted_label'].to_dict()
    errors = []

    def mock_print(msg):  # Capture print statements
        errors.append(msg)

    # Monkey-patch print to capture output
    import builtins
    original_print = builtins.print
    builtins.print = mock_print

    check_errors(gold_df, predictions)

    builtins.print = original_print  # Restore original print

    assert "Missing predictions for IDs:" in errors[1]  # Expect an error message


def test_check_errors_UnknownLabels():
    """
    Test that check_errors correctly identifies and reports when predictions
    contain label values that are not valid according to the gold standard.
    """
    gold_df = pd.DataFrame({
        'id': ['de_0', 'en_0', 'fr_0'],
        'label': [0, 1, 0]
    })
    gold_df = gold_df.set_index('id')
    predictions = pd.DataFrame({
        'id': ['de_0', 'en_0', 'fr_0'],
        'predicted_label': [0, 1, 2]
    })
    predictions = predictions.set_index('id')['predicted_label'].to_dict()
    errors = []

    def mock_print(msg):  # Capture print statements
        errors.append(msg)

    # Monkey-patch print to capture output
    import builtins
    original_print = builtins.print
    builtins.print = mock_print

    check_errors(gold_df, predictions)

    builtins.print = original_print  # Restore original print

    assert "Unknown labels found in predictions:" in errors[1]  # Expect error


def test_check_errors_UnknownKeys():
    """
    Test that check_errors correctly identifies and reports when predictions
    contain IDs that do not exist in the gold standard dataset.
    """
    gold_df = pd.DataFrame({
        'id': ['de_0', 'en_0', 'fr_0'],
        'label': [0, 1, 0]
    })
    gold_df = gold_df.set_index('id')
    predictions = pd.DataFrame({
        'id': ['de_0', 'en_0', 'fr_0', 'en_1'],
        'predicted_label': [0, 1, 0, 1]
    })
    predictions = predictions.set_index('id')['predicted_label'].to_dict()
    errors = []

    def mock_print(msg):  # Capture print statements
        errors.append(msg)

    # Monkey-patch print to capture output
    import builtins
    original_print = builtins.print
    builtins.print = mock_print

    check_errors(gold_df, predictions)

    builtins.print = original_print  # Restore original print

    assert "Unknown keys found in predictions:" in errors[1]  # Expect error