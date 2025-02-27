import pytest
import pandas as pd
from scoring import load_goldstandard, load_predictions, check_errors

def test_load_goldstandard(tmp_path):
    # Create a sample CSV file
    gold_file = tmp_path / "goldstandard.csv"
    gold_file.write_text("id,label\n1,1\n2,0\n3,1\n")

    df = load_goldstandard(str(gold_file))

    assert isinstance(df, pd.DataFrame)
    assert df.loc[1, 'label'] == 1
    assert df.loc[2, 'label'] == 0
    assert df.loc[3, 'label'] == 1


def test_load_predictions(tmp_path):
    # Create a sample CSV file for predictions
    pred_file = tmp_path / "predictions.csv"
    pred_file.write_text("id,predicted_label\n1,1\n2,0\n3,0\n")

    predictions = load_predictions(str(pred_file))

    assert isinstance(predictions, dict)
    assert predictions[1] == 1
    assert predictions[2] == 0
    assert predictions[3] == 0


def test_check_errors():
    gold_df = pd.DataFrame({"label": [1, 0, 1]}, index=[1, 2, 3])
    predictions = {1: 1, 2: 0, 3: 0}  # No missing or duplicate IDs, one wrong label

    errors = []

    def mock_print(msg):  # Capture print statements
        errors.append(msg)

    # Monkey-patch print to capture output
    import builtins
    original_print = builtins.print
    builtins.print = mock_print

    check_errors(gold_df, predictions)

    builtins.print = original_print  # Restore original print

    assert "Unknown labels found in predictions" in errors[0]  # Expect an error message

