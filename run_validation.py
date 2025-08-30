import argparse
import os
import pandas as pd
from pii_validator import TargetedPIIValidator, ModelConfig, PII_LABELS

def normalize_columns(df):
    # Case/space/underscore-insensitive column-name map
    mapping = {}
    for c in df.columns:
        key = (str(c) if c is not None else "").strip().lower().replace("_", "").replace(" ", "")
        mapping[key] = c
    return mapping

def load_table(path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    Load only CSV or XLSX, always returning a DataFrame.
    - CSV: try default, then encoding & parser fallbacks (utf-8-sig, latin-1, engine='python', sep=None sniff)
    - XLSX: use openpyxl if available; otherwise default engine
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = os.path.splitext(path.lower())[1]
    if ext == ".xlsx":
        try:
            return pd.read_excel(path, engine="openpyxl", sheet_name=sheet_name)
        except ImportError:
            # Fallback if openpyxl isn't installed; pandas may still read with default
            return pd.read_excel(path, sheet_name=sheet_name)
    elif ext == ".csv":
        # 1) vanilla
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            # 2) BOM
            try:
                return pd.read_csv(path, encoding="utf-8-sig")
            except Exception:
                pass
            # 3) latin-1
            try:
                return pd.read_csv(path, encoding="latin-1")
            except Exception as e:
                raise e
        except pd.errors.ParserError:
            # 4) let python engine sniff delimiter
            try:
                return pd.read_csv(path, engine="python", sep=None)
            except Exception:
                # 5) common fallbacks
                for enc in ("utf-8-sig", "latin-1"):
                    try:
                        return pd.read_csv(path, engine="python", sep=None, encoding=enc)
                    except Exception:
                        continue
                raise
        except Exception as e:
            # Try one more permissive attempt before giving up
            try:
                return pd.read_csv(path, engine="python", sep=None, on_bad_lines="skip")
            except Exception:
                raise e
    else:
        raise ValueError("Only .csv and .xlsx are supported. Got: " + ext)

def save_table(df: pd.DataFrame, path: str, sheet_name: str = None):
    """Save DataFrame back to CSV or XLSX format based on file extension"""
    ext = os.path.splitext(path.lower())[1]
    if ext == ".xlsx":
        # For Excel files, we need to preserve other sheets
        if sheet_name:
            # Read existing workbook to preserve other sheets
            try:
                with pd.ExcelWriter(path, mode='a', if_sheet_exists='replace', engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            except FileNotFoundError:
                # File doesn't exist, create new
                df.to_excel(path, sheet_name=sheet_name, index=False, engine="openpyxl")
        else:
            df.to_excel(path, index=False, engine="openpyxl")
    elif ext == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError("Only .csv and .xlsx are supported for output. Got: " + ext)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="none")
    ap.add_argument("--model", default="")
    ap.add_argument("--base_url", default="http://localhost:11434")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None, help="Output file path (defaults to modifying input file)")
    ap.add_argument("--debug", action="store_true", help="Enable verbose model I/O logging")
    args = ap.parse_args()

    # Load the input file from 'Sensitive Data Summary' sheet
    df = load_table(args.input, sheet_name='Sensitive Data Summary')
    
    # Find the required columns by looking for exact matches (case sensitive)
    col_col = None
    classification_col = None
    confidence_col = None
    
    for col in df.columns:
        if col == "COLUMN/PAGE":
            col_col = col
        elif col == "DATA CLASSIFICATION":
            classification_col = col
        elif col == "CONFIDENCE TYPE":
            confidence_col = col
    
    if not col_col:
        raise ValueError("Input must contain a 'COLUMN/PAGE' column")
    if not classification_col:
        raise ValueError("Input must contain a 'DATA CLASSIFICATION' column")
    if not confidence_col:
        raise ValueError("Input must contain a 'CONFIDENCE TYPE' column")

    # Filter for only 'Investigate' rows
    investigate_mask = df[confidence_col] == 'Investigate'
    investigate_rows = df[investigate_mask].copy()
    
    print(f"Found {len(investigate_rows)} rows with 'Investigate' confidence type out of {len(df)} total rows")
    
    if len(investigate_rows) == 0:
        print("No rows found with 'Investigate' confidence type. Nothing to process.")
        return

    # Use the fixed, hard-coded PII label set
    cfg = ModelConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        label_space=list(PII_LABELS),
        debug=args.debug,
    )
    validator = TargetedPIIValidator(cfg)

    # Initialize the new columns for the investigate rows
    slm_guesses = []
    slm_confidences = []
    slm_reasonings = []

    print(f"Processing {len(investigate_rows)} 'Investigate' rows...")
    
    for idx, (orig_idx, row) in enumerate(investigate_rows.iterrows()):
        # Required fields
        col_val = row[col_col]
        classification_val = row[classification_col]
        
        # Clean column name (remove brackets if present)
        colname = "" if pd.isna(col_val) else str(col_val).strip('[]')
        guessed = "" if pd.isna(classification_val) else str(classification_val)

        # Get the validation result (now returns: slm_guess, slm_confidence, slm_reasoning)
        slm_guess, slm_confidence, slm_reasoning = validator.validate_row(
            column_name=colname,
            guessed=guessed,
            datatype=None,  # Not available in this sheet
            col_length=None,  # Not available in this sheet
        )

        slm_guesses.append(slm_guess)
        slm_confidences.append(slm_confidence)
        slm_reasonings.append(slm_reasoning)
        
        # Progress indicator
        if (idx + 1) % 5 == 0 or idx + 1 == len(investigate_rows):
            print(f"  Processed {idx + 1}/{len(investigate_rows)} rows")

    # Add the new columns to the original DataFrame for investigate rows only
    # Initialize all rows with empty values first
    df['SLM Guess'] = ''
    df['SLM Confidence'] = ''
    df['SLM Reasoning'] = ''
    
    # Update only the investigate rows
    df.loc[investigate_mask, 'SLM Guess'] = slm_guesses
    df.loc[investigate_mask, 'SLM Confidence'] = slm_confidences
    df.loc[investigate_mask, 'SLM Reasoning'] = slm_reasonings

    # Determine output path
    output_path = args.output if args.output else args.input
    
    # Save the enhanced DataFrame back to the same sheet
    save_table(df, output_path, sheet_name='Sensitive Data Summary')
    
    print(f"\nEnhanced file saved to: {output_path}")
    print(f"Added columns: 'SLM Guess', 'SLM Confidence', 'SLM Reasoning'")
    
    # Show summary statistics
    try:
        print(f"\nResults Summary for 'Investigate' rows:")
        confidence_counts = pd.Series(slm_confidences).value_counts(dropna=False)
        for conf_type, count in confidence_counts.items():
            print(f"  {conf_type}: {count}")
            
        # Show accuracy if we have True positives
        if "True Positive" in confidence_counts:
            total_classified = len([c for c in slm_confidences if c != "Unsure"])
            accuracy = confidence_counts.get("True Positive", 0) / total_classified * 100 if total_classified > 0 else 0
            print(f"  Accuracy (excluding Unsure): {accuracy:.1f}%")
        
        # Show sample reasoning for negatives and unsure cases
        reasoning_samples = [r for r in slm_reasonings if r.strip()]
        if reasoning_samples:
            print(f"\nSample Model Reasoning:")
            for i, sample in enumerate(reasoning_samples[:3]):  # Show up to 3 examples
                print(f"  {i+1}. {sample}")
            
    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    main()
