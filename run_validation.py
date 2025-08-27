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

def load_table(path: str) -> pd.DataFrame:
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
            return pd.read_excel(path, engine="openpyxl")
        except ImportError:
            # Fallback if openpyxl isn't installed; pandas may still read with default
            return pd.read_excel(path)
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

def save_table(df: pd.DataFrame, path: str):
    """Save DataFrame back to CSV or XLSX format based on file extension"""
    ext = os.path.splitext(path.lower())[1]
    if ext == ".xlsx":
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

    # Load the input file
    df = load_table(args.input)
    mapping = normalize_columns(df)

    # Required columns (same names as before, case-insensitive)
    if "column/page" in mapping:
        col_col = mapping["column/page"]
    elif "column_name" in mapping:
        col_col = mapping["column_name"]
    else:
        raise ValueError("Input must contain a 'Column Name' or 'column/page' column")

    if "classification" in mapping:
        guessed_col = mapping["classification"]
    elif "guessed_pii" in mapping:
        guessed_col = mapping["guessed_pii"]
    else:
        raise ValueError("Input must contain a 'classification' or 'Guessed Classification' column")

    # Optional columns (exact names requested)
    dtype_col = mapping.get("datatype") or mapping.get("columndatatype")                  # 'Datatype' or 'Column Datatype'
    length_col = mapping.get("columnlength") or mapping.get("column_length")  # 'Column Length'

    # Use the fixed, hard-coded PII label set (ignore labels in file)
    cfg = ModelConfig(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        label_space=list(PII_LABELS),
        debug=args.debug,   # pass-through to enable logs
    )
    validator = TargetedPIIValidator(cfg)

    # Initialize the new columns
    confidence_types = []
    slm_guesses = []
    reasonings = []

    print(f"Processing {len(df)} rows...")
    
    for idx, row in df.iterrows():
        # Required fields
        col_val = row[col_col]
        guessed_val = row[guessed_col]
        colname = "" if pd.isna(col_val) else str(col_val)
        guessed = "" if pd.isna(guessed_val) else str(guessed_val)

        # Optional fields
        dtype_val = None
        if dtype_col is not None and dtype_col in row:
            v = row.get(dtype_col, None)
            dtype_val = None if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)

        length_val = None
        if length_col is not None and length_col in row:
            v = row.get(length_col, None)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                length_val = None
            else:
                # keep as int if possible; otherwise keep as string
                try:
                    length_val = int(v)
                except Exception:
                    try:
                        length_val = int(str(v).strip())
                    except Exception:
                        length_val = str(v)

        # Get the validation result (now returns tuple: confidence_type, slm_guess, reasoning)
        confidence_type, slm_guess, reasoning = validator.validate_row(
            column_name=colname,
            guessed=guessed,
            datatype=dtype_val,
            col_length=length_val,
        )

        confidence_types.append(confidence_type)
        slm_guesses.append(slm_guess)
        reasonings.append(reasoning)
        
        # Progress indicator
        if (idx + 1) % 5 == 0 or idx + 1 == len(df):
            print(f"  Processed {idx + 1}/{len(df)} rows")

    # Add the new columns to the DataFrame
    df['confidence type'] = confidence_types
    df['SLM guess'] = slm_guesses
    df['reasoning'] = reasonings

    # Determine output path
    output_path = args.output if args.output else args.input
    
    # Save the enhanced DataFrame
    save_table(df, output_path)
    
    print(f"\n✅ Enhanced file saved to: {output_path}")
    print(f"📊 Added columns: 'confidence type', 'SLM guess', 'reasoning'")
    
    # Show summary statistics
    try:
        print(f"\n📈 Results Summary:")
        confidence_counts = pd.Series(confidence_types).value_counts(dropna=False)
        for conf_type, count in confidence_counts.items():
            print(f"  {conf_type}: {count}")
            
        # Show accuracy if we have True positives
        if "True positive" in confidence_counts:
            total_classified = len([c for c in confidence_types if c != "Unsure"])
            accuracy = confidence_counts.get("True positive", 0) / total_classified * 100 if total_classified > 0 else 0
            print(f"  Accuracy (excluding Unsure): {accuracy:.1f}%")
        
        # Show sample reasoning for negatives
        reasoning_samples = [r for r in reasonings if r.strip()]
        if reasoning_samples:
            print(f"\n🧠 Sample Model Reasoning:")
            for i, sample in enumerate(reasoning_samples[:3]):  # Show up to 3 examples
                print(f"  {i+1}. {sample}")
            
    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    main()
