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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="none")
    ap.add_argument("--model", default="")
    ap.add_argument("--base_url", default="http://localhost:11434")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="validation_results.csv")
    ap.add_argument("--debug", action="store_true", help="Enable verbose model I/O logging")
    args = ap.parse_args()

    df = load_table(args.input)
    mapping = normalize_columns(df)

    # Required columns (same names as before, case-insensitive)
    if "column/page" in mapping:
        col_col = mapping["column/page"]
    elif "column_name" in mapping:
        col_col = mapping["column_name"]
    else:
        raise ValueError("Input must contain a 'Column Name' column")

    if "classification" in mapping:
        guessed_col = mapping["classification"]
    elif "guessed_pii" in mapping:
        guessed_col = mapping["guessed_pii"]
    else:
        raise ValueError("Input must contain a 'Guessed Classification' column")

    # Optional columns (exact names requested)
    dtype_col = mapping.get("datatype")                  # 'Datatype'
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

    results = []
    for _, row in df.iterrows():
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

        verdict = validator.validate_row(
            column_name=colname,
            guessed=guessed,
            datatype=dtype_val,
            col_length=length_val,
        )

        if verdict.startswith("Negative: "):
            slm_guess = verdict.replace("Negative: ", "")
        elif verdict == "True positive":
            slm_guess = guessed.strip()
        else:
            slm_guess = ""

        results.append({
            **row.to_dict(),
            "SLM_Guess": slm_guess,
            "Validation_Result": verdict
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")
    try:
        print(out_df["Validation_Result"].value_counts(dropna=False))
    except Exception:
        pass

if __name__ == "__main__":
    main()
