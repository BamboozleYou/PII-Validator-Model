import argparse
import pandas as pd
from pii_validator import TargetedPIIValidator, ModelConfig

def normalize_columns(df):
    mapping = {}
    for c in df.columns:
        key = c.strip().lower().replace("_", "").replace(" ", "")
        mapping[key] = c
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="none")
    ap.add_argument("--model", default="")
    ap.add_argument("--base_url", default="http://localhost:11434")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="validation_results.csv")
    args = ap.parse_args()

    if args.input.lower().endswith(".xlsx"):
        df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input)

    mapping = normalize_columns(df)
    if "columnname" in mapping:
        col_col = mapping["columnname"]
    elif "column_name" in mapping:
        col_col = mapping["column_name"]
    else:
        raise ValueError("CSV must contain 'Column Name' column")

    if "guessedclassification" in mapping:
        guessed_col = mapping["guessedclassification"]
    elif "guessed_pii" in mapping:
        guessed_col = mapping["guessed_pii"]
    else:
        raise ValueError("CSV must contain 'Guessed Classification' column")

    label_space = sorted(df[guessed_col].dropna().unique().tolist())

    cfg = ModelConfig(provider=args.provider, model=args.model,
                      base_url=args.base_url, label_space=label_space)
    validator = TargetedPIIValidator(cfg)

    results = []
    for _, row in df.iterrows():
        colname = str(row[col_col])
        guessed = str(row[guessed_col])
        verdict = validator.validate_row(colname, guessed)
        results.append({
            **row.to_dict(),
            "SLM_Guess": verdict.replace("Negative: ", "")
                              if verdict.startswith("Negative") else (guessed if verdict=="True positive" else ""),
            "Validation_Result": verdict
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote: {args.output}")
    print(out_df["Validation_Result"].value_counts())

if __name__ == "__main__":
    main()

