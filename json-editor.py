import json
import pandas as pd
import sys
import os
from collections import defaultdict

def flatten_json_column(json_str):
    try:
        return json.loads(json_str)
    except:
        return {}

def extract_med_columns(df, key):
    med_cols = defaultdict(dict)
    for i, row in df.iterrows():
        entry = flatten_json_column(row["structured_output"])
        for med in entry.get(key, []):
            name = med.get("name", "").strip().lower().replace(" ", "_")
            dose = med.get("dose")
            dose_unit = med.get("dose_unit", "")
            col_prefix = f"{key}_{name}"
            med_cols[i][f"{col_prefix}"] = True
            med_cols[i][f"{col_prefix}_dose"] = dose
            med_cols[i][f"{col_prefix}_dose_unit"] = dose_unit
    return pd.DataFrame.from_dict(med_cols, orient="index")

def extract_prev_med_reason(df):
    reason_cols = defaultdict(dict)
    for i, row in df.iterrows():
        entry = flatten_json_column(row["structured_output"])
        for med in entry.get("previous_medications", []):
            name = med.get("name", "").strip().lower().replace(" ", "_")
            reason = med.get("reason_stopped", "")
            col_prefix = f"previous_{name}"
            reason_cols[i][f"{col_prefix}_reason_stopped"] = reason
    return pd.DataFrame.from_dict(reason_cols, orient="index")

def extract_core_fields(df):
    flat_data = []
    for i, row in df.iterrows():
        entry = flatten_json_column(row["structured_output"])
        flat_row = {"PATNR": row["PATNR"]}

        for key, value in entry.items():
            if key in ["medications", "previous_medications"]:
                continue
            elif isinstance(value, dict):
                for subkey, subval in value.items():
                    flat_row[f"{key}_{subkey}"] = subval
            else:
                flat_row[key] = value

        flat_data.append(flat_row)
    return pd.DataFrame(flat_data)

def main():
    if len(sys.argv) < 2:
        print("Usage: python flatten_structured_csv.py structured_output.csv")
        return

    input_file = sys.argv[1]
    df = pd.read_csv(input_file)

    core_df = extract_core_fields(df)
    current_df = extract_med_columns(df, "medications")
    prev_df = extract_med_columns(df, "previous_medications")
    reason_df = extract_prev_med_reason(df)

    result = pd.concat([core_df, current_df, prev_df, reason_df], axis=1)
    output_file = os.path.splitext(input_file)[0] + "_flattened.csv"
    result.to_csv(output_file, index=False)
    print(f"✅ Flattened output saved to {output_file}")

if __name__ == "__main__":
    main()