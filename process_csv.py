# process_csv.py
import json
import os
import sys

import pandas as pd
import ollama
from pydantic import BaseModel, Field


PROMPT_FILES = {
    "Epilepsy": "prompts/epilepsy_prompt.txt",
    "Stroke": "prompts/stroke_prompt.txt",
    "MS": "prompts/ms_prompt.txt"
}

class EpilepsyDiagnosis(BaseModel):
    is_focal: bool = Field(description="Indicates if the epilepsy is focal.")
    seizure_frequency: int | None = Field(description="Frequency of seizures experienced by the patient, measured in times per month.")

class Medication(BaseModel):
    name: str = Field(description="Name of the medication.")
    dose: float | None = Field(default=None, description="Dose of the medication, if known.")
    dose_unit: str | None = Field(default=None, description="Unit of the dose, if known.")

class PreviousMedication(Medication):
    reason_stopped: str | None = Field(default=None, description="Reason for stopping the medication, if known.")

class PatientEplilepsyReport(BaseModel):
    age: int | None = Field(description="Age of the patient, taken from the most recent report if there are multiple.")
    epilepsy_diagnosis_present: bool = Field(description="Indicates if the patient has a diagnosis of epilepsy.")
    medications: list[Medication] = Field(default_factory=list, description="List of medications the patient is currently taking, including their respective doses if available.")
    previous_medications: list[PreviousMedication] = Field(default_factory=list, description="List of medications the patient has previously taken, including their respective reasons for stopping, if known.")


def select_prompt():
    print("Select a disease prompt:")
    for i, key in enumerate(PROMPT_FILES.keys()):
        print(f"{i+1}. {key}")
    idx = int(input("Enter choice: ")) - 1
    return list(PROMPT_FILES.keys())[idx]

def load_prompt(filepath):
    with open(filepath, "r") as f:
        return f.read()

def query_llama(report_text, prompt_template):
    full_prompt = prompt_template.replace("{report}", report_text)
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": ""}
            ],
            format=PatientEplilepsyReport.model_json_schema(),
        )
        report = PatientEplilepsyReport.model_validate(json.loads(response["message"]["content"]))
        print(report.model_dump_json(indent=2))
        return report.model_dump_json()
    except Exception as e:
        print(f"Error querying Llama: {str(e)}")
        return f"[ERROR: {str(e)}]"

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_csv.py your_file.csv")
        return

    file_path = sys.argv[1]
    sep = "\t" if file_path.endswith(".tsv") else ","

    try:
        df = pd.read_csv(file_path, sep=sep, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, sep=sep, encoding="latin1")

    if "PATNR" not in df.columns or "Beurteilung" not in df.columns:
        print("Missing required columns 'PATNR' or 'Beurteilung'")
        return

    prompt_name = select_prompt()
    prompt_text = load_prompt(PROMPT_FILES[prompt_name])

    grouped = df.groupby("PATNR")
    outputs = []

    for patnr, group in grouped:
        print(f"Processing patient {patnr}...")
        all_texts = group["Beurteilung"].dropna().astype(str).tolist()
        if not all_texts:
            outputs.append({"PATNR": patnr, "structured_output": ""})
            continue

        combined_text = "\n\n".join(all_texts)
        response = query_llama(combined_text, prompt_text)
        outputs.append({"PATNR": patnr, "structured_output": response})

    out_df = pd.DataFrame(outputs)
    output_file = os.path.splitext(file_path)[0] + "_structured.csv"
    out_df.to_csv(output_file, index=False)
    print(f"\n✅ Done. Output saved to {output_file}")

if __name__ == "__main__":
    main()