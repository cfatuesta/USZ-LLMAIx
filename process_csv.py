import json
import os
import sys
import pandas as pd
import ollama
from datetime import date
from pydantic import BaseModel, Field
from typing import List, Optional

PROMPT_FILES = {
    "Epilepsy": "prompts/epilepsy_prompt.txt"
}


class Medication(BaseModel):
    name: str
    dose: Optional[float] = None
    dose_unit: str = ""


class PreviousMedication(Medication):
    reason_stopped: str = ""


class MedicalHistory(BaseModel):
    febrile_seizures: bool
    ischemic_stroke: bool
    hemorraghic_stroke: bool
    traumatic_brain_injury: bool
    neuroinfection: bool
    psychiatric_disorder: bool
    heart_failure: bool
    diabetes: bool


class ImagingEEG(BaseModel):
    mri_abnormal: bool
    mri_findings_summary: str
    interictal_spikes_present: bool
    ictal_pattern: bool
    eeg_lateralization: str


class EpilepsySurgery(BaseModel):
    epilepsy_surgery_done: bool
    surgery_type: str
    surgery_outcome: str


class SocialImpact(BaseModel):
    driving_status: str
    working_status: str
    quality_of_life_comments: str


class PatientEpilepsyReport(BaseModel):
    patient_id: str
    age: Optional[int]
    sex: str
    epilepsy_diagnosis_present: bool
    earliest_report_date: str
    latest_report_date: str
    medications: List[Medication]
    previous_medications: List[PreviousMedication]
    is_focal: bool
    seizure_frequency: Optional[float]
    duration_epilepsy: Optional[int]
    ever_status_epilepsy: bool
    location_epilepsy: str
    hippocampal_sclerosis_present: bool
    focal_cortical_dysplasia_present: bool
    refractory_epilepsy: bool
    seizure_free: bool
    last_seizure_date: str
    medical_history: MedicalHistory
    imaging_eeg: ImagingEEG
    epilepsy_surgery: EpilepsySurgery
    social_impact: SocialImpact


def load_prompt(filepath):
    with open(filepath, "r") as f:
        return f.read()


def query_llama(report_text, prompt_template):
    system_prompt = prompt_template.replace("{report}", "")
    try:
        response = ollama.chat(
            model="que",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": report_text}
            ],
            options={"temperature": 0}
        )
        content = response["message"]["content"]
        content = content.strip()

        # Try to extract JSON even if some text is around
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            json_str = content[start:end]
            data = json.loads(json_str)
        except Exception as e:
            print("❌ JSON parsing failed")
            return json.dumps({"error": f"Failed to parse JSON: {str(e)}", "raw": content})

        try:
            validated = PatientEpilepsyReport.model_validate(data)
            flat_data = validated.model_dump()

            # Print each top-level variable
            print("🔍 Extracted variables:")
            for key, value in flat_data.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        print(f"  {key}.{sub_key}: {sub_value}")
                elif isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")

            return json.dumps(flat_data, ensure_ascii=False)

        except Exception as e:
            print("❌ Validation failed")
            return json.dumps({"error": f"Validation error: {str(e)}", "raw": json_str})

    except Exception as e:
        print("❌ Ollama call failed")

        return json.dumps({"error": f"Ollama call failed: {str(e)}"})


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

    prompt_text = load_prompt(PROMPT_FILES["Epilepsy"])
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
    output_file = os.path.splitext(file_path)[0] + "_structured_gpt.csv"
    out_df.to_csv(output_file, index=False)
    print(f"\n✅ Done. Output saved to {output_file}")


if __name__ == "__main__":
    main()