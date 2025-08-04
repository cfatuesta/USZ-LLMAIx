# process_csv.py
import json
import os
import sys

import pandas as pd
import ollama
from pydantic import BaseModel, Field
import datetime as datetime

from typing import List, Optional
from datetime import date


PROMPT_FILES = {
    "Epilepsy": "prompts/epilepsy_prompt.txt",
    "Stroke": "prompts/stroke_prompt.txt",
    "MS": "prompts/ms_prompt.txt"
}


class EpilepsyDiagnosis(BaseModel):
    is_focal: bool = Field(description="Indicates if the epilepsy is focal.")
    seizure_frequency: Optional[float] = Field(description="Seizure frequency, in times per month (e.g., 1.5).")
    duration_epilepsy: Optional[int] = Field(description="Duration of epilepsy in years.")
    ever_status_epilepsy: bool = Field(description="Whether the patient has ever had a status epilepticus.")
    location_epilepsy: Optional[str] = Field(description="Anatomical seizure origin (e.g., temporal, frontal, multilobar).")
    hippocampal_sclerosis_present: bool = Field(description="Whether hippocampal sclerosis is present.")
    focal_cortical_dysplasia_present: bool = Field(description="Whether focal cortical dysplasia is present.")
    refractory_epilepsy: Optional[bool] = Field(description="Whether epilepsy is drug-resistant (failed at least 2 ASMs).")
    seizure_free: Optional[bool] = Field(description="Whether the patient is currently seizure-free.")
    last_seizure_date: Optional[date] = Field(description="Date of last seizure (YYYY-MM-DD), if known.")

class Medication(BaseModel):
    name: str = Field(description="Generic name of the medication.")
    dose: Optional[float] = Field(default=None, description="Dose, if available.")
    dose_unit: Optional[str] = Field(default=None, description="Dose unit (e.g., mg, ml), if available.")

class PreviousMedication(Medication):
    reason_stopped: Optional[str] = Field(default=None, description="Reason for discontinuation, if known.")

class ImagingAndEEG(BaseModel):
    mri_abnormal: Optional[bool] = Field(description="Whether the MRI was reported as abnormal.")
    mri_findings_summary: Optional[str] = Field(description="Summary of abnormal MRI findings (e.g., MTS, lesions).")
    interictal_spikes_present: Optional[bool] = Field(description="Presence of interictal epileptiform discharges on EEG.")
    ictal_pattern: Optional[bool] = Field(description="Whether an ictal EEG pattern was captured.")
    eeg_lateralization: Optional[str] = Field(description="EEG lateralization (e.g., left-sided, right-sided, generalized).")

class EpilepsySurgery(BaseModel):
    epilepsy_surgery_done: Optional[bool] = Field(description="Whether the patient underwent epilepsy surgery.")
    surgery_type: Optional[str] = Field(description="Type of surgery (e.g., ATL, laser ablation).")
    surgery_outcome: Optional[str] = Field(description="Reported outcome (e.g., Engel class, percent seizure reduction).")

class SocialImpact(BaseModel):
    driving_status: Optional[str] = Field(description="Driving clearance status (e.g., allowed, restricted).")
    working_status: Optional[str] = Field(description="Employment or education status (e.g., working, studying, disability).")
    quality_of_life_comments: Optional[str] = Field(description="Summary of any quality-of-life issues mentioned.")

class MedicalHistory(BaseModel):
    febrile_seizures: bool = Field(description="History of febrile seizures.")
    ischemic_stroke: bool = Field(description="History of ischemic stroke.")
    hemorraghic_stroke: bool = Field(description="History of hemorrhagic stroke.")
    traumatic_brain_injury: bool = Field(description="History of TBI.")
    neuroinfection: bool = Field(description="History of neuroinfection (e.g., meningitis, encephalitis).")
    psychiatric_disorder: bool = Field(description="Presence of psychiatric comorbidity (e.g., depression, anxiety).")
    heart_failure: bool = Field(description="History of heart failure.")
    diabetes: bool = Field(description="History of diabetes mellitus.")

class PatientEpilepsyReport(BaseModel):
    patient_id: Optional[str] = Field(description="Patient unique identifier, if available.")
    age: Optional[int] = Field(description="Patient age in years, ideally from the earliest report.")
    sex: Optional[str] = Field(description="Sex of the patient (e.g., male, female).")
    epilepsy_diagnosis_present: bool = Field(description="Whether the patient has a diagnosis of epilepsy or seizures.")
    
    # Dates from earliest/latest reports
    earliest_report_date: Optional[date] = Field(description="Date of earliest report available (YYYY-MM-DD).")
    latest_report_date: Optional[date] = Field(description="Date of most recent report available (YYYY-MM-DD).")

    # Medications
    medications: List[Medication] = Field(default_factory=list, description="Current ASMs from latest relevant report.")
    previous_medications: List[PreviousMedication] = Field(default_factory=list, description="All previous ASMs and stop reasons, up to the current medication date.")

    # Flattened epilepsy diagnosis fields
    is_focal: bool
    seizure_frequency: Optional[float]
    duration_epilepsy: Optional[int]
    ever_status_epilepsy: bool
    location_epilepsy: Optional[str]
    hippocampal_sclerosis_present: bool
    focal_cortical_dysplasia_present: bool
    refractory_epilepsy: Optional[bool]
    seizure_free: Optional[bool]
    last_seizure_date: Optional[date]

    # Grouped components
    medical_history: MedicalHistory
    imaging_eeg: Optional[ImagingAndEEG]
    epilepsy_surgery: Optional[EpilepsySurgery]
    social_impact: Optional[SocialImpact]


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
    system_prompt = prompt_template.replace("{report}", "")
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": report_text}
            ],
            format="json"
        )
        data = json.loads(response["message"]["content"])
        report = PatientEpilepsyReport.model_validate(data)
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