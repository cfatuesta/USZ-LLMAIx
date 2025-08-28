import os, sys, json
import pandas as pd
from typing import Optional, List
from pydantic import BaseModel, ValidationError
import ollama

CHUNK_SIZE = 4000
PROMPT_PATH = "prompts/epilepsy_prompt_chunked.txt"

class Medication(BaseModel):
    name: str
    dose: Optional[float] = None
    dose_unit: Optional[str] = None

class PreviousMedication(Medication):
    reason_stopped: Optional[str] = None

class MedicalHistory(BaseModel):
    febrile_seizures: str
    ischemic_stroke: str
    hemorraghic_stroke: str
    traumatic_brain_injury: str
    neuroinfection: str
    psychiatric_disorder: str
    heart_failure: str
    diabetes: str

class ImagingAndEEG(BaseModel):
    mri_abnormal: str
    mri_findings_summary: str
    interictal_spikes_present: str
    ictal_pattern: str
    eeg_lateralization: str

class EpilepsySurgery(BaseModel):
    epilepsy_surgery_done: str
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
    epilepsy_diagnosis_present: str
    earliest_report_date: str
    latest_report_date: str
    is_focal: str
    seizure_frequency: Optional[float]
    duration_epilepsy: Optional[int]
    ever_status_epilepsy: str
    location_epilepsy: str
    hippocampal_sclerosis_present: str
    focal_cortical_dysplasia_present: str
    refractory_epilepsy: str
    seizure_free: str
    last_seizure_date: str
    medications: List[Medication]
    previous_medications: List[PreviousMedication]
    medical_history: MedicalHistory
    imaging_eeg: ImagingAndEEG
    epilepsy_surgery: EpilepsySurgery
    social_impact: SocialImpact

def load_prompt(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def chunk_text(text: str, chunk_size=CHUNK_SIZE) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n\n"
        else:
            chunks.append(current)
            current = para + "\n\n"
    if current:
        chunks.append(current)
    return chunks

def query_model(report_chunk: str, prompt: str) -> Optional[dict]:
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": report_chunk}
            ],
            format="json"
        )
        content = response["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def sanitize_output(data: dict):
    def to_str(x):
        if isinstance(x, bool):
            return "yes" if x else "no"
        if x is None:
            return ""
        return str(x)

    def to_float(x):
        try:
            return float(str(x).split()[0])
        except Exception:
            return None

    # Convert all top-level fields
    for key in data:
        if isinstance(data[key], bool) or data[key] is None:
            data[key] = to_str(data[key])

    # Handle nested dicts
    for section in ["medical_history", "imaging_eeg", "epilepsy_surgery", "social_impact"]:
        if section in data:
            for k, v in data[section].items():
                data[section][k] = to_str(v)

    # Fix medication doses
    for meds_key in ["medications", "previous_medications"]:
        if meds_key in data:
            for med in data[meds_key]:
                if "dose" in med:
                    med["dose"] = to_float(med["dose"])

    return data

def process_patient(patnr, text, prompt):
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        print(f"  → Sending chunk {i+1}/{len(chunks)}...")
        result = query_model(chunk, prompt)
        if result:
            try:
                result = sanitize_output(result)  # sanitize before validation
                parsed = PatientEpilepsyReport(**result)
                print(f"  ✓ Parsed successfully.")
                return patnr, parsed.model_dump_json()
            except ValidationError as ve:
                print(f"  ✗ Validation error: {ve}")
    return patnr, "[ERROR]"

def main():
    if len(sys.argv) < 2:
        print("Usage: python process_csv_chunked.py your_file.csv")
        return
    file_path = sys.argv[1]
    df = pd.read_csv(file_path, encoding="latin1")
    if "PATNR" not in df.columns or "Beurteilung" not in df.columns:
        print("Missing required columns: PATNR or Beurteilung")
        return
    prompt = load_prompt(PROMPT_PATH)
    outputs = []
    for patnr, group in df.groupby("PATNR"):
        print(f"Processing patient {patnr}...")
        all_text = "\n\n".join(group["Beurteilung"].dropna().astype(str))
        if not all_text:
            outputs.append({"PATNR": patnr, "structured_output": ""})
            continue
        patnr, structured = process_patient(patnr, all_text, prompt)
        outputs.append({"PATNR": patnr, "structured_output": structured})
    out_path = os.path.splitext(file_path)[0] + "_structured_chunked.csv"
    pd.DataFrame(outputs).to_csv(out_path, index=False)
    print(f"\n✅ Done. Output saved to {out_path}")

if __name__ == "__main__":
    main()