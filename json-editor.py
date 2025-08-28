import sys
import os
import json
import pandas as pd


def parse_structured_output(row):
    try:
        data = json.loads(row["structured_output"])
        flat = {
            "PATNR": row["PATNR"],
            "age": data.get("age"),
            "sex": data.get("sex"),
            "epilepsy_diagnosis_present": data.get("epilepsy_diagnosis_present"),
            "is_focal": data.get("is_focal"),
            "seizure_frequency": data.get("seizure_frequency"),
            "duration_epilepsy": data.get("duration_epilepsy"),
            "ever_status_epilepsy": data.get("ever_status_epilepsy"),
            "location_epilepsy": data.get("location_epilepsy"),
            "hippocampal_sclerosis_present": data.get("hippocampal_sclerosis_present"),
            "focal_cortical_dysplasia_present": data.get("focal_cortical_dysplasia_present"),
            "report_date": data.get("report_date"),
            "refractory_epilepsy": data.get("refractory_epilepsy"),
            "seizure_free": data.get("seizure_free"),
            "last_seizure_date": data.get("last_seizure_date")
        }

        # Medical History
        med_hist = data.get("medical_history", {})
        flat["febrile_seizures"] = med_hist.get("febrile_seizures")
        flat["ischemic_stroke"] = med_hist.get("ischemic_stroke")
        flat["hemorraghic_stroke"] = med_hist.get("hemorraghic_stroke")
        flat["traumatic_brain_injury"] = med_hist.get("traumatic_brain_injury")
        flat["neuroinfection"] = med_hist.get("neuroinfection")
        flat["psychiatric_disorder"] = med_hist.get("psychiatric_disorder")
        flat["heart_failure"] = med_hist.get("heart_failure")
        flat["diabetes"] = med_hist.get("diabetes")

        # Imaging + EEG
        imaging = data.get("imaging_eeg", {})
        flat["mri_abnormal"] = imaging.get("mri_abnormal")
        flat["mri_findings_summary"] = imaging.get("mri_findings_summary")
        flat["interictal_spikes_present"] = imaging.get("interictal_spikes_present")
        flat["ictal_pattern"] = imaging.get("ictal_pattern")
        flat["eeg_lateralization"] = imaging.get("eeg_lateralization")

        # Epilepsy Surgery
        surgery = data.get("epilepsy_surgery", {})
        flat["epilepsy_surgery_done"] = surgery.get("epilepsy_surgery_done")
        flat["surgery_type"] = surgery.get("surgery_type")
        flat["surgery_outcome"] = surgery.get("surgery_outcome")

        # Social impact
        social = data.get("social_impact", {})
        flat["driving_status"] = social.get("driving_status")
        flat["working_status"] = social.get("working_status")
        flat["quality_of_life_comments"] = social.get("quality_of_life_comments")

        # Medications
        for med in data.get("medications", []):
            name = med.get("name", "").strip().lower().replace(" ", "_")
            if name:
                flat[f"current_{name}"] = True
                flat[f"dose_current_{name}"] = med.get("dose")
                flat[f"unit_current_{name}"] = med.get("dose_unit")

        for med in data.get("previous_medications", []):
            name = med.get("name", "").strip().lower().replace(" ", "_")
            if name:
                flat[f"previous_{name}"] = True
                flat[f"dose_previous_{name}"] = med.get("dose")
                flat[f"unit_previous_{name}"] = med.get("dose_unit")
                flat[f"reason_previous_{name}"] = med.get("reason_stopped")

        return flat

    except Exception as e:
        return {"PATNR": row["PATNR"], "parse_error": str(e)}


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_structured_output.py yourfile_structured.csv")
        return

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    if "PATNR" not in df.columns or "structured_output" not in df.columns:
        print("Input file must contain 'PATNR' and 'structured_output' columns")
        return

    parsed = [parse_structured_output(row) for _, row in df.iterrows()]
    parsed_df = pd.DataFrame(parsed).fillna("")

    output_path = os.path.splitext(input_path)[0] + "_parsed.csv"
    parsed_df.to_csv(output_path, index=False)

    print(f"✅ Parsed output saved to {output_path}")


if __name__ == "__main__":
    main()