# rules_engine.py

def compute_bmi(height_cm, weight_kg):
    """Compute BMI = weight (kg) / height (m^2)."""
    if height_cm <= 0:
        return 0.0
    h_m = height_cm / 100
    return round(weight_kg / (h_m * h_m), 2)


def evaluate_risks(age, bmi, glucose, sbp, dbp, hr=70, spo2=98):
    """Simple numeric risk scoring for conditions."""
    risks = {}

    # Type 2 Diabetes
    risks["Type 2 Diabetes Risk"] = min(1.0, (glucose - 90) / 100 + bmi / 40)

    # Hypertension
    risks["Hypertension Risk"] = min(1.0, (sbp - 120) / 60 + (dbp - 80) / 40)

    # Depression / Mood
    risks["Depression/Mood Concern"] = 0.3 if age > 40 else 0.2

    # Migraine
    risks["Migraine Risk"] = 0.4 if sbp > 135 else 0.2

    # Sleep Apnea
    risks["Sleep Apnea Risk"] = 0.5 if bmi > 30 else 0.2

    # Anemia
    risks["Anemia Risk"] = 0.3 if spo2 < 92 else 0.1

    return risks


def rules_risk_assessment(age, sex, sbp, dbp, hr, spo2, glucose, bmi, symptoms):
    """
    Return (scores, explanations).
    scores: dict of condition → 0–1 float
    explanations: dict of condition → reason string
    """
    scores = {}
    explain = {}

    risks = evaluate_risks(age, bmi, glucose, sbp, dbp, hr, spo2)
    for cond, val in risks.items():
        scores[cond] = round(val, 3)
        if cond == "Type 2 Diabetes Risk":
            explain[cond] = f"Glucose={glucose}, BMI={bmi}"
        elif cond == "Hypertension Risk":
            explain[cond] = f"SBP={sbp}, DBP={dbp}"
        elif cond == "Depression/Mood Concern":
            explain[cond] = f"Age={age}, Sex={sex}"
        elif cond == "Migraine Risk":
            explain[cond] = "Headache symptom" if "Headache" in symptoms else "Vitals"
        elif cond == "Sleep Apnea Risk":
            explain[cond] = "Snoring / daytime sleepiness" if any(s in symptoms for s in ["Loud snoring", "Daytime sleepiness"]) else "BMI factor"
        elif cond == "Anemia Risk":
            explain[cond] = f"SpO₂={spo2}"
        else:
            explain[cond] = "Rule-based heuristic"

    return scores, explain


def explain_rules(explain_dict):
    """Format explanation dict into readable text."""
    lines = [f"- **{cond}**: {reason}" for cond, reason in explain_dict.items()]
    return "\n".join(lines)
