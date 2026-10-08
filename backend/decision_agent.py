"""
CropCare AI - Agentic Decision & Risk Prediction Layer
Perception-Decision Loop: Correlates Vision outputs (Disease + Severity %)
with Weather Data and Crop Growth Stage to calculate Near-Term Risk & Actionable Prescriptions.
Author: CropCare AI Agronomy & Agentic AI Team
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("CropCareAgentic")

AGRONOMIC_PRESCRIPTIONS = {
    "Early Blight": {
        "fungicide_chemical": "Copper Oxychloride 50% WP @ 2.5 g/L or Mancozeb 75% WP @ 2.0 g/L",
        "bio_alternative": "Trichoderma viride @ 5 g/L + Neem Oil 10,000 ppm @ 2 ml/L",
        "action_en": "Apply contact fungicide spray within 48 hours. Remove heavily spotted lower foliage. Avoid overhead sprinkler irrigation.",
        "action_hi": "48 ghante ke andar Copper Oxychloride ya Mancozeb ka chhidkaw karein. Niche ki sankramit pattiyan tod kar hata dein. Fuhara sinchai na karein.",
        "prevention": "Ensure 60cm plant-to-plant spacing for airflow. Water directly at root zone."
    },
    "Late Blight": {
        "fungicide_chemical": "Metalaxyl 8% + Mancozeb 64% WP (Ridomil MZ) @ 2.5 g/L or Dimethomorph 50% WP @ 1 g/L",
        "bio_alternative": "Pseudomonas fluorescens @ 5 g/L spray at early dusk",
        "action_en": "CRITICAL: High spread risk. Apply systemic fungicide immediately within 24 hours. Destroy necrotic stems.",
        "action_hi": "ATYANT GAMBHIR: Rog tezi se fail raha hai. 24 ghante mein systemic fungicide (Metalaxyl + Mancozeb) ka chhidkaw karein.",
        "prevention": "Never leave infected debris in field. Strictly maintain furrow drainage."
    },
    "Yellow Rust / Stripe Rust": {
        "fungicide_chemical": "Propiconazole 25% EC (Tilt) @ 1.0 ml/L or Tebuconazole 25.9% EC @ 1.25 ml/L",
        "bio_alternative": "Neem seed kernel extract (NSKE) 5% preventive spray",
        "action_en": "Foliar spray required within 48 hours to prevent grain-filling yield drop.",
        "action_hi": "48 ghante ke andar Propiconazole ka chhidkaw karein taaki daane bharne mein nuksaan na ho.",
        "prevention": "Avoid excessive urea/nitrogen application; scout border rows daily."
    },
    "Bacterial Leaf Blight": {
        "fungicide_chemical": "Streptocycline (90:10) @ 6 g / 50 L water + Copper Oxychloride @ 50 g / 50 L water",
        "bio_alternative": "Pseudomonas fluorescens 0.2% foliar application",
        "action_en": "Drain standing field water if possible. Apply bactericide + copper combination spray.",
        "action_hi": "Khet se atirikt paani nikal dein. Streptocycline aur Copper Oxychloride ka mishran chhidkein.",
        "prevention": "Use balanced potash (K) to strengthen leaf epidermis against bacteria."
    },
    "Bacterial Blight / Angular Leaf Spot": {
        "fungicide_chemical": "Copper Hydroxide 53.8% DF @ 2.0 g/L + Streptocycline @ 100 ppm",
        "bio_alternative": "Bacillus subtilis formulation @ 3 ml/L",
        "action_en": "Spray copper bactericide. Do not enter field while plants are wet to avoid manual spread.",
        "action_hi": "Copper Hydroxide ka chhidkaw karein. Jab pattiya geeli hon tab khet mein kaam na karein.",
        "prevention": "Destruction of infected cotton crop residues post-harvest."
    }
}

class AgenticDecisionEngine:
    """
    Perception-Decision Loop:
    Correlates OpenCV 5 vision metrics with meteorological APIs and crop stage.
    """

    def __init__(self, aws_manager=None):
        self.aws = aws_manager

    def evaluate_perception_decision_loop(
        self,
        disease_name: str,
        severity_pct: float,
        crop_name: str = "Tomato",
        growth_stage: str = "Flowering",
        humidity_pct: float = 86.0,
        temperature_c: float = 29.0,
        rainfall_prob_pct: float = 65.0,
        wind_speed_kmh: float = 8.5
    ) -> Dict[str, Any]:
        """
        Calculates near-term risk projection (7 days) and actionable farmer prescription.
        """
        # Clean disease key
        clean_disease = "Early Blight"
        for key in AGRONOMIC_PRESCRIPTIONS.keys():
            if key.lower() in disease_name.lower():
                clean_disease = key
                break

        prescription = AGRONOMIC_PRESCRIPTIONS.get(clean_disease, AGRONOMIC_PRESCRIPTIONS["Early Blight"])

        # 1. Environmental Risk Coefficient
        # High humidity (>75%) heavily triggers spore germination
        humidity_factor = 1.35 if humidity_pct >= 80 else 1.15 if humidity_pct >= 65 else 0.85
        temp_factor = 1.25 if (20 <= temperature_c <= 32) else 0.90
        rain_risk_add = 15.0 if rainfall_prob_pct >= 60 else 8.0 if rainfall_prob_pct >= 30 else 0.0

        # 2. Stage Vulnerability
        stage_multipliers = {
            "Seedling": 1.15,
            "Vegetative": 1.0,
            "Flowering": 1.30,
            "Fruiting": 1.25,
            "Maturity / Harvesting": 0.85
        }
        stage_mult = stage_multipliers.get(growth_stage, 1.1)

        # 3. Base Near-term Risk Score (0-100%)
        calculated_risk = (severity_pct * humidity_factor * temp_factor * stage_mult) + rain_risk_add
        risk_score_7day = min(98.0, max(12.0, round(calculated_risk, 1)))

        # 4. Generate 7-Day Disease Risk Forecast Curve
        growth_daily_rate = 1.07 if humidity_pct >= 80 else 1.04
        trend_7days: List[Dict[str, Any]] = []
        curr = severity_pct
        for d in range(1, 8):
            trend_val = min(96.0, round(curr * (growth_daily_rate ** (d - 1)) + (d * 1.8 if rainfall_prob_pct > 50 else 0), 1))
            trend_7days.append({
                "day": f"Day {d}",
                "day_label": f"Day {d}",
                "projected_severity": trend_val,
                "projected_risk": min(100.0, round(trend_val * humidity_factor, 1))
            })

        # 5. Determine Spray Window & Action Urgency
        if rainfall_prob_pct >= 70:
            spray_window_hours = 24
            urgency = "URGENT"
            spray_advice = f"Fungicide spray required within 24 hours BEFORE heavy rain ({rainfall_prob_pct}% chance). Rain will wash off protectants and spread spores."
        elif severity_pct >= 30 or humidity_pct >= 80:
            spray_window_hours = 48
            urgency = "HIGH"
            spray_advice = "Fungicide spray required within 48 hours. Sustained high humidity (>80%) accelerates fungal spore growth."
        else:
            spray_window_hours = 72
            urgency = "MODERATE"
            spray_advice = "Preventative spray recommended within 72 hours. Maintain regular field scouting."

        # Check wind speed for spray drift safety
        if wind_speed_kmh > 14.0:
            wind_warning = f"Notice: Wind speed is {wind_speed_kmh} km/h. Avoid midday spraying to prevent chemical drift; spray during calm early morning hours."
        else:
            wind_warning = "Wind conditions are calm. Optimal for uniform foliar spray coverage."

        # 6. Optional AWS Bedrock Agentic Loop
        bedrock_advice = None
        if self.aws:
            bedrock_prompt = (
                f"You are CropCare AI Agronomist on AWS Bedrock. "
                f"Farmer Crop: {crop_name}, Stage: {growth_stage}. "
                f"Detected: {disease_name} at {severity_pct}% severity. "
                f"Weather: Humidity {humidity_pct}%, Temp {temperature_c}C, Rain {rainfall_prob_pct}%. "
                f"Provide concise, 2-sentence actionable advice with exact spray timing and fungicide recommendation."
            )
            bedrock_advice = self.aws.invoke_bedrock_agent(bedrock_prompt)

        action_summary_en = bedrock_advice or f"{spray_advice} {prescription['action_en']} Recommended dose: {prescription['fungicide_chemical']}."
        action_summary_hi = f"{prescription['action_hi']} Anumodit Matra: {prescription['fungicide_chemical']}."

        return {
            "disease_detected": clean_disease,
            "severity_percentage": severity_pct,
            "growth_stage": growth_stage,
            "risk_score_7day": risk_score_7day,
            "urgency": urgency,
            "spray_window_hours": spray_window_hours,
            "weather_correlation": {
                "humidity_pct": humidity_pct,
                "temperature_c": temperature_c,
                "rainfall_prob_pct": rainfall_prob_pct,
                "wind_speed_kmh": wind_speed_kmh,
                "wind_advisory": wind_warning
            },
            "trend_7days": trend_7days,
            "prescription": {
                "chemical_treatment": prescription["fungicide_chemical"],
                "bio_treatment": prescription["bio_alternative"],
                "prevention_guideline": prescription["prevention"],
                "action_summary_en": action_summary_en,
                "action_summary_hi": action_summary_hi
            },
            "agentic_source": "AWS Bedrock (Claude 3 Haiku)" if bedrock_advice else "CropCare Rule Engine / CIBRC Agronomic Expert"
        }
