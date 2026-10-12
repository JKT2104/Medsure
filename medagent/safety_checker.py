"""Clinical Safety & Decision Support Engine.
Performs Look-Alike Sound-Alike (LASA) analysis, maximum dosage safety checks,
Drug-Drug Interaction (DDI) detection, and OCR uncertainty risk flagging.
"""

from typing import List, Dict, Any, Tuple, Optional
from itertools import combinations
import re

# High-risk Look-Alike Sound-Alike (LASA) pairs based on ISMP & FDA list
# Includes Tall Man lettering for clear visual distinction
LASA_PAIRS = [
    {
        "pair": ("Celebrex", "Celexa"),
        "generics": ("Celecoxib", "Citalopram"),
        "tall_man": ("celeCOXIB (Celebrex)", "citALOPram (Celexa)"),
        "risk_level": "CRITICAL",
        "description": "Celebrex (NSAID for pain) confused with Celexa (SSRI antidepressant). Misdispensing risks untreated depression or GI bleed."
    },
    {
        "pair": ("Clonidine", "Clonazepam"),
        "generics": ("Clonidine", "Clonazepam"),
        "tall_man": ("cloniDINE", "clonaZEPAM"),
        "risk_level": "CRITICAL",
        "description": "Clonidine (antihypertensive) confused with Clonazepam (sedative/anticonvulgant). Risk of severe hypotension or excessive sedation."
    },
    {
        "pair": ("Hydralazine", "Hydroxyzine"),
        "generics": ("Hydralazine", "Hydroxyzine"),
        "tall_man": ("hydrALAZINE", "hydrOXYzine"),
        "risk_level": "CRITICAL",
        "description": "Hydralazine (arterial vasodilator) confused with Hydroxyzine (antihistamine). Risk of profound hypotension or anticholinergic toxidrome."
    },
    {
        "pair": ("Metformin", "Metronidazole"),
        "generics": ("Metformin", "Metronidazole"),
        "tall_man": ("metFORMIN", "metroNIDAZOLE"),
        "risk_level": "WARNING",
        "description": "Metformin (antidiabetic) confused with Metronidazole (antibacterial/antiprotozoal)."
    },
    {
        "pair": ("Lamictal", "Lamisil"),
        "generics": ("Lamotrigine", "Terbinafine"),
        "tall_man": ("LAMICTAL", "LAMISIL"),
        "risk_level": "CRITICAL",
        "description": "Anticonvulsant confused with antifungal. Risk of Stevens-Johnson syndrome or breakthrough seizures."
    },
    {
        "pair": ("Amlodipine", "Amiloride"),
        "generics": ("Amlodipine", "Amiloride"),
        "tall_man": ("amLODIPine", "aMILoride"),
        "risk_level": "WARNING",
        "description": "Calcium channel blocker confused with potassium-sparing diuretic."
    },
    {
        "pair": ("Azithral", "Allegra"),
        "generics": ("Azithromycin", "Fexofenadine"),
        "tall_man": ("azithroMYCIN (Azithral)", "fexofenADINE (Allegra)"),
        "risk_level": "WARNING",
        "description": "Azithromycin (macrolide antibiotic) confused with Allegra/Fexofenadine (antihistamine). Risk of untreated bacterial infection or inappropriate antibiotic use."
    },
    {
        "pair": ("Taxim-O", "Taxol"),
        "generics": ("Cefixime", "Paclitaxel"),
        "tall_man": ("taxim-O (Cefixime)", "TAXOL (Paclitaxel)"),
        "risk_level": "CRITICAL",
        "description": "Oral antibiotic (Cefixime) confused with IV antineoplastic chemotherapeutic (Paclitaxel). Severe toxicity risk if chemotherapeutic misidentified."
    },
    {
        "pair": ("Metformin", "Metoprolol"),
        "generics": ("Metformin", "Metoprolol"),
        "tall_man": ("metFORMIN", "metOPROLOL"),
        "risk_level": "WARNING",
        "description": "Metformin (antidiabetic) confused with Metoprolol (beta-blocker). Risk of uncontrolled hyperglycemia or profound bradycardia/hypotension."
    },
    {
        "pair": ("Losar", "Lasix"),
        "generics": ("Losartan", "Furosemide"),
        "tall_man": ("loSARtan (Losar)", "laSIX (Furosemide)"),
        "risk_level": "WARNING",
        "description": "Losar/Losartan (ARB antihypertensive) confused with Lasix/Furosemide (potent loop diuretic). Risk of profound dehydration or electrolyte depletion."
    }
]

# Clinical Drug-Drug Interactions Matrix
DDI_RULES = [
    {
        "drugs": ("Warfarin", "Ibuprofen"),
        "severity": "CRITICAL",
        "title": "Severe Bleeding & Hemorrhage Risk",
        "description": "Ibuprofen inhibits platelet COX-1 and causes gastric mucosal injury, drastically elevating GI bleeding risk when combined with Warfarin.",
        "recommendation": "Avoid combination. Use Paracetamol for analgesia; closely monitor INR if concurrent use is unavoidable."
    },
    {
        "drugs": ("Warfarin", "Aspirin"),
        "severity": "CRITICAL",
        "title": "Dual Antithrombotic Hemorrhagic Risk",
        "description": "Concurrent anticoagulant and antiplatelet therapy significantly multiplies major bleeding risk.",
        "recommendation": "Prescribe together only when strictly indicated (e.g. recent coronary stent + mechanical valve) with PPI gastroprotection."
    },
    {
        "drugs": ("Lisinopril", "Spironolactone"),
        "severity": "MAJOR",
        "title": "Severe Hyperkalemia Risk",
        "description": "ACE inhibitors reduce aldosterone production, which when paired with potassium-sparing diuretics can trigger life-threatening hyperkalemia.",
        "recommendation": "Monitor serum potassium (K+) and creatinine within 1-2 weeks of initiation."
    },
    {
        "drugs": ("Losartan", "Spironolactone"),
        "severity": "MAJOR",
        "title": "Severe Hyperkalemia Risk",
        "description": "ARB reduces aldosterone secretion; combining with potassium-sparing diuretic increases hyperkalemia risk.",
        "recommendation": "Regular electrolyte and renal function monitoring required."
    },
    {
        "drugs": ("Atorvastatin", "Azithromycin"),
        "severity": "MODERATE",
        "title": "Elevated Statin Levels & Myopathy Risk",
        "description": "Azithromycin and macrolides may elevate serum statin concentration, increasing the risk of muscle soreness, myopathy, and rhabdomyolysis.",
        "recommendation": "Monitor patient for unexplained muscle pain, tenderness, or weakness."
    },
    {
        "drugs": ("Ciprofloxacin", "Ibuprofen"),
        "severity": "MODERATE",
        "title": "Enhanced CNS Toxicity & Seizure Risk",
        "description": "Concurrent fluoroquinolones and NSAIDs may potentiate GABA antagonism, increasing neuro-excitability and seizure risk.",
        "recommendation": "Use caution in patients with seizure history or renal impairment."
    },
    {
        "drugs": ("Ibuprofen", "Celecoxib"),
        "severity": "MAJOR",
        "title": "Duplicate NSAID Therapy",
        "description": "Prescribing two non-steroidal anti-inflammatory agents simultaneously increases GI ulceration and renal toxicity without additive pain relief.",
        "recommendation": "Discontinue one agent; avoid co-prescribing multiple NSAIDs."
    },
    {
        "drugs": ("Ketorolac Tromethamine", "Ibuprofen"),
        "severity": "CRITICAL",
        "title": "Duplicate Potent NSAID Therapy & GI Perforation Risk",
        "description": "Combining Ketorolac (potent parenteral/oral NSAID) with another NSAID substantially increases risk of severe GI ulceration, bleeding, and acute renal failure.",
        "recommendation": "Contraindicated combination. Ketorolac must never be combined with another NSAID or aspirin."
    },
    {
        "drugs": ("Warfarin", "Ketorolac Tromethamine"),
        "severity": "CRITICAL",
        "title": "Severe Hemorrhage & GI Bleeding Risk",
        "description": "Ketorolac strongly inhibits platelet function and causes gastric mucosa irritation, severely elevating major bleeding when given with Warfarin.",
        "recommendation": "Absolute contraindication. Avoid concurrent use."
    },
    {
        "drugs": ("Itraconazole", "Atorvastatin"),
        "severity": "CRITICAL",
        "title": "Severe Statin Toxicity & Rhabdomyolysis Risk",
        "description": "Itraconazole is a potent CYP3A4 inhibitor that increases serum Atorvastatin concentrations up to 10-fold, creating life-threatening rhabdomyolysis risk.",
        "recommendation": "Temporarily suspend Atorvastatin during Itraconazole antifungal therapy, or substitute with Pravastatin/Rosuvastatin."
    },
    {
        "drugs": ("Amoxicillin + Clavulanic Acid", "Ketorolac Tromethamine"),
        "severity": "MODERATE",
        "title": "GI Mucosal & Nephrotoxicity Caution",
        "description": "High-dose Ketorolac causes gastric mucosal erosion and reduces renal clearance, potentiating GI intolerance when co-administered with clavulanate.",
        "recommendation": "Co-prescribe with gastroprotection (PPI e.g., Pantoprazole); limit Ketorolac duration to ≤ 5 days."
    },
    {
        "drugs": ("Diclofenac + Paracetamol + Serratiopeptidase", "Ketorolac Tromethamine"),
        "severity": "CRITICAL",
        "title": "Severe Duplicate NSAID Therapy",
        "description": "Co-prescribing Ketorolac with Diclofenac (in Enzoflam) creates profound double-NSAID COX inhibition, multiplying risk of severe GI hemorrhage and acute renal failure.",
        "recommendation": "Contraindicated. Discontinue one NSAID agent immediately."
    },
    {
        "drugs": ("Ibuprofen", "Aspirin"),
        "severity": "CRITICAL",
        "title": "Aspirin Antiplatelet Antagonism & Severe GI Bleeding Risk",
        "description": "Ibuprofen competitively blocks the irreversible acetylation of platelet COX-1 by low-dose Aspirin, diminishing its cardioprotective effect. Concurrently, combining multiple NSAIDs multiplies gastrointestinal ulceration and major haemorrhage risk.",
        "recommendation": "If cardioprotective aspirin is required, take aspirin at least 30 minutes to 2 hours before ibuprofen, or substitute with Paracetamol for analgesia. Add PPI gastroprotection."
    },
    {
        "drugs": ("Ibuprofen", "Aceclofenac + Paracetamol + Serratiopeptidase"),
        "severity": "CRITICAL",
        "title": "Duplicate NSAID Therapy (Ibuprofen + Aceclofenac)",
        "description": "Concurrent administration of two systemic NSAIDs provides no additional analgesia while substantially compounding the hazard of upper GI ulceration, bleeding, and acute renal impairment.",
        "recommendation": "Contraindicated duplicate NSAID regimen. Discontinue one anti-inflammatory agent immediately."
    },
    {
        "drugs": ("Ibuprofen", "Diclofenac + Paracetamol + Serratiopeptidase"),
        "severity": "CRITICAL",
        "title": "Duplicate NSAID Therapy (Ibuprofen + Diclofenac)",
        "description": "Co-prescribing multiple NSAIDs compounds nephrotoxicity and gastrointestinal ulceration risk without incremental analgesic benefit.",
        "recommendation": "Discontinue one NSAID agent; maintain monotherapy with gastroprotection if required."
    },
    {
        "drugs": ("Aceclofenac + Paracetamol + Serratiopeptidase", "Ketorolac Tromethamine"),
        "severity": "CRITICAL",
        "title": "Severe Duplicate NSAID Therapy (Aceclofenac + Ketorolac)",
        "description": "Combining Ketorolac with Aceclofenac causes extreme double-COX inhibition, drastically escalating risk of fatal GI bleeding, perforation, and acute renal failure.",
        "recommendation": "Contraindicated. Ketorolac must never be combined with another NSAID or aspirin."
    },
    {
        "drugs": ("Ciprofloxacin", "Aluminium Hydroxide + Magnesium Hydroxide + Simethicone"),
        "severity": "MAJOR",
        "title": "Chelation & Antibiotic Bioavailability Impairment",
        "description": "Polyvalent metal cations (Al3+, Mg2+) in antacids bind to Ciprofloxacin, forming insoluble chelate complexes that reduce antibiotic absorption by up to 90%, risking therapeutic treatment failure.",
        "recommendation": "Administer Ciprofloxacin at least 2 hours before or 4 hours after antacids containing aluminium, magnesium, or calcium."
    },
    {
        "drugs": ("Ofloxacin", "Aluminium Hydroxide + Magnesium Hydroxide + Simethicone"),
        "severity": "MAJOR",
        "title": "Chelation & Fluoroquinolone Malabsorption",
        "description": "Multivalent cations in antacids precipitate Ofloxacin absorption failure through cation chelation.",
        "recommendation": "Space administration: take Ofloxacin 2 hours prior or 4 to 6 hours after antacids."
    },
    {
        "drugs": ("Azithromycin", "Ciprofloxacin"),
        "severity": "MAJOR",
        "title": "Additive QTc Prolongation Risk",
        "description": "Concurrent use of macrolide and fluoroquinolone antibiotics exerts additive electrophysiological effects on ventricular repolarization, increasing risk of QTc prolongation and Torsades de Pointes.",
        "recommendation": "Monitor ECG QTc interval, especially in patients with pre-existing cardiac conditions or electrolyte derangements."
    },
    {
        "drugs": ("Escitalopram + Clonazepam", "Ciprofloxacin"),
        "severity": "MAJOR",
        "title": "Additive QTc Prolongation & Seizure Risk",
        "description": "Ciprofloxacin can inhibit CYP1A2/CYP3A4 and additively prolong cardiac QTc interval and lower seizure threshold when paired with Escitalopram.",
        "recommendation": "Use alternative antibiotic if feasible; baseline and periodic ECG monitoring recommended."
    },
    {
        "drugs": ("Telmisartan", "Ibuprofen"),
        "severity": "MAJOR",
        "title": "Blunted Antihypertensive Effect & Acute Renal Impairment Risk",
        "description": "NSAIDs inhibit renal vasodilatory prostaglandins, decreasing renal perfusion and directly counteracting the blood pressure lowering effect of ARBs (Telmisartan). Increases risk of acute kidney injury (AKI).",
        "recommendation": "Avoid chronic concurrent use; monitor renal function, serum potassium, and blood pressure regularly."
    },
    {
        "drugs": ("Telmisartan", "Aceclofenac + Paracetamol + Serratiopeptidase"),
        "severity": "MAJOR",
        "title": "Impaired Renal Perfusion & Antihypertensive Attenuation",
        "description": "Aceclofenac attenuates ARB antihypertensive efficacy and impairs renal blood flow, heightening acute kidney injury risk.",
        "recommendation": "Use short-course analgesic only, ensure adequate hydration, and monitor blood pressure."
    },
    {
        "drugs": ("Losartan", "Ibuprofen"),
        "severity": "MAJOR",
        "title": "Reduced Antihypertensive Efficacy & Nephrotoxicity Risk",
        "description": "Concurrent NSAID therapy attenuates Losartan's antihypertensive effect and increases risk of acute renal failure.",
        "recommendation": "Monitor BP and renal profile; use Paracetamol when possible."
    },
    {
        "drugs": ("Clopidogrel", "Ibuprofen"),
        "severity": "CRITICAL",
        "title": "Synergistic Antiplatelet Inhibition & Major GI Bleed Risk",
        "description": "Dual inhibition of platelet aggregation and gastric mucosal protection elevates risk of gastrointestinal hemorrhage.",
        "recommendation": "Avoid combination if possible; prescribe concurrent PPI gastroprotection and monitor hemoglobin."
    },
    {
        "drugs": ("Clopidogrel", "Ketorolac Tromethamine"),
        "severity": "CRITICAL",
        "title": "Severe Bleeding & Hemorrhage Contraindication",
        "description": "Ketorolac combined with antiplatelet therapy (Clopidogrel) produces profound bleeding risk and is contraindicated.",
        "recommendation": "Contraindicated. Discontinue Ketorolac; select alternative analgesic."
    },
    {
        "drugs": ("Tramadol", "Escitalopram + Clonazepam"),
        "severity": "CRITICAL",
        "title": "Severe Serotonin Syndrome & Seizure Risk",
        "description": "Tramadol inhibits serotonin and norepinephrine reuptake. Combining with an SSRI (Escitalopram) sharply increases the risk of life-threatening Serotonin Syndrome and lowers the seizure threshold.",
        "recommendation": "Avoid concurrent use or use with extreme caution. Watch for autonomic instability, neuromuscular hyperreactivity, and mental status changes."
    },
    {
        "drugs": ("Digoxin", "Spironolactone"),
        "severity": "MAJOR",
        "title": "Elevated Serum Digoxin Concentration",
        "description": "Spironolactone reduces digoxin clearance and may interfere with digoxin assays.",
        "recommendation": "Monitor serum digoxin concentration and watch for digitalis toxicity (nausea, arrhythmias)."
    }
]


def _drug_matches(target_spec: str, prescribed_drug: str) -> bool:
    """Checks if a target drug name matches a prescribed medication or one of its active components."""
    if not prescribed_drug or not target_spec:
        return False
    t_clean = re.sub(r'\(.*?\)', '', target_spec).strip().lower()
    p_clean = re.sub(r'\(.*?\)', '', prescribed_drug).strip().lower()
    if t_clean == p_clean:
        return True

    prescribed_components = [re.sub(r'\(.*?\)', '', c).strip().lower() for c in prescribed_drug.split("+")]
    target_components = [re.sub(r'\(.*?\)', '', c).strip().lower() for c in target_spec.split("+")]

    if len(target_components) == 1 and target_components[0] in prescribed_components:
        return True
    if len(target_components) > 1 and all(tc in prescribed_components for tc in target_components):
        return True
    return False


class SafetyChecker:
    """Evaluates prescription safety, flagging dosage violations, DDIs, LASA, paediatric weight limits, and OCR confidence."""

    def evaluate_prescription(
        self,
        orders: List[Dict[str, Any]],
        patient_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Runs comprehensive safety checks across all parsed medication orders."""
        alerts: List[Dict[str, Any]] = []
        overall_status = "SAFE"  # SAFE, WARNING, CRITICAL

        # 1. OCR Uncertainty Flags
        for i, order in enumerate(orders):
            ocr_conf = order.get("ocr_confidence", 100.0)
            match_conf = order.get("match_confidence", 100.0)
            raw = order.get("raw_text", "")
            det_name = order.get("detected_name", "")
            val_name = order.get("validated_drug", "")

            if order.get("is_uncertain") or ocr_conf < 75.0 or match_conf < 75.0:
                alerts.append({
                    "category": "OCR_UNCERTAINTY",
                    "severity": "WARNING",
                    "line_index": i,
                    "target_drug": val_name,
                    "title": f"Low OCR / Drug Recognition Confidence ({ocr_conf}%)",
                    "message": f"Line '{raw}' transcribed as '{det_name}'. Matched to '{val_name}' with {match_conf}% confidence. Quirky doctor handwriting requires manual verification.",
                    "action": "Pharmacist review required before dispensing."
                })
                if overall_status == "SAFE":
                    overall_status = "WARNING"

        # 2. Maximum Dosage Checks
        for i, order in enumerate(orders):
            daily_dose = order.get("daily_dose_mg")
            drug_meta = order.get("drug_meta")
            val_name = order.get("validated_drug", "")

            if drug_meta and daily_dose is not None:
                max_daily = drug_meta.get("max_daily_dose_mg")
                if max_daily and daily_dose > max_daily:
                    alerts.append({
                        "category": "DOSAGE_LIMIT_EXCEEDED",
                        "severity": "CRITICAL",
                        "line_index": i,
                        "target_drug": val_name,
                        "title": f"Max Safe Daily Dose Exceeded for {val_name}",
                        "message": f"Calculated daily dose is {daily_dose:g} mg, which exceeds the recommended maximum daily threshold of {max_daily:g} mg.",
                        "action": "Confirm dosage with prescribing physician immediately."
                    })
                    overall_status = "CRITICAL"

        # 3. Paediatric Weight-Based Dosage Checks & Contraindications
        patient_wt_kg = None
        if patient_details and patient_details.get("weight"):
            wt_match = re.search(r'(\d+(?:\.\d+)?)\s*kg', str(patient_details["weight"]).lower())
            if wt_match:
                patient_wt_kg = float(wt_match.group(1))

        patient_age_yrs = None
        if patient_details:
            age_raw = str(patient_details.get("age_gender", "") or patient_details.get("age", ""))
            age_m = re.search(r'\b(\d{1,3})\s*(?:y|yr|yrs|year|years)?\b', age_raw, re.IGNORECASE)
            if age_m:
                patient_age_yrs = float(age_m.group(1))

        if patient_wt_kg and patient_wt_kg > 0:
            for i, order in enumerate(orders):
                val_name = order.get("validated_drug", "")
                strength_mg = order.get("strength_mg")

                # Mefenamic Acid (Meftal-P): Max single dose 8-10 mg/kg; Max daily 25 mg/kg
                if "Mefenamic Acid" in val_name:
                    max_single_safe = round(patient_wt_kg * 8.0, 1)
                    if strength_mg and strength_mg > (patient_wt_kg * 12.0):
                        alerts.append({
                            "category": "PAEDIATRIC_OVERDOSE",
                            "severity": "CRITICAL",
                            "line_index": i,
                            "target_drug": val_name,
                            "title": f"Paediatric Weight Overdose Alert for {val_name}",
                            "message": f"Patient weight is {patient_wt_kg} kg. Prescribed dose of {strength_mg:g} mg exceeds the maximum single paediatric ceiling (safe ceiling: {max_single_safe} mg at 8 mg/kg).",
                            "action": "Reduce paediatric suspension dose volume immediately."
                        })
                        overall_status = "CRITICAL"
                    elif strength_mg:
                        actual_mg_per_kg = round(strength_mg / patient_wt_kg, 1)
                        alerts.append({
                            "category": "PAEDIATRIC_DOSE_VALIDATION",
                            "severity": "INFO",
                            "line_index": i,
                            "target_drug": val_name,
                            "title": f"Paediatric Weight-Based Dose Verified ({actual_mg_per_kg} mg/kg)",
                            "message": f"Dose {strength_mg:g} mg for patient weight {patient_wt_kg} kg corresponds to {actual_mg_per_kg} mg/kg (safe therapeutic range: 6.5 - 8.0 mg/kg, max 25 mg/kg/day).",
                            "action": "Safe paediatric dose verified."
                        })

                # Paracetamol (Calpol): Max single dose 15 mg/kg; Max daily 60 mg/kg
                elif "Paracetamol" in val_name and order.get("dosage_form") == "Syrup":
                    max_single_safe = round(patient_wt_kg * 15.0, 1)
                    if strength_mg and strength_mg > (patient_wt_kg * 20.0):
                        alerts.append({
                            "category": "PAEDIATRIC_OVERDOSE",
                            "severity": "CRITICAL",
                            "line_index": i,
                            "target_drug": val_name,
                            "title": f"Paediatric Paracetamol Overdose Alert",
                            "message": f"Prescribed dose of {strength_mg:g} mg exceeds maximum paediatric ceiling for {patient_wt_kg} kg (max safe single dose: {max_single_safe} mg at 15 mg/kg). Risk of hepatotoxicity.",
                            "action": "Adjust syrup dose volume immediately."
                        })
                        overall_status = "CRITICAL"
                    elif strength_mg:
                        actual_mg_per_kg = round(strength_mg / patient_wt_kg, 1)
                        alerts.append({
                            "category": "PAEDIATRIC_DOSE_VALIDATION",
                            "severity": "INFO",
                            "line_index": i,
                            "target_drug": val_name,
                            "title": f"Paediatric Weight-Based Dose Verified ({actual_mg_per_kg} mg/kg)",
                            "message": f"Dose {strength_mg:g} mg for patient weight {patient_wt_kg} kg corresponds to {actual_mg_per_kg} mg/kg (safe therapeutic range: 10 - 15 mg/kg).",
                            "action": "Safe paediatric dose verified."
                        })

        # Paediatric Absolute Contraindication: Aspirin (Reye's syndrome)
        is_paediatric = (patient_age_yrs is not None and patient_age_yrs < 16) or (patient_wt_kg and patient_wt_kg < 40.0)
        if is_paediatric:
            for i, order in enumerate(orders):
                val_name = order.get("validated_drug", "")
                if _drug_matches("Aspirin", val_name):
                    alerts.append({
                        "category": "PAEDIATRIC_CONTRAINDICATION",
                        "severity": "CRITICAL",
                        "line_index": i,
                        "target_drug": val_name,
                        "title": "Fatal Reye's Syndrome Risk (Aspirin in Paediatrics)",
                        "message": f"Aspirin is strictly contraindicated in children and adolescents under 16 years of age (patient: {patient_details.get('age_gender', '') or f'{patient_wt_kg} kg'}). Administration during viral illness precipitates Reye's syndrome—a severe, frequently fatal condition causing acute non-inflammatory encephalopathy and fatty liver failure.",
                        "action": "Absolute contraindication. Discontinue Aspirin immediately; use Paracetamol or Ibuprofen for analgesia and fever."
                    })
                    overall_status = "CRITICAL"

        # Paediatric Fluoroquinolone Restriction (< 18 years): Arthropathy & cartilage toxicity
        is_adolescent = (patient_age_yrs is not None and patient_age_yrs < 18) or (patient_wt_kg and patient_wt_kg < 45.0)
        if is_adolescent:
            for i, order in enumerate(orders):
                val_name = order.get("validated_drug", "")
                if any(_drug_matches(fq, val_name) for fq in ["Ciprofloxacin", "Ofloxacin", "Norfloxacin + Tinidazole"]):
                    alerts.append({
                        "category": "PAEDIATRIC_WARNING",
                        "severity": "MAJOR",
                        "line_index": i,
                        "target_drug": val_name,
                        "title": f"Paediatric Arthropathy & Cartilage Toxicity Alert ({val_name})",
                        "message": f"Fluoroquinolones ({val_name}) are restricted in paediatric patients under 18 years due to established risk of cartilage erosion, arthropathy, and tendinopathy in weight-bearing joints.",
                        "action": "Evaluate risk-benefit ratio. Consider safer paediatric antibiotic alternatives (e.g. Amoxicillin-Clavulanate, Cefixime, Azithromycin)."
                    })
                    if overall_status != "CRITICAL":
                        overall_status = "CRITICAL"

        # 4. Look-Alike Sound-Alike (LASA) Checks
        prescribed_generics = [order.get("validated_drug") for order in orders if order.get("validated_drug")]
        for i, order in enumerate(orders):
            val_drug = order.get("validated_drug", "")
            det_drug = order.get("detected_name", "")

            for lasa in LASA_PAIRS:
                g1, g2 = lasa["generics"]
                b1, b2 = lasa["pair"]
                if val_drug in (g1, g2) or det_drug.lower() in (b1.lower(), b2.lower()):
                    other_drug = g2 if val_drug == g1 else g1
                    alerts.append({
                        "category": "LASA_WARNING",
                        "severity": lasa["risk_level"],
                        "line_index": i,
                        "target_drug": val_drug,
                        "title": f"High-Risk Sound-Alike / Look-Alike Drug ({lasa['tall_man'][0]} vs {lasa['tall_man'][1]})",
                        "message": f"'{val_drug}' is a recognized high-risk LASA medication. Caution: do not confuse with '{other_drug}'. {lasa['description']}",
                        "action": "Verify indication and drug spelling carefully."
                    })
                    if lasa["risk_level"] == "CRITICAL":
                        overall_status = "CRITICAL"
                    elif overall_status == "SAFE":
                        overall_status = "WARNING"

        # 5. Drug-Drug Interactions (DDI) Checks
        valid_orders = [o for o in orders if o.get("validated_drug") and o.get("validated_drug") != "Unverified / Unknown"]
        seen_ddi_keys = set()
        for o1, o2 in combinations(valid_orders, 2):
            d1 = o1.get("validated_drug")
            d2 = o2.get("validated_drug")

            for rule in DDI_RULES:
                p1, p2 = rule["drugs"]
                if (_drug_matches(p1, d1) and _drug_matches(p2, d2)) or (_drug_matches(p1, d2) and _drug_matches(p2, d1)):
                    ddi_key = (rule["title"], min(d1, d2), max(d1, d2))
                    if ddi_key in seen_ddi_keys:
                        continue
                    seen_ddi_keys.add(ddi_key)

                    alerts.append({
                        "category": "DRUG_INTERACTION",
                        "severity": rule["severity"],
                        "target_drug": f"{d1} + {d2}",
                        "title": f"{rule['title']} ({d1} + {d2})",
                        "message": rule["description"],
                        "action": rule["recommendation"]
                    })
                    if rule["severity"] == "CRITICAL":
                        overall_status = "CRITICAL"
                    elif rule["severity"] == "MAJOR" and overall_status != "CRITICAL":
                        overall_status = "CRITICAL"
                    elif overall_status == "SAFE":
                        overall_status = "WARNING"

        return {
            "overall_status": overall_status,
            "total_alerts": len(alerts),
            "alerts": alerts
        }


# Singleton instance
safety_checker = SafetyChecker()
