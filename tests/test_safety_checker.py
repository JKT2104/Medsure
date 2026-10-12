"""Unit tests for clinical decision support, DDIs, LASA, and max dosage rules.
"""

import unittest
from medagent.clinical_parser import clinical_parser
from medagent.safety_checker import safety_checker


class TestSafetyChecker(unittest.TestCase):

    def test_warfarin_nsaid_critical_ddi(self):
        orders = [
            clinical_parser.parse_line("Tab Warfarin 5mg 1-0-0 x 14d", ocr_confidence=95.0),
            clinical_parser.parse_line("Tab Brufen 400mg 1-0-1 x 5d pc", ocr_confidence=92.0)
        ]
        eval_result = safety_checker.evaluate_prescription(orders)

        self.assertEqual(eval_result["overall_status"], "CRITICAL")
        ddi_alerts = [a for a in eval_result["alerts"] if a["category"] == "DRUG_INTERACTION"]
        self.assertGreater(len(ddi_alerts), 0)
        self.assertIn("Hemorrhage", ddi_alerts[0]["title"])

    def test_lasa_confusion_detection(self):
        orders = [
            clinical_parser.parse_line("Tab Celebrex 200mg 1-0-1 x 10d", ocr_confidence=90.0)
        ]
        eval_result = safety_checker.evaluate_prescription(orders)

        lasa_alerts = [a for a in eval_result["alerts"] if a["category"] == "LASA_WARNING"]
        self.assertGreater(len(lasa_alerts), 0)
        self.assertIn("Celexa", lasa_alerts[0]["title"])

    def test_maximum_daily_dose_violation(self):
        # Paracetamol max is 4000mg/day. 1000mg x 5 times = 5000mg/day -> Violation!
        order = clinical_parser.parse_line("Tab Paracetamol 1000mg 5 times daily", ocr_confidence=95.0)
        order["daily_multiplier"] = 5
        order["daily_dose_mg"] = 5000.0

        eval_result = safety_checker.evaluate_prescription([order])
        dose_alerts = [a for a in eval_result["alerts"] if a["category"] == "DOSAGE_LIMIT_EXCEEDED"]

        self.assertGreater(len(dose_alerts), 0)
        self.assertEqual(eval_result["overall_status"], "CRITICAL")
        self.assertIn("4000", dose_alerts[0]["message"])

    def test_augmentin_ketorol_interaction(self):
        orders = [
            clinical_parser.parse_line("Tab Augmentin 625mg 1-0-1 x 5d", ocr_confidence=95.0),
            clinical_parser.parse_line("Tab Ketorol DT 10mg 1-0-1 x 3d", ocr_confidence=95.0)
        ]
        eval_result = safety_checker.evaluate_prescription(orders)
        ddi_alerts = [a for a in eval_result["alerts"] if a["category"] == "DRUG_INTERACTION"]
        self.assertGreater(len(ddi_alerts), 0)
        self.assertIn("Ketorolac", ddi_alerts[0]["title"])

    def test_paediatric_weight_dose_validation_and_overdose(self):
        # Safe dose: Meftal-P 3ml (60mg) for 13.25 kg child (4.5 mg/kg within 6.5-8 mg/kg range)
        safe_order = clinical_parser.parse_line("Syp Meftal-P (100/5) 3ml SOS", ocr_confidence=95.0)
        eval_safe = safety_checker.evaluate_prescription([safe_order], patient_details={"weight": "13.25 kg"})
        self.assertEqual(eval_safe["overall_status"], "SAFE")
        valid_alerts = [a for a in eval_safe["alerts"] if a["category"] == "PAEDIATRIC_DOSE_VALIDATION"]
        self.assertGreater(len(valid_alerts), 0)

        # Overdose: 15ml (300mg) Meftal-P for 13.25 kg child (> 22 mg/kg, ceiling is ~106 mg) -> CRITICAL
        overdose_order = clinical_parser.parse_line("Syp Meftal-P (100/5) 15ml TDS", ocr_confidence=95.0)
        eval_overdose = safety_checker.evaluate_prescription([overdose_order], patient_details={"weight": "13.25 kg"})
        self.assertEqual(eval_overdose["overall_status"], "CRITICAL")
        overdose_alerts = [a for a in eval_overdose["alerts"] if a["category"] == "PAEDIATRIC_OVERDOSE"]
        self.assertGreater(len(overdose_alerts), 0)
        self.assertIn("13.25", overdose_alerts[0]["message"])


if __name__ == '__main__':
    unittest.main()
