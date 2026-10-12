"""Unit tests for comprehensive prescription line categorization and zero-drop guarantee."""
import unittest
from medagent.clinical_parser import clinical_parser


class TestLineCategorization(unittest.TestCase):

    def test_real_world_medications_categorized(self):
        """Tests that common Indian real-world medications are correctly recognized and not dropped."""
        lines = [
            {"text": "Tab Augmentin 625mg 1-0-1 x 5 days", "confidence": 95.0},
            {"text": "Tab Allegra 120 1-0-0 x 5d", "confidence": 93.0},
            {"text": "Combiflam 1-0-1 x 3 days", "confidence": 90.0},
            {"text": "Syp Ascoril D 5ml TDS x 5 days", "confidence": 92.0},
            {"text": "Supradyn 1 OD x 15 days", "confidence": 89.0},
            {"text": "Folvite 5mg 1 OD x 30 days", "confidence": 94.0},
            {"text": "Gudcef 200 1-0-1 x 5d", "confidence": 91.0},
        ]
        orders = clinical_parser.parse_prescription(lines)
        self.assertEqual(len(orders), 7, f"Expected 7 orders, got {len(orders)}")
        
        # Verify unverified brand with dosing was kept
        gudcef = next((o for o in orders if "Gudcef" in o["corrected_name"] or "Gudcef" in o["detected_name"]), None)
        self.assertIsNotNone(gudcef)
        self.assertEqual(gudcef["strength"], "200mg")
        self.assertEqual(gudcef["frequency"], "Twice daily (Morning & Night)")
        self.assertEqual(gudcef["duration"], "5 days")

    def test_vitals_and_demographics_separated(self):
        """Tests that patient weight, vitals, and demographics are separated from orders and update patient clinical profile."""
        lines = [
            {"text": "Patient: Ramesh Kumar  Age: 42 Y / Male  Wt: 68 kg", "confidence": 92.0},
            {"text": "BP: 130/85 mmHg  Pulse: 76 bpm  Temp: 99.1 F", "confidence": 90.0},
            {"text": "Diagnosis: Acute Bronchitis with productive cough", "confidence": 88.0},
            {"text": "Tab Augmentin 625mg 1-0-1 x 5 days", "confidence": 95.0}
        ]
        orders = clinical_parser.parse_prescription(lines)
        
        # Exactly 1 medication order, 0 vitals as medications
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["validated_drug"], "Amoxicillin + Clavulanic Acid")
        
        # Patient updates populated
        self.assertEqual(clinical_parser.patient_updates.get("weight"), "68 kg")
        self.assertEqual(clinical_parser.patient_updates.get("bp"), "130/85 mmHg")
        self.assertEqual(clinical_parser.patient_updates.get("patient_name"), "Ramesh Kumar")
        self.assertIn("Bronchitis", clinical_parser.patient_updates.get("diagnosis", ""))

    def test_clinical_advice_and_directives_categorized(self):
        """Tests that care directives, oral care, dietary advice, and reviews are cleanly categorized."""
        lines = [
            {"text": "Tab Augmentin 625mg 1-0-1 x 5d", "confidence": 95.0},
            {"text": "Warm saline gargle 3-4 times a day", "confidence": 91.0},
            {"text": "Steam inhalation twice daily x 5 days", "confidence": 93.0},
            {"text": "Soft diet and avoid cold drinks", "confidence": 90.0},
            {"text": "Review after 5 days or SOS if fever persists", "confidence": 94.0},
            {"text": "Fasting blood sugar test after 1 week", "confidence": 89.0},
            {"text": "Bed rest for 3 days", "confidence": 92.0}
        ]
        orders = clinical_parser.parse_prescription(lines)
        self.assertEqual(len(orders), 1)
        
        advice = clinical_parser.clinical_advice
        self.assertGreaterEqual(len(advice), 6)
        
        categories = [a["category"] for a in advice]
        self.assertIn("Oral Hygiene & Care", categories)
        self.assertIn("Physical Therapy & Home Care", categories)
        self.assertIn("Dietary Directives & Hydration", categories)
        self.assertIn("Follow-Up & Doctor Review", categories)
        self.assertIn("Diagnostic & Lab Investigations", categories)
        self.assertIn("Post-Procedure Recovery / Rest", categories)

    def test_categorization_audit_records_all_lines(self):
        """Tests that categorized_lines audit log accounts for 100% of non-noise input lines."""
        lines = [
            {"text": "Patient: Anita Devi  Age: 35 Y / Female  Wt: 54 kg", "confidence": 90.0},
            {"text": "Combiflam 1-0-1 x 3 days", "confidence": 90.0},
            {"text": "Warm water salt gargles", "confidence": 90.0},
            {"text": "Review after 3 days", "confidence": 90.0}
        ]
        clinical_parser.parse_prescription(lines)
        audit = clinical_parser.categorized_lines
        self.assertEqual(len(audit), 4, "Every input line must be logged in categorization audit")


if __name__ == "__main__":
    unittest.main()
