"""Unit tests for Authentic Smile Care Dental Hospital Prescription Recognition.
Validates extraction and clinical parsing of:
1. Cap Novamox 500mg x 3 days
2. Tab Ketorol-DT x 3 days
Ensures strict differentiation between medicines, instructions, patient demographics, and prescriber header.
"""

import unittest
from PIL import Image
import os
from medagent.ocr_engine import ocr_engine
from medagent.clinical_parser import clinical_parser
from medagent.safety_checker import safety_checker

IMAGE_PATH = r"C:\Users\Lenovo\.gemini\antigravity\brain\4fb41c2b-33e7-4771-b4f9-6c51a5d72955\.user_uploaded\media_1791738363155.jpg"


class TestSmileCarePrescriptionDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image_exists = os.path.exists(IMAGE_PATH)
        if cls.image_exists:
            cls.img = Image.open(IMAGE_PATH)
            cls.doc_analysis = ocr_engine.extract_prescription_document(cls.img)
            cls.orders = clinical_parser.parse_prescription(cls.doc_analysis["med_lines"])
            cls.advice = clinical_parser.clinical_advice
            cls.safety_eval = safety_checker.evaluate_prescription(cls.orders)

    def test_medication_count(self):
        """Must detect exactly 2 medication orders, with NO fake patient or duration noise."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        self.assertEqual(len(self.orders), 2, f"Expected 2 medication orders, got {len(self.orders)}")

    def test_novamox_capsule_extraction(self):
        """Cap. Novamox 500mg: Amoxicillin 500mg Capsule, 3 days duration."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        nova = self.orders[0]
        self.assertEqual(nova["validated_drug"], "Amoxicillin")
        self.assertIn("Novamox", nova["corrected_name"])
        self.assertEqual(nova["dosage_form"], "Capsule")
        self.assertEqual(nova["strength"], "500mg")
        self.assertEqual(nova["duration"], "3 days")
        self.assertEqual(nova["therapeutic_class"], "Aminopenicillin Antibiotic")

    def test_ketorol_dt_extraction(self):
        """Tab. Ketorol-DT: Ketorolac Tromethamine 10mg Dispersible Tablet, 3 days duration."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        keto = self.orders[1]
        self.assertEqual(keto["validated_drug"], "Ketorolac Tromethamine")
        self.assertIn("Ketorol", keto["corrected_name"])
        self.assertEqual(keto["dosage_form"], "Dispersible Tablet")
        self.assertEqual(keto["strength"], "10mg")
        self.assertEqual(keto["duration"], "3 days")
        self.assertIn("NSAID", keto["therapeutic_class"])

    def test_header_and_patient_demographics_separation(self):
        """Prescriber clinic and patient details must be separated and NEVER classified as medicines."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        # Check patient info contains Prasao
        self.assertIn("Prasao", self.doc_analysis["patient_info"])
        # Check prescriber header contains Smile Care
        self.assertIn("SMILE", self.doc_analysis["doctor_info"])
        
        # Verify no order has drug name matching patient name 'Prasao'
        for o in self.orders:
            self.assertNotIn("Prasao", o.get("validated_drug", ""))
            self.assertNotIn("Prasao", o.get("corrected_name", ""))

    def test_user_transcribed_input_variant(self):
        """Validates user text: 'Cap novamax 500 mg 3 days' and 'tab. ketorel DT 3 days'."""
        lines = [
            {"text": "Cap novamax 500 mg 3 days"},
            {"text": "tab. ketorel DT 3 days"}
        ]
        orders = clinical_parser.parse_prescription(lines)
        self.assertEqual(len(orders), 2)
        
        # Order 1: Novamax 500mg
        self.assertEqual(orders[0]["validated_drug"], "Amoxicillin")
        self.assertIn("Novamox", orders[0]["corrected_name"])
        self.assertEqual(orders[0]["dosage_form"], "Capsule")
        self.assertEqual(orders[0]["strength"], "500mg")
        self.assertEqual(orders[0]["duration"], "3 days")
        
        # Order 2: Ketorel DT
        self.assertEqual(orders[1]["validated_drug"], "Ketorolac Tromethamine")
        self.assertIn("Ketorol", orders[1]["corrected_name"])
        self.assertEqual(orders[1]["dosage_form"], "Dispersible Tablet")
        self.assertEqual(orders[1]["strength"], "10mg")
        self.assertEqual(orders[1]["duration"], "3 days")

    def test_raw_ocr_noisy_tokens_variant(self):
        """Validates raw OCR noisy strings: 'ap. omn S00wg', 'Tolo.ketorol', 'x30an'."""
        raw_tokens = ["ap. omn S00wg", "Tolo.ketorol", "x30an"]
        cleaned_tokens = [{"text": ocr_engine.post_process_handwriting(t)} for t in raw_tokens]
        orders = clinical_parser.parse_prescription(cleaned_tokens)
        self.assertEqual(len(orders), 2)
        
        self.assertEqual(orders[0]["validated_drug"], "Amoxicillin")
        self.assertEqual(orders[0]["dosage_form"], "Capsule")
        self.assertEqual(orders[0]["strength"], "500mg")
        self.assertEqual(orders[0]["duration"], "3 days")

        self.assertEqual(orders[1]["validated_drug"], "Ketorolac Tromethamine")
        self.assertEqual(orders[1]["dosage_form"], "Dispersible Tablet")
        self.assertEqual(orders[1]["strength"], "10mg")
        self.assertEqual(orders[1]["duration"], "3 days")


if __name__ == "__main__":
    unittest.main()
