"""Unit tests for Authentic Dental Prescription Recognition and Clinical Parsing.
Tests Mishra Dental Clinic prescription pad with quirky handwriting,
ensuring strict differentiation between medicines, instructions, and non-drug advice.
"""

import unittest
from PIL import Image
import os
from medagent.ocr_engine import ocr_engine
from medagent.clinical_parser import clinical_parser
from medagent.safety_checker import safety_checker

IMAGE_PATH = r"C:\Users\Lenovo\.gemini\antigravity\brain\4fb41c2b-33e7-4771-b4f9-6c51a5d72955\.user_uploaded\media_1791202840509.jpg"


class TestDentalPrescriptionDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.image_exists = os.path.exists(IMAGE_PATH)
        if cls.image_exists:
            cls.img = Image.open(IMAGE_PATH)
            cls.doc_analysis = ocr_engine.extract_prescription_document(cls.img)
            cls.orders = clinical_parser.parse_prescription(cls.doc_analysis["med_lines"])
            cls.advice = clinical_parser.clinical_advice

    def test_medication_count(self):
        """Must detect exactly 4 real prescribed medications, with NO fake unverified tablet noise."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        self.assertEqual(len(self.orders), 4, f"Expected 4 medication orders, got {len(self.orders)}")

    def test_augmentin_extraction(self):
        """Augmentin 625mg: Amoxicillin + Clavulanic Acid 625mg Tablet, twice daily x 5 days after meals."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        aug = self.orders[0]
        self.assertEqual(aug["validated_drug"], "Amoxicillin + Clavulanic Acid")
        self.assertIn("Augmentin", aug["corrected_name"])
        self.assertEqual(aug["dosage_form"], "Tablet")
        self.assertEqual(aug["strength"], "625mg")
        self.assertIn("Twice daily", aug["frequency"])
        self.assertEqual(aug["duration"], "5 days")
        self.assertEqual(aug["instructions_entity"]["timing"], "After Meals (pc)")

    def test_enzoflam_extraction(self):
        """Enzoflam: Diclofenac + Paracetamol + Serratiopeptidase Tablet, twice daily x 5 days after meals."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        enzo = self.orders[1]
        self.assertEqual(enzo["validated_drug"], "Diclofenac + Paracetamol + Serratiopeptidase")
        self.assertEqual(enzo["corrected_name"], "Enzoflam")
        self.assertEqual(enzo["dosage_form"], "Tablet")
        self.assertIn("Twice daily", enzo["frequency"])
        self.assertEqual(enzo["duration"], "5 days")
        self.assertEqual(enzo["instructions_entity"]["timing"], "After Meals (pc)")

    def test_pan_d_extraction(self):
        """Pan-D 40mg: Pantoprazole + Domperidone 40mg, once daily morning x 5 days before meals."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        pand = self.orders[2]
        self.assertEqual(pand["validated_drug"], "Pantoprazole + Domperidone")
        self.assertIn("Pan-D", pand["corrected_name"])
        self.assertEqual(pand["strength"], "40mg")
        self.assertIn("Once daily", pand["frequency"])
        self.assertEqual(pand["duration"], "5 days")
        self.assertEqual(pand["instructions_entity"]["timing"], "Before Meals (ac / Empty stomach)")

    def test_hexigel_gum_paint_extraction(self):
        """Hexigel Gum Paint: Chlorhexidine Gluconate 1% w/w, Topical Gums, 1 week duration, massage."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        hexi = self.orders[3]
        self.assertEqual(hexi["validated_drug"], "Chlorhexidine Gluconate")
        self.assertIn("Hexigel", hexi["corrected_name"])
        self.assertEqual(hexi["dosage_form"], "Gum Paint / Oral Gel")
        self.assertEqual(hexi["strength"], "1% w/w")
        self.assertEqual(hexi["duration"], "1 week")
        self.assertIn("Massage", hexi["instructions"])

    def test_clinical_advice_separation(self):
        """Clinical advice ('Rest for 4 days') must be separated and NEVER classified as a medicine."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        self.assertGreaterEqual(len(self.advice), 1)
        adv_texts = [a["advice_text"] for a in self.advice]
        self.assertTrue(any("Rest for 4 days" in t for t in adv_texts))
        
        # Verify no order has drug name containing 'Rest' or 'Rtpe'
        for o in self.orders:
            self.assertNotIn("Rtpe", o["raw_text"])
            self.assertNotIn("Rest", o["validated_drug"])

    def test_no_unverified_tablets(self):
        """Verify that massage, XIweek, and clinic footer are never treated as unverified tablets."""
        if not self.image_exists:
            self.skipTest("Prescription image not found at path")
        for o in self.orders:
            self.assertFalse(o["validated_drug"].startswith("Unverified"),
                             f"Found unverified medication order: {o['validated_drug']}")
            self.assertNotIn("massage", o["validated_drug"].lower())
            self.assertNotIn("ashutosh", o["validated_drug"].lower())
            self.assertNotIn("doxycycline", o["validated_drug"].lower())
            self.assertNotIn("telmisartan", o["validated_drug"].lower())


if __name__ == '__main__':
    unittest.main()
