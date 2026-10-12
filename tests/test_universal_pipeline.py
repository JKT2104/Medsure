"""Unit tests for Universal Prescription Layout Extraction and Recognition.
"""

import unittest
import os
from PIL import Image
from medagent.ocr_engine import ocr_engine
from medagent.clinical_parser import clinical_parser
from medagent.safety_checker import safety_checker


class TestUniversalPipeline(unittest.TestCase):

    def test_extract_prescription_document_sample1(self):
        sample_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "sample_1.png")
        if os.path.exists(sample_path):
            img = Image.open(sample_path)
            doc_data = ocr_engine.extract_prescription_document(img, use_neural_trocr=False)
            
            self.assertIn("doctor_info", doc_data)
            self.assertIn("patient_info", doc_data)
            self.assertIn("med_lines", doc_data)
            self.assertGreater(len(doc_data["med_lines"]), 0)
            
            # Check clinical parsing of extracted lines
            orders = clinical_parser.parse_prescription(doc_data["med_lines"])
            self.assertGreater(len(orders), 0)
            
            # Verify safety evaluation completes without error
            safety_eval = safety_checker.evaluate_prescription(orders)
            self.assertIn(safety_eval["overall_status"], ["SAFE", "WARNING", "CRITICAL"])

    def test_stopword_non_drug_rejection(self):
        from medagent.drug_db import drug_db
        # Ensure common instruction words and prepositions are NEVER matched as drugs
        self.assertIsNone(drug_db.find_match("for"))
        self.assertIsNone(drug_db.find_match("morn"))
        self.assertIsNone(drug_db.find_match("night"))
        self.assertIsNone(drug_db.find_match("soap"))
        self.assertIsNone(drug_db.find_match("cream"))
        self.assertIsNone(drug_db.find_match("sundayclosed"))
        self.assertIsNone(drug_db.find_match("to"))
        self.assertIsNone(drug_db.find_match("and"))

    def test_arbitrary_drug_recognition_and_parsing(self):
        # Arbitrary clinical order lines
        lines = [
            {"line_idx": 1, "text": "Tab Metformin 500mg 1-0-1 x 30d pc", "confidence": 92.0},
            {"line_idx": 2, "text": "Tab Telma 40mg 1-0-0 x 30d", "confidence": 90.0},
            {"line_idx": 3, "text": "Tab Atorva 10mg 0-0-1 x 30d hs", "confidence": 91.0}
        ]
        orders = clinical_parser.parse_prescription(lines)
        self.assertEqual(len(orders), 3)
        self.assertEqual(orders[0]["validated_drug"], "Metformin")
        self.assertEqual(orders[1]["validated_drug"], "Telmisartan")
        self.assertEqual(orders[2]["validated_drug"], "Atorvastatin")


if __name__ == '__main__':
    unittest.main()
