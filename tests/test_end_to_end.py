"""End-to-end integration test for MedAgent pipeline.
Tests prescription image generation, line segmentation, clinical parsing,
fuzzy formulary resolution, and safety decision support.
"""

import unittest
import os
from PIL import Image
from medagent.samples import SAMPLE_PRESCRIPTIONS, generate_prescription_image
from medagent.image_preprocessor import image_preprocessor
from medagent.clinical_parser import clinical_parser
from medagent.safety_checker import safety_checker


class TestMedAgentEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.join("tests", "fixtures")
        os.makedirs(cls.test_dir, exist_ok=True)
        cls.sample = SAMPLE_PRESCRIPTIONS[2]  # Warfarin + Brufen (Critical DDI)
        cls.img_path = os.path.join(cls.test_dir, "test_prescription.png")
        generate_prescription_image(cls.sample, cls.img_path)

    def test_line_segmentation(self):
        img = Image.open(self.img_path)
        segments = image_preprocessor.segment_lines(img)
        self.assertGreater(len(segments), 0)
        self.assertTrue(all("crop" in s and "box" in s for s in segments))

    def test_end_to_end_clinical_safety_workflow(self):
        # Ground truth lines from sample 3
        test_lines = [
            {"line_idx": 1, "text": "Tab Warfarin 5mg 1-0-0 x 14d", "confidence": 91.0},
            {"line_idx": 2, "text": "Tab Brufen 400mg 1-0-1 x 5d pc", "confidence": 88.0},
            {"line_idx": 3, "text": "Tab Calpol 500mg 1-0-1 SOS", "confidence": 89.0}
        ]

        parsed = clinical_parser.parse_prescription(test_lines)
        self.assertEqual(len(parsed), 3)

        # Check drug validation
        val_drugs = [p["validated_drug"] for p in parsed]
        self.assertIn("Warfarin", val_drugs)
        self.assertIn("Ibuprofen", val_drugs)
        self.assertIn("Paracetamol", val_drugs)

        # Check safety evaluation
        eval_result = safety_checker.evaluate_prescription(parsed)
        self.assertEqual(eval_result["overall_status"], "CRITICAL")
        self.assertTrue(any(a["category"] == "DRUG_INTERACTION" for a in eval_result["alerts"]))


if __name__ == '__main__':
    unittest.main()
