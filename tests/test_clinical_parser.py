"""Unit tests for clinical entity parser and doctor shorthand decoding.
"""

import unittest
from medagent.clinical_parser import clinical_parser
from medagent.abbreviation_expander import expand_frequency, expand_duration, expand_instructions


class TestClinicalParser(unittest.TestCase):

    def test_abbreviation_frequency(self):
        desc, mult = expand_frequency("1-0-1")
        self.assertEqual(mult, 2)
        self.assertIn("Twice daily", desc)

        desc, mult = expand_frequency("TID")
        self.assertEqual(mult, 3)
        self.assertIn("Three times daily", desc)

        desc, mult = expand_frequency("OD")
        self.assertEqual(mult, 1)

        desc, mult = expand_frequency("SOS")
        self.assertIn("As needed", desc)

    def test_quirky_duration_formats(self):
        self.assertEqual(expand_duration("x 5/7"), "5 days")
        self.assertEqual(expand_duration("x 3/7"), "3 days")
        self.assertEqual(expand_duration("x 2/52"), "2 weeks")
        self.assertEqual(expand_duration("x 1/12"), "1 month(s)")
        self.assertEqual(expand_duration("x 10d"), "10 days")
        self.assertEqual(expand_duration("for 14 days"), "14 days")

    def test_clinical_instructions(self):
        inst = expand_instructions("take 1 tab po pc")
        self.assertIn("After meals (Post Cibum)", inst)
        self.assertIn("Oral (Per Os)", inst)

        inst_ac = expand_instructions("cap omez 20 ac")
        self.assertIn("Before meals (Ante Cibum)", inst_ac)

    def test_line_parsing_antibiotic(self):
        line = "1. Tab Amox 500mg 1 cap TID x 7d PO pc"
        result = clinical_parser.parse_line(line, ocr_confidence=92.0)

        self.assertEqual(result["validated_drug"], "Amoxicillin")
        self.assertEqual(result["strength"], "500mg")
        self.assertEqual(result["daily_multiplier"], 3)
        self.assertEqual(result["daily_dose_mg"], 1500.0)
        self.assertEqual(result["duration"], "7 days")
        self.assertIn("After meals", result["instructions"])

    def test_line_parsing_standalone_dose(self):
        line = "Rx: Dolo 650 1-0-1 x 5d SOS"
        result = clinical_parser.parse_line(line, ocr_confidence=88.0)

        self.assertIn("Paracetamol", result["validated_drug"])
        self.assertEqual(result["strength"], "650mg")
        self.assertEqual(result["daily_multiplier"], 2)
        self.assertEqual(result["daily_dose_mg"], 1300.0)


if __name__ == '__main__':
    unittest.main()
