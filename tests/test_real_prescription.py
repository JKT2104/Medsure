"""Unit tests for real clinic prescription (Dr. Gopinath - Pityriasis Versicolor).
"""

import unittest
from medagent.clinical_parser import clinical_parser
from medagent.safety_checker import safety_checker


class TestRealPrescription(unittest.TestCase):

    def setUp(self):
        self.lines = [
            "Cap. Itaspor 200mg 1-0-0 x 10 days",
            "Ebernet Cream for L/A Morn",
            "Lamifin Cream for L/A Night",
            "Nizoclin Soap for topical bath",
            "Tab. Lejet / Safecet 5mg 0-0-1 x 15 days"
        ]

    def test_parse_real_prescription_lines(self):
        orders = [clinical_parser.parse_line(l) for l in self.lines]

        # 1. Itaspor 200mg -> Itraconazole
        self.assertEqual(orders[0]["validated_drug"], "Itraconazole")
        self.assertEqual(orders[0]["strength"], "200mg")
        self.assertEqual(orders[0]["dosage_form"], "Capsule")
        self.assertIn("Once daily", orders[0]["frequency"])
        self.assertEqual(orders[0]["duration"], "10 days")

        # 2. Ebernet Cream -> Eberconazole
        self.assertEqual(orders[1]["validated_drug"], "Eberconazole")
        self.assertEqual(orders[1]["dosage_form"], "Cream")
        self.assertIn("Topical", orders[1]["instructions"])
        self.assertIn("morning", orders[1]["instructions"])

        # 3. Lamifin Cream -> Terbinafine
        self.assertEqual(orders[2]["validated_drug"], "Terbinafine")
        self.assertEqual(orders[2]["dosage_form"], "Cream")
        self.assertIn("Topical", orders[2]["instructions"])
        self.assertIn("night", orders[2]["instructions"])

        # 4. Nizoclin Soap -> Ketoconazole
        self.assertEqual(orders[3]["validated_drug"], "Ketoconazole")
        self.assertEqual(orders[3]["dosage_form"], "Medicated Soap")

        # 5. Tab. Lejet / Safecet 5mg -> Levocetirizine
        self.assertEqual(orders[4]["validated_drug"], "Levocetirizine")
        self.assertEqual(orders[4]["strength"], "5mg")
        self.assertIn("Night / Bedtime", orders[4]["frequency"])
        self.assertEqual(orders[4]["duration"], "15 days")

    def test_antifungal_safety_evaluation(self):
        orders = [clinical_parser.parse_line(l) for l in self.lines]
        eval_result = safety_checker.evaluate_prescription(orders)
        # Accurately detects LASA warning for Lamifin/Lamisil (Terbinafine) vs Lamictal
        self.assertEqual(eval_result["overall_status"], "CRITICAL")
        self.assertTrue(any(a["category"] == "LASA_WARNING" for a in eval_result["alerts"]))


if __name__ == '__main__':
    unittest.main()
