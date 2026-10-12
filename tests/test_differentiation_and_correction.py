"""Unit tests for Medicine vs Instructions Differentiation and Close-Match Correction.
"""

import unittest
from medagent.clinical_parser import clinical_parser
from medagent.drug_db import drug_db


class TestDifferentiationAndCorrection(unittest.TestCase):

    def test_differentiate_medicine_and_instructions_oral(self):
        line = "Tab Amox 500mg 1 cap TID x 7d PO pc"
        parsed = clinical_parser.parse_line(line)
        
        # Verify Medicine Entity
        med = parsed["medicine_entity"]
        self.assertEqual(med["dosage_form"], "Tablet")
        self.assertEqual(med["strength"], "500mg")
        self.assertEqual(med["validated_drug"], "Amoxicillin")
        self.assertIn("Amox", med["corrected_name"])
        
        # Verify Instructions Entity
        inst = parsed["instructions_entity"]
        self.assertIn("Three times daily", inst["frequency"])
        self.assertEqual(inst["duration"], "7 days")
        self.assertEqual(inst["route"], "Oral")
        self.assertEqual(inst["timing"], "After Meals (pc)")

    def test_differentiate_medicine_and_instructions_topical(self):
        line = "Ebernet Cream for L/A Morn"
        parsed = clinical_parser.parse_line(line)
        
        # Verify Medicine Entity
        med = parsed["medicine_entity"]
        self.assertEqual(med["dosage_form"], "Cream")
        self.assertEqual(med["validated_drug"], "Eberconazole")
        
        # Verify Instructions Entity
        inst = parsed["instructions_entity"]
        self.assertEqual(inst["route"], "Topical")
        self.assertEqual(inst["timing"], "Morning")
        self.assertIn("Local Application (Topical)", inst["regimen_summary"])

    def test_differentiate_medicated_bath(self):
        line = "Nizoclin Soap for topical bath"
        parsed = clinical_parser.parse_line(line)
        
        med = parsed["medicine_entity"]
        self.assertEqual(med["dosage_form"], "Medicated Soap")
        self.assertEqual(med["validated_drug"], "Ketoconazole")
        
        inst = parsed["instructions_entity"]
        self.assertEqual(inst["route"], "Topical")
        self.assertIn("Topical bath / Medicated wash", inst["regimen_summary"])

    def test_smart_instruction_fragment_linking(self):
        raw_lines = [
            {"text": "OPC -Doclowel", "confidence": 75.0},
            {"text": "x 10day", "confidence": 70.0}
        ]
        orders = clinical_parser.parse_prescription(raw_lines)
        
        # Must merge into a single order rather than creating a fake 'x' drug
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]["validated_drug"], "Itraconazole")
        self.assertEqual(orders[0]["duration"], "10 days")
        self.assertEqual(orders[0]["instructions_entity"]["duration"], "10 days")

    def test_close_match_corrector(self):
        test_queries = {
            "Itaspr": "Itraconazole",
            "Lanofa": "Terbinafine",
            "Ebrnet": "Eberconazole",
            "Nizocn": "Ketoconazole",
            "Lejt": "Levocetirizine",
            "Safcet": "Levocetirizine",
            "DoIo": "Paracetamol",
            "Augmntin": "Amoxicillin + Clavulanic Acid"
        }
        
        for q, expected_generic in test_queries.items():
            close = drug_db.find_close_matches(q, top_k=3)
            self.assertGreater(len(close), 0, f"Failed to find close matches for '{q}'")
            best = close[0]
            self.assertEqual(best["generic_name"], expected_generic,
                             f"Query '{q}' best match was {best['generic_name']}, expected {expected_generic}")
            self.assertGreaterEqual(best["confidence"], 70.0)

    def test_shorthand_frequency_and_duration_expansions(self):
        """Verifies expanded 4-slot timing patterns, alternate-day frequencies, and duration ranges."""
        from medagent.abbreviation_expander import expand_frequency, expand_duration

        # 4-slot frequencies
        desc1, m1 = expand_frequency("1-0-0-1")
        self.assertIn("Bedtime", desc1)
        self.assertEqual(m1, 2)

        desc2, m2 = expand_frequency("1-0-1-0")
        self.assertIn("Morning & Night", desc2)
        self.assertEqual(m2, 2)

        desc3, m3 = expand_frequency("1-1-1-1")
        self.assertIn("Four times daily", desc3)
        self.assertEqual(m3, 4)

        # Alternate days
        desc4, m4 = expand_frequency("Tab QOD")
        self.assertEqual(desc4, "Alternate days")
        desc5, m5 = expand_frequency("alt die")
        self.assertEqual(desc5, "Alternate days")

        # Duration shorthand and ranges
        self.assertEqual(expand_duration("x 3/7"), "3 days")
        self.assertEqual(expand_duration("for 5/7"), "5 days")
        self.assertEqual(expand_duration("x 2/52"), "2 weeks")
        self.assertEqual(expand_duration("x 1/12"), "1 month(s)")
        self.assertEqual(expand_duration("3-5 days"), "3-5 days")


if __name__ == '__main__':
    unittest.main()
