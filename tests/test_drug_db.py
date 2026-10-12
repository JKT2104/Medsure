"""Unit tests for drug database fuzzy matching and phonetic resolution.
"""

import unittest
from medagent.drug_db import drug_db, soundex


class TestDrugDatabase(unittest.TestCase):

    def test_exact_match(self):
        match = drug_db.find_match("Amoxicillin")
        self.assertIsNotNone(match)
        self.assertEqual(match["generic_name"], "Amoxicillin")
        self.assertEqual(match["confidence"], 100.0)

    def test_popular_brand_name_resolution(self):
        match_dolo = drug_db.find_match("Dolo")
        self.assertIsNotNone(match_dolo)
        self.assertEqual(match_dolo["generic_name"], "Paracetamol")

        match_augmentin = drug_db.find_match("Augmentin")
        self.assertIsNotNone(match_augmentin)
        self.assertEqual(match_augmentin["generic_name"], "Amoxicillin + Clavulanic Acid")

        match_pan = drug_db.find_match("Pantocid")
        self.assertIsNotNone(match_pan)
        self.assertEqual(match_pan["generic_name"], "Pantoprazole")

    def test_doctor_quirky_cursive_misspellings(self):
        # OCR reads "Paracefamol" instead of "Paracetamol"
        match = drug_db.find_match("Paracefamol")
        self.assertIsNotNone(match)
        self.assertEqual(match["generic_name"], "Paracetamol")
        self.assertGreater(match["confidence"], 75.0)

        # OCR reads "Metformn" instead of "Metformin"
        match_met = drug_db.find_match("Metformn")
        self.assertIsNotNone(match_met)
        self.assertEqual(match_met["generic_name"], "Metformin")

        # Cursive abbreviation "Amox"
        match_amox = drug_db.find_match("Amox")
        self.assertIsNotNone(match_amox)
        self.assertEqual(match_amox["generic_name"], "Amoxicillin")

    def test_soundex_encoding(self):
        code1 = soundex("Paracetamol")
        code2 = soundex("Paracetamal")
        self.assertEqual(code1, code2)


if __name__ == '__main__':
    unittest.main()
