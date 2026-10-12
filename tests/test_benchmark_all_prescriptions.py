"""Benchmark Test Suite for Authentic Clinical Prescriptions.
Validates end-to-end OCR extraction, layout analysis, clinical parsing,
and medicine vs instructions differentiation across 4 authentic doctor prescriptions.
"""

import unittest
import os
from PIL import Image
from medagent.ocr_engine import ocr_engine
from medagent.clinical_parser import clinical_parser

BASE_DIR = r"C:\Users\Lenovo\.gemini\antigravity\brain\4fb41c2b-33e7-4771-b4f9-6c51a5d72955\.user_uploaded"


class TestBenchmarkAllPrescriptions(unittest.TestCase):

    def test_smile_care_prescription(self):
        """Prescription 1: Smile Care Dental Hospital (Shyam Prasad).
        Novamox 500mg Capsule + Ketorol-DT Dispersible Tablet x 3 days.
        """
        p = os.path.join(BASE_DIR, "media_1791748128336.jpg")
        if not os.path.exists(p):
            self.skipTest(f"Image not found at {p}")

        img = Image.open(p)
        doc = ocr_engine.extract_prescription_document(img)
        orders = clinical_parser.parse_prescription(doc["med_lines"])

        self.assertEqual(len(orders), 2, f"Expected 2 medication orders, got {len(orders)}")
        
        # Prescriber & Patient differentiation
        self.assertIn("Smile Care", doc["prescriber_details"]["clinic_name"])
        self.assertEqual(doc["patient_details"]["patient_name"], "Shyam Prasad")

        # Order 1: Novamox 500mg
        nov = orders[0]
        self.assertEqual(nov["validated_drug"], "Amoxicillin")
        self.assertIn("Novamox", nov["corrected_name"])
        self.assertEqual(nov["dosage_form"], "Capsule")
        self.assertEqual(nov["strength"], "500mg")
        self.assertEqual(nov["duration"], "3 days")

        # Order 2: Ketorol-DT
        ket = orders[1]
        self.assertEqual(ket["validated_drug"], "Ketorolac Tromethamine")
        self.assertIn("Ketorol", ket["corrected_name"])
        self.assertEqual(ket["dosage_form"], "Dispersible Tablet")
        self.assertEqual(ket["strength"], "10mg")
        self.assertEqual(ket["duration"], "3 days")

    def test_mishra_dental_prescription(self):
        """Prescription 2: Mishra Dental Clinic (Raj Chaurasiya).
        Augmentin 625mg, Enzoflam, Pan-D 40mg, Hexigel Gum Paint, Rest for 4 days.
        """
        p = os.path.join(BASE_DIR, "media_1791748139656.jpg")
        if not os.path.exists(p):
            self.skipTest(f"Image not found at {p}")

        img = Image.open(p)
        doc = ocr_engine.extract_prescription_document(img)
        orders = clinical_parser.parse_prescription(doc["med_lines"])

        self.assertEqual(len(orders), 4, f"Expected 4 medication orders, got {len(orders)}")

        # Prescriber & Patient differentiation
        self.assertEqual(doc["prescriber_details"]["doctor_name"], "Dr. Ashutosh Mishra")
        self.assertEqual(doc["prescriber_details"]["clinic_name"], "Mishra Dental Clinic")
        self.assertEqual(doc["patient_details"]["date"], "31/05/25")

        # Augmentin
        aug = orders[0]
        self.assertEqual(aug["validated_drug"], "Amoxicillin + Clavulanic Acid")
        self.assertIn("Augmentin", aug["corrected_name"])
        self.assertEqual(aug["strength"], "625mg")
        self.assertEqual(aug["duration"], "5 days")
        self.assertEqual(aug["instructions_entity"]["timing"], "After Meals (pc)")

        # Enzoflam
        enzo = orders[1]
        self.assertEqual(enzo["validated_drug"], "Diclofenac + Paracetamol + Serratiopeptidase")
        self.assertEqual(enzo["corrected_name"], "Enzoflam")
        self.assertEqual(enzo["duration"], "5 days")
        self.assertEqual(enzo["instructions_entity"]["timing"], "After Meals (pc)")

        # Pan-D 40mg
        pand = orders[2]
        self.assertEqual(pand["validated_drug"], "Pantoprazole + Domperidone")
        self.assertIn("Pan-D", pand["corrected_name"])
        self.assertEqual(pand["strength"], "40mg")
        self.assertEqual(pand["duration"], "5 days")
        self.assertEqual(pand["instructions_entity"]["timing"], "Before Meals (ac / Empty stomach)")

        # Hexigel Gum Paint
        hexi = orders[3]
        self.assertEqual(hexi["validated_drug"], "Chlorhexidine Gluconate")
        self.assertEqual(hexi["dosage_form"], "Gum Paint / Oral Gel")
        self.assertEqual(hexi["duration"], "1 week")

        # Clinical Advice
        self.assertGreaterEqual(len(clinical_parser.clinical_advice), 1)
        self.assertIn("Rest for 4 days", clinical_parser.clinical_advice[0]["advice_text"])

    def test_dr_gopinath_dermatology_prescription(self):
        """Prescription 3: Dr. M. Gopinath MD(DVL) (Suresh, 19M, P. Versicolor).
        Itaspor 200mg, Ebernet Cream, Lamifin Cream, Lejet 5mg.
        """
        p = os.path.join(BASE_DIR, "media_1791748139704.jpg")
        if not os.path.exists(p):
            self.skipTest(f"Image not found at {p}")

        img = Image.open(p)
        doc = ocr_engine.extract_prescription_document(img)
        orders = clinical_parser.parse_prescription(doc["med_lines"])

        self.assertGreaterEqual(len(orders), 4, f"Expected at least 4 medication orders, got {len(orders)}")

        # Prescriber & Patient differentiation
        self.assertEqual(doc["prescriber_details"]["doctor_name"], "Dr. M. Gopinath")
        self.assertEqual(doc["patient_details"]["patient_name"], "Suresh")
        self.assertEqual(doc["patient_details"]["age_gender"], "19 Y / Male")
        self.assertIn("Versicolor", doc["patient_details"]["diagnosis"])

        # Verify Itaspor 200mg
        itaspor_order = next((o for o in orders if "Itraconazole" in o["validated_drug"]), None)
        self.assertIsNotNone(itaspor_order)
        self.assertEqual(itaspor_order["dosage_form"], "Capsule")
        self.assertEqual(itaspor_order["strength"], "200mg")

        # Verify Ebernet Cream
        eber_order = next((o for o in orders if "Eberconazole" in o["validated_drug"]), None)
        self.assertIsNotNone(eber_order)
        self.assertEqual(eber_order["dosage_form"], "Cream")
        self.assertEqual(eber_order["instructions_entity"]["route"], "Topical")

        # Verify Lamifin Cream
        lami_order = next((o for o in orders if "Terbinafine" in o["validated_drug"]), None)
        self.assertIsNotNone(lami_order)
        self.assertEqual(lami_order["dosage_form"], "Cream")
        self.assertEqual(lami_order["instructions_entity"]["route"], "Topical")

        # Verify Lejet 5mg
        lejet_order = next((o for o in orders if "Levocetirizine" in o["validated_drug"]), None)
        self.assertIsNotNone(lejet_order)
        self.assertEqual(lejet_order["dosage_form"], "Tablet")
        self.assertEqual(lejet_order["strength"], "5mg")

    def test_dr_roy_thomas_neurology_prescription(self):
        """Prescription 4: Dr. Roy Thomas MD DM Neuro (Beena George, 48F).
        10 Medications: Losar H, Losar 25, Amlodac 5, Ator 10, Dicorate ER 250,
        Localm PLUS, Lonezep 1mg, Glucored Forte, Glycomet SR 500, Pantop 40.
        """
        p = os.path.join(BASE_DIR, "media_1791748139705.jpg")
        if not os.path.exists(p):
            self.skipTest(f"Image not found at {p}")

        img = Image.open(p)
        doc = ocr_engine.extract_prescription_document(img)
        orders = clinical_parser.parse_prescription(doc["med_lines"])

        self.assertEqual(len(orders), 10, f"Expected exactly 10 neurology orders, got {len(orders)}")

        # Prescriber & Patient differentiation
        self.assertEqual(doc["prescriber_details"]["doctor_name"], "Dr. Roy Thomas")
        self.assertEqual(doc["prescriber_details"]["reg_no"], "31219")
        self.assertEqual(doc["patient_details"]["patient_name"], "Beena George")

        expected_drugs = [
            ("Losartan + Hydrochlorothiazide", "Losar H", "50mg/12.5mg"),
            ("Losartan", "Losar", "25mg"),
            ("Amlodipine", "Amlodac", "5mg"),
            ("Atorvastatin", "Ator 10", "10mg"),
            ("Divalproex Sodium", "Dicorate ER", "250mg"),
            ("Escitalopram + Clonazepam", "Localm PLUS", "10mg/0.5mg"),
            ("Clonazepam", "Lonezep 1mg", "1mg"),
            ("Metformin + Glibenclamide", "Glucored Forte", "Forte (500mg/5mg)"),
            ("Metformin", "Glycomet SR 500", "500mg"),
            ("Pantoprazole", "Pantop 40", "40mg")
        ]

        for idx, (exp_generic, exp_brand, exp_strength) in enumerate(expected_drugs):
            actual = orders[idx]
            self.assertEqual(actual["validated_drug"], exp_generic,
                             f"Order {idx+1}: expected generic {exp_generic}, got {actual['validated_drug']}")
            self.assertIn(exp_brand.split()[0], actual["corrected_name"],
                          f"Order {idx+1}: expected brand {exp_brand}, got {actual['corrected_name']}")
            self.assertEqual(actual["strength"], exp_strength,
                             f"Order {idx+1}: expected strength {exp_strength}, got {actual['strength']}")

    def test_dr_nithin_paediatrics_prescription(self):
        """Prescription 5: Dr. Nithin Narayanan MBBS MD Paediatrics JIPMER (Ashvika, 4F, 13.25kg, URTI).
        4 Paediatric Orders: Calpol 250 (250/5) 4ml, Delcon 3ml, Levolin 3ml, Meftal-P (100/5) 3ml.
        Zero vitals (Weight, RR, RS, Clinical Description) or Malayalam noise counted as medicine.
        """
        p = os.path.join(BASE_DIR, "media_1791755287453.jpg")
        if not os.path.exists(p):
            self.skipTest(f"Image not found at {p}")

        img = Image.open(p)
        doc = ocr_engine.extract_prescription_document(img)
        orders = clinical_parser.parse_prescription(doc["med_lines"])

        # Prescriber & Patient differentiation
        self.assertEqual(doc["prescriber_details"]["doctor_name"], "Dr. Nithin Narayanan")
        self.assertEqual(doc["patient_details"]["patient_name"], "Ashvika")
        self.assertEqual(doc["patient_details"]["age_gender"], "4 Y / Female")
        self.assertEqual(doc["patient_details"]["weight"], "13.25 kg")
        self.assertIn("URTI", doc["patient_details"]["diagnosis"])

        # Exactly 4 medication orders
        self.assertEqual(len(orders), 4, f"Expected exactly 4 paediatric orders, got {len(orders)}")

        # Order 1: Calpol 250
        calpol = orders[0]
        self.assertIn("Paracetamol", calpol["validated_drug"])
        self.assertIn("Calpol", calpol["corrected_name"])
        self.assertEqual(calpol["dosage_form"], "Syrup")
        self.assertIn("250mg/5ml", calpol["strength"])
        self.assertEqual(calpol["frequency"], "Every 6 hours")
        self.assertEqual(calpol["duration"], "3 days")

        # Order 2: Delcon Syrup
        delcon = orders[1]
        self.assertIn("Phenylephrine", delcon["validated_drug"])
        self.assertIn("Delcon", delcon["corrected_name"])
        self.assertEqual(delcon["dosage_form"], "Syrup")
        self.assertEqual(delcon["strength"], "3ml")
        self.assertEqual(delcon["frequency"], "Three times daily")
        self.assertEqual(delcon["duration"], "5 days")

        # Order 3: Levolin Syrup
        levolin = orders[2]
        self.assertEqual(levolin["validated_drug"], "Levosalbutamol")
        self.assertIn("Levolin", levolin["corrected_name"])
        self.assertEqual(levolin["dosage_form"], "Syrup")
        self.assertEqual(levolin["strength"], "3ml")
        self.assertEqual(levolin["frequency"], "Three times daily")
        self.assertEqual(levolin["duration"], "5 days")

        # Order 4: Meftal-P Suspension
        meftal = orders[3]
        self.assertEqual(meftal["validated_drug"], "Mefenamic Acid")
        self.assertIn("Meftal-P", meftal["corrected_name"])
        self.assertIn("Suspension", meftal["dosage_form"])
        self.assertIn("100mg/5ml", meftal["strength"])
        self.assertIn("SOS", meftal["frequency"])

        # Verify no non-medication lines were added as orders
        for o in orders:
            self.assertFalse(o["validated_drug"].startswith("Unverified (Weight)"), "Weight must not be an order")
            self.assertFalse("Pge Cender" in o["validated_drug"], "Pge Cender must not be an order")
            self.assertFalse("Clinical" in o["validated_drug"], "Clinical notes must not be an order")


if __name__ == "__main__":
    unittest.main()
