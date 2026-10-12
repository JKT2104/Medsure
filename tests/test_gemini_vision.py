"""Unit tests for Gemini Multimodal Vision Client."""

import unittest
import os
from PIL import Image
from medagent.gemini_vision import gemini_vision, GeminiVisionClient


class TestGeminiVision(unittest.TestCase):

    def test_client_initialization_and_key_loading(self):
        """Verifies that GeminiVisionClient loads API key from .env or environment."""
        self.assertTrue(gemini_vision.is_available(), "Gemini client should detect loaded API key")
        self.assertGreater(len(gemini_vision.api_key), 20)

    def test_key_setter(self):
        """Verifies setting API key dynamically."""
        client = GeminiVisionClient(api_key="test_dummy_key_1234567890")
        self.assertEqual(client.api_key, "test_dummy_key_1234567890")
        self.assertTrue(client.is_available())

    def test_mock_format_response(self):
        """Verifies formatting of structured JSON into MedAgent document schema."""
        client = GeminiVisionClient(api_key="test")
        sample_raw = {
            "prescriber_details": {
                "doctor_name": "Dr. Test MD",
                "clinic_name": "Test Clinic",
                "qualifications": "MBBS",
                "reg_no": "12345"
            },
            "patient_details": {
                "patient_name": "John Doe",
                "age_gender": "30M",
                "weight": "70 kg",
                "date": "01/01/2026",
                "diagnosis": "Fever"
            },
            "footer_info": "Contact: 9999999999",
            "med_lines": [
                {"line_idx": 1, "text": "Tab Paracetamol 500mg 1-0-1 x 3d"}
            ],
            "clinical_advice": ["Drink warm water"]
        }
        dummy_img = Image.new("RGB", (200, 200), color="white")
        doc = client._format_response(sample_raw, "gemini-3.8-flash", dummy_img)

        self.assertEqual(doc["prescriber_details"]["doctor_name"], "Dr. Test MD")
        self.assertEqual(doc["patient_details"]["weight"], "70 kg")
        self.assertEqual(len(doc["med_lines"]), 1)
        self.assertEqual(doc["med_lines"][0]["text"], "Tab Paracetamol 500mg 1-0-1 x 3d")
        self.assertEqual(doc["clinical_advice"], ["Drink warm water"])


if __name__ == "__main__":
    unittest.main()
