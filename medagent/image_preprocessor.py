"""Image Preprocessing & Line Segmentation Engine.
Optimized for doctor prescriptions. Handles faint ink, border margins,
ruled lines, and extracts individual line strips for OCR.
"""

from typing import List, Tuple, Dict, Any
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np


class ImagePreprocessor:
    """Prepares prescription images for OCR and extracts individual line crops."""

    def enhance_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhances contrast, sharpens cursive ink strokes, and normalizes brightness."""
        # Convert to Grayscale
        gray = image.convert("L")

        # Auto-contrast with 2% cutoff to remove paper yellowing/shadows
        contrasted = ImageOps.autocontrast(gray, cutoff=2)

        # Enhance contrast further
        enhancer = ImageEnhance.Contrast(contrasted)
        enhanced = enhancer.enhance(1.8)

        # Slight sharpening to recover faint ink edges
        sharpened = enhanced.filter(ImageFilter.SHARPEN)

        return sharpened

    def binarize(self, gray_image: Image.Image) -> np.ndarray:
        """Converts grayscale PIL image to a binary numpy array (1=ink, 0=background)."""
        img_arr = np.array(gray_image)
        # Dynamic Otsu-like thresholding
        threshold = int(np.percentile(img_arr, 25))
        threshold = max(80, min(200, threshold))
        binary = (img_arr < threshold).astype(np.uint8)
        return binary

    def remove_ruled_lines(self, binary: np.ndarray) -> np.ndarray:
        """Erases long horizontal ruled notebook/pad lines so they don't merge text lines."""
        cleaned = binary.copy()
        h, w = cleaned.shape
        # Check rows where ink spans > 45% of the width
        row_sums = np.sum(cleaned, axis=1)
        for y in range(h):
            if row_sums[y] > w * 0.45:
                # If it's a thin line (thickness <= 3 pixels), erase it
                cleaned[max(0, y - 1):min(h, y + 2), :] = 0
        return cleaned

    def segment_lines(self, image: Image.Image, min_line_height: int = 18, padding: int = 8) -> List[Dict[str, Any]]:
        """Segments the prescription into text line strips.
        
        Ignores outer borders and paper edges, removes ruled divider lines,
        and isolates individual medication order lines.
        """
        w, h = image.size
        enhanced = self.enhance_for_ocr(image)
        binary = self.binarize(enhanced)

        # Crop out outer margins (5% left/right, 4% top/bottom) to exclude page borders
        x_margin = int(w * 0.05)
        y_margin = int(h * 0.04)
        inner_bin = binary[y_margin:h - y_margin, x_margin:w - x_margin]

        # Erase ruled notebook lines
        cleaned_inner = self.remove_ruled_lines(inner_bin)

        # Horizontal projection profile on inner content
        profile = np.sum(cleaned_inner, axis=1)

        # Smooth profile with moving average filter
        kernel_size = 9
        if len(profile) > kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(profile, kernel, mode='same')
        else:
            smoothed = profile

        # Dynamic noise cutoff
        noise_threshold = max(smoothed) * 0.06 if len(smoothed) > 0 else 5

        # Detect line start and end points
        raw_lines = []
        in_line = False
        start_y = 0

        for y, val in enumerate(smoothed):
            if val > noise_threshold and not in_line:
                in_line = True
                start_y = y
            elif val <= noise_threshold and in_line:
                in_line = False
                end_y = y
                if (end_y - start_y) >= min_line_height:
                    raw_lines.append((start_y + y_margin, end_y + y_margin))

        if in_line and (len(smoothed) - start_y) >= min_line_height:
            raw_lines.append((start_y + y_margin, len(smoothed) + y_margin))

        # Merge segments that are too close to each other (e.g. < 10 pixels gap)
        merged = []
        i = 0
        while i < len(raw_lines):
            cur_s, cur_e = raw_lines[i]
            while (i + 1 < len(raw_lines)) and (raw_lines[i + 1][0] - cur_e < 12):
                cur_e = raw_lines[i + 1][1]
                i += 1
            merged.append((cur_s, cur_e))
            i += 1

        # Filter out very thin header/divider artifacts (< 12px)
        valid_lines = [seg for seg in merged if (seg[1] - seg[0]) >= min_line_height]

        # Fallback if no lines found
        if not valid_lines:
            # Create 3-4 sensible horizontal strips across the central 70% of the pad
            body_start = int(h * 0.25)
            body_end = int(h * 0.80)
            strip_h = (body_end - body_start) // 3
            for k in range(3):
                valid_lines.append((body_start + k * strip_h, body_start + (k + 1) * strip_h))

        # Build line crop dicts
        line_crops = []
        for idx, (y1, y2) in enumerate(valid_lines):
            pad_y1 = max(0, y1 - padding)
            pad_y2 = min(h, y2 + padding)
            crop_img = image.crop((x_margin, pad_y1, w - x_margin, pad_y2))

            line_crops.append({
                "line_idx": idx + 1,
                "box": (pad_y1, pad_y2, x_margin, w - x_margin),
                "crop": crop_img
            })

        return line_crops


# Singleton instance
image_preprocessor = ImagePreprocessor()
