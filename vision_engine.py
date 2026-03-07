import cv2
import numpy as np
import easyocr
import re
from PIL import Image

class VisionEngine:
    def __init__(self):
        # Initialize OCR reader (English only for now)
        # Using CPU to ensure compatibility in all environments
        self.reader = easyocr.Reader(['en'], gpu=False)

    def process_screenshot(self, image_path):
        """
        Main entry point for analyzing a chart screenshot.
        """
        try:
            # 1. OCR Extraction
            ocr_data = self._extract_metadata(image_path)

            # 2. Chart Element Detection
            levels = self._detect_support_resistance(image_path)

            # 3. Trend Analysis (Simple heuristic)
            trend = self._analyze_trend(image_path)

            return {
                "ticker": ocr_data.get("ticker", "UNKNOWN"),
                "current_price": ocr_data.get("price", 0.0),
                "support_levels": levels.get("support", []),
                "resistance_levels": levels.get("resistance", []),
                "trend": trend,
                "confidence": 0.75 # Heuristic confidence
            }
        except Exception as e:
            return {"error": str(e)}

    def _extract_metadata(self, image_path):
        """
        Extracts Ticker and Price using OCR.
        """
        results = self.reader.readtext(image_path)
        full_text = " ".join([res[1] for res in results]).upper()

        # Heuristic Ticker: 1-5 uppercase letters (e.g., AAPL, NVDA)
        # Usually appears at top left
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', full_text)
        ticker = ticker_match.group(0) if ticker_match else "UNKNOWN"

        # Heuristic Price: Numbers with decimal (e.g., 150.25)
        price_matches = re.findall(r'\d+\.\d{2}', full_text)
        price = float(price_matches[0]) if price_matches else 0.0

        return {"ticker": ticker, "price": price}

    def _detect_support_resistance(self, image_path):
        """
        Uses OpenCV to identify horizontal lines (S/R candidates).
        """
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) < 5: # Horizontal check
                    horizontal_lines.append(y1)

        # Cluster lines into support and resistance levels
        # In a real system, we'd compare Y-coordinates to current price pixels
        # For now, we return empty placeholders to be filled by quantitative engine
        return {"support": [], "resistance": []}

    def _analyze_trend(self, image_path):
        """
        Basic color-based trend heuristic.
        """
        img = cv2.imread(image_path)
        # Check ratio of green vs red pixels in the center of the chart
        # This is very naive but serves as a vision-based feature
        h, w, _ = img.shape
        center_crop = img[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)]

        green_mask = cv2.inRange(center_crop, (0, 100, 0), (100, 255, 100))
        red_mask = cv2.inRange(center_crop, (0, 0, 100), (100, 100, 255))

        green_px = cv2.countNonZero(green_mask)
        red_px = cv2.countNonZero(red_mask)

        if green_px > red_px * 1.2: return "BULLISH"
        if red_px > green_px * 1.2: return "BEARISH"
        return "NEUTRAL"

if __name__ == "__main__":
    # Test block
    engine = VisionEngine()
    print("Vision Engine initialized.")
