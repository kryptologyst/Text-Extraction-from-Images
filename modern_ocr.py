"""
Modern Text Extraction from Images
Supports multiple OCR engines with advanced preprocessing
"""

import cv2
import numpy as np
import easyocr
import pytesseract
from paddleocr import PaddleOCR
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRResult:
    """Data class for OCR results"""
    text: str
    confidence: float
    engine: str
    processing_time: float
    bounding_boxes: List[Dict] = None
    metadata: Dict = None

class ImagePreprocessor:
    """Advanced image preprocessing for better OCR accuracy"""
    
    @staticmethod
    def denoise_image(image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        return cv2.fastNlMeansDenoising(image)
    
    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """Correct skew in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(image)
    
    @staticmethod
    def preprocess_image(image: np.ndarray, denoise: bool = True, deskew: bool = True, 
                        enhance: bool = True) -> np.ndarray:
        """Apply comprehensive preprocessing"""
        processed = image.copy()
        
        if denoise:
            processed = ImagePreprocessor.denoise_image(processed)
        
        if deskew:
            processed = ImagePreprocessor.deskew_image(processed)
            
        if enhance:
            processed = ImagePreprocessor.enhance_contrast(processed)
            
        return processed

class ModernOCREngine:
    """Modern OCR engine supporting multiple backends"""
    
    def __init__(self):
        self.easyocr_reader = None
        self.paddleocr_reader = None
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize OCR engines"""
        try:
            self.easyocr_reader = easyocr.Reader(['en'])
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
        
        try:
            self.paddleocr_reader = PaddleOCR(use_angle_cls=True, lang='en')
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
    
    def extract_with_easyocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using EasyOCR"""
        import time
        start_time = time.time()
        
        if self.easyocr_reader is None:
            raise RuntimeError("EasyOCR not initialized")
        
        results = self.easyocr_reader.readtext(image)
        text = " ".join([result[1] for result in results])
        confidences = [result[2] for result in results]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        bounding_boxes = []
        for result in results:
            bbox = result[0]
            bounding_boxes.append({
                'points': bbox,
                'text': result[1],
                'confidence': result[2]
            })
        
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=text,
            confidence=avg_confidence,
            engine="EasyOCR",
            processing_time=processing_time,
            bounding_boxes=bounding_boxes
        )
    
    def extract_with_paddleocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using PaddleOCR"""
        import time
        start_time = time.time()
        
        if self.paddleocr_reader is None:
            raise RuntimeError("PaddleOCR not initialized")
        
        results = self.paddleocr_reader.ocr(image, cls=True)
        
        text_parts = []
        confidences = []
        bounding_boxes = []
        
        if results and results[0]:
            for line in results[0]:
                if line:
                    bbox, (text, confidence) = line
                    text_parts.append(text)
                    confidences.append(confidence)
                    bounding_boxes.append({
                        'points': bbox,
                        'text': text,
                        'confidence': confidence
                    })
        
        text = " ".join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=text,
            confidence=avg_confidence,
            engine="PaddleOCR",
            processing_time=processing_time,
            bounding_boxes=bounding_boxes
        )
    
    def extract_with_tesseract(self, image: np.ndarray) -> OCRResult:
        """Extract text using Tesseract"""
        import time
        start_time = time.time()
        
        # Convert to PIL Image for pytesseract
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        # Get detailed data for confidence scores
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        text_parts = []
        confidences = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Only include text with confidence > 0
                text_parts.append(data['text'][i])
                confidences.append(int(data['conf'][i]))
        
        text = " ".join(text_parts)
        avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0  # Convert to 0-1 scale
        processing_time = time.time() - start_time
        
        return OCRResult(
            text=text,
            confidence=avg_confidence,
            engine="Tesseract",
            processing_time=processing_time
        )
    
    def extract_text(self, image: np.ndarray, engine: str = "all") -> List[OCRResult]:
        """Extract text using specified engine(s)"""
        results = []
        
        if engine == "all" or engine == "easyocr":
            try:
                result = self.extract_with_easyocr(image)
                results.append(result)
            except Exception as e:
                logger.error(f"EasyOCR failed: {e}")
        
        if engine == "all" or engine == "paddleocr":
            try:
                result = self.extract_with_paddleocr(image)
                results.append(result)
            except Exception as e:
                logger.error(f"PaddleOCR failed: {e}")
        
        if engine == "all" or engine == "tesseract":
            try:
                result = self.extract_with_tesseract(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Tesseract failed: {e}")
        
        return results

class TextExtractor:
    """Main text extraction class"""
    
    def __init__(self):
        self.ocr_engine = ModernOCREngine()
        self.preprocessor = ImagePreprocessor()
    
    def extract_from_image(self, image_path: str, preprocess: bool = True, 
                          engine: str = "all") -> List[OCRResult]:
        """Extract text from image file"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Preprocess if requested
        if preprocess:
            image = self.preprocessor.preprocess_image(image)
        
        # Extract text
        results = self.ocr_engine.extract_text(image, engine)
        
        # Add metadata
        for result in results:
            result.metadata = {
                'image_path': image_path,
                'image_shape': image.shape,
                'preprocessed': preprocess,
                'timestamp': datetime.now().isoformat()
            }
        
        return results
    
    def visualize_results(self, image_path: str, results: List[OCRResult]):
        """Visualize OCR results on image"""
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 5))
        if len(results) == 1:
            axes = [axes]
        
        # Show original image
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Show results for each engine
        for i, result in enumerate(results):
            axes[i + 1].imshow(image_rgb)
            axes[i + 1].set_title(f"{result.engine}\nConfidence: {result.confidence:.2f}")
            axes[i + 1].axis('off')
            
            # Draw bounding boxes if available
            if result.bounding_boxes:
                for bbox in result.bounding_boxes:
                    points = np.array(bbox['points'], dtype=np.int32)
                    cv2.polylines(image_rgb, [points], True, (0, 255, 0), 2)
        
        plt.tight_layout()
        plt.show()
        
        # Print extracted text
        for result in results:
            print(f"\n🔤 {result.engine} Results:")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Text: {result.text}")
            print("-" * 50)

def main():
    """Main function for testing"""
    extractor = TextExtractor()
    
    # Test with sample image (you'll need to provide an image)
    image_path = "scene_text.jpg"  # Replace with your image path
    
    if Path(image_path).exists():
        try:
            results = extractor.extract_from_image(image_path)
            extractor.visualize_results(image_path, results)
        except Exception as e:
            logger.error(f"Error processing image: {e}")
    else:
        print(f"Image file '{image_path}' not found. Please provide a valid image path.")

if __name__ == "__main__":
    main()
