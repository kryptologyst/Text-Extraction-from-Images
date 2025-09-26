"""
Unit tests for OCR text extraction application
"""

import pytest
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Import modules to test
from modern_ocr import ImagePreprocessor, ModernOCREngine, TextExtractor, OCRResult
from config import ConfigManager, OCRSettings
from database import DatabaseManager, OCRResultDB, BatchJobDB
from batch_processor import BatchProcessor

class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = ImagePreprocessor()
        # Create a test image
        self.test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.test_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    def test_denoise_image(self):
        """Test image denoising"""
        result = self.preprocessor.denoise_image(self.test_image)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_deskew_image(self):
        """Test image deskewing"""
        result = self.preprocessor.deskew_image(self.test_image)
        assert result.shape == self.test_image.shape
    
    def test_enhance_contrast(self):
        """Test contrast enhancement"""
        result = self.preprocessor.enhance_contrast(self.test_image)
        assert result.shape == self.test_image.shape
        assert result.dtype == self.test_image.dtype
    
    def test_preprocess_image_full(self):
        """Test full preprocessing pipeline"""
        result = self.preprocessor.preprocess_image(
            self.test_image, 
            denoise=True, 
            deskew=True, 
            enhance=True
        )
        assert result.shape == self.test_image.shape
    
    def test_preprocess_image_partial(self):
        """Test partial preprocessing"""
        result = self.preprocessor.preprocess_image(
            self.test_image,
            denoise=True,
            deskew=False,
            enhance=False
        )
        assert result.shape == self.test_image.shape

class TestModernOCREngine:
    """Test cases for ModernOCREngine class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Mock the OCR engines to avoid actual initialization
        with patch('easyocr.Reader'), \
             patch('paddleocr.PaddleOCR'):
            self.engine = ModernOCREngine()
    
    def test_initialization(self):
        """Test OCR engine initialization"""
        assert self.engine is not None
    
    @patch('pytesseract.image_to_string')
    @patch('pytesseract.image_to_data')
    def test_extract_with_tesseract(self, mock_data, mock_string):
        """Test Tesseract extraction"""
        # Mock tesseract responses
        mock_string.return_value = "Test text"
        mock_data.return_value = {
            'text': ['Test', 'text'],
            'conf': [90, 85]
        }
        
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.engine.extract_with_tesseract(test_image)
        
        assert isinstance(result, OCRResult)
        assert result.engine == "Tesseract"
        assert result.text == "Test text"
        assert result.confidence > 0
    
    def test_extract_text_all_engines(self):
        """Test extraction with all engines"""
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with patch.object(self.engine, 'extract_with_tesseract') as mock_tesseract:
            mock_tesseract.return_value = OCRResult(
                text="Test",
                confidence=0.9,
                engine="Tesseract",
                processing_time=1.0
            )
            
            results = self.engine.extract_text(test_image, engine="tesseract")
            assert len(results) == 1
            assert results[0].engine == "Tesseract"

class TestTextExtractor:
    """Test cases for TextExtractor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('easyocr.Reader'), \
             patch('paddleocr.PaddleOCR'):
            self.extractor = TextExtractor()
    
    def test_initialization(self):
        """Test TextExtractor initialization"""
        assert self.extractor is not None
        assert self.extractor.ocr_engine is not None
        assert self.extractor.preprocessor is not None
    
    def test_extract_from_image_file_not_found(self):
        """Test extraction with non-existent file"""
        with pytest.raises(ValueError):
            self.extractor.extract_from_image("nonexistent.jpg")

class TestConfigManager:
    """Test cases for ConfigManager class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Reset singleton
        ConfigManager._instance = None
        ConfigManager._settings = None
    
    def test_singleton_pattern(self):
        """Test singleton pattern"""
        config1 = ConfigManager()
        config2 = ConfigManager()
        assert config1 is config2
    
    def test_settings_property(self):
        """Test settings property"""
        config = ConfigManager()
        settings = config.settings
        assert isinstance(settings, OCRSettings)
    
    def test_update_setting(self):
        """Test dynamic setting update"""
        config = ConfigManager()
        original_value = config.settings.default_engine
        config.update_setting('default_engine', 'tesseract')
        assert config.settings.default_engine == 'tesseract'
    
    def test_update_invalid_setting(self):
        """Test updating invalid setting"""
        config = ConfigManager()
        with pytest.raises(ValueError):
            config.update_setting('invalid_setting', 'value')
    
    def test_get_engine_config(self):
        """Test getting engine configuration"""
        config = ConfigManager()
        tesseract_config = config.get_engine_config('tesseract')
        assert 'config' in tesseract_config
        assert 'language' in tesseract_config
    
    def test_validate_image_file(self):
        """Test image file validation"""
        config = ConfigManager()
        
        # Test with non-existent file
        assert not config.validate_image_file("nonexistent.jpg")
        
        # Test with valid temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b"fake image data")
            tmp_path = tmp.name
        
        try:
            assert config.validate_image_file(tmp_path)
        finally:
            os.unlink(tmp_path)

class TestDatabaseManager:
    """Test cases for DatabaseManager class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Use in-memory SQLite for testing
        self.test_db_url = "sqlite:///:memory:"
        with patch('database.config') as mock_config:
            mock_config.settings.database_url = self.test_db_url
            mock_config.settings.debug_mode = False
            self.db_manager = DatabaseManager()
    
    def test_initialization(self):
        """Test database initialization"""
        assert self.db_manager is not None
        assert self.db_manager.engine is not None
        assert self.db_manager.SessionLocal is not None
    
    def test_save_ocr_result(self):
        """Test saving OCR result"""
        result_data = {
            'image_path': 'test.jpg',
            'image_filename': 'test.jpg',
            'engine': 'tesseract',
            'extracted_text': 'Test text',
            'confidence': 0.9,
            'processing_time': 1.0,
            'bounding_boxes': None,
            'metadata': {'test': 'data'},
            'preprocessed': True
        }
        
        result_id = self.db_manager.save_ocr_result(result_data)
        assert isinstance(result_id, int)
        assert result_id > 0
    
    def test_get_ocr_results(self):
        """Test getting OCR results"""
        # First save a result
        result_data = {
            'image_path': 'test.jpg',
            'image_filename': 'test.jpg',
            'engine': 'tesseract',
            'extracted_text': 'Test text',
            'confidence': 0.9,
            'processing_time': 1.0,
            'bounding_boxes': None,
            'metadata': None,
            'preprocessed': False
        }
        
        self.db_manager.save_ocr_result(result_data)
        
        # Get results
        results = self.db_manager.get_ocr_results()
        assert len(results) == 1
        assert results[0]['engine'] == 'tesseract'
        assert results[0]['extracted_text'] == 'Test text'
    
    def test_search_results(self):
        """Test searching OCR results"""
        # Save a result
        result_data = {
            'image_path': 'test.jpg',
            'image_filename': 'test.jpg',
            'engine': 'tesseract',
            'extracted_text': 'Hello World',
            'confidence': 0.9,
            'processing_time': 1.0,
            'bounding_boxes': None,
            'metadata': None,
            'preprocessed': False
        }
        
        self.db_manager.save_ocr_result(result_data)
        
        # Search for text
        results = self.db_manager.search_results('Hello')
        assert len(results) == 1
        assert 'Hello' in results[0]['extracted_text']
    
    def test_get_statistics(self):
        """Test getting statistics"""
        # Save some test data
        for i in range(3):
            result_data = {
                'image_path': f'test{i}.jpg',
                'image_filename': f'test{i}.jpg',
                'engine': 'tesseract',
                'extracted_text': f'Test text {i}',
                'confidence': 0.8 + i * 0.05,
                'processing_time': 1.0 + i * 0.1,
                'bounding_boxes': None,
                'metadata': None,
                'preprocessed': False
            }
            self.db_manager.save_ocr_result(result_data)
        
        stats = self.db_manager.get_statistics()
        assert stats['total_results'] == 3
        assert 'tesseract' in stats['engine_statistics']
        assert stats['engine_statistics']['tesseract']['count'] == 3
    
    def test_create_batch_job(self):
        """Test creating batch job"""
        job_id = self.db_manager.create_batch_job("Test Job", 10)
        assert isinstance(job_id, int)
        assert job_id > 0
    
    def test_update_batch_job(self):
        """Test updating batch job"""
        job_id = self.db_manager.create_batch_job("Test Job", 10)
        
        success = self.db_manager.update_batch_job(
            job_id, 
            status="processing",
            processed_images=5
        )
        assert success

class TestBatchProcessor:
    """Test cases for BatchProcessor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        with patch('easyocr.Reader'), \
             patch('paddleocr.PaddleOCR'):
            self.processor = BatchProcessor(max_workers=2)
    
    def test_initialization(self):
        """Test BatchProcessor initialization"""
        assert self.processor is not None
        assert self.processor.max_workers == 2
        assert self.processor.extractor is not None
    
    def test_set_progress_callback(self):
        """Test setting progress callback"""
        def test_callback(processed, total, current_file):
            pass
        
        self.processor.set_progress_callback(test_callback)
        assert self.processor.progress_callback is not None
    
    def test_process_single_image_invalid_file(self):
        """Test processing invalid image file"""
        with patch('config.config.validate_image_file', return_value=False):
            result = self.processor.process_single_image("invalid.jpg")
            assert 'error' in result
    
    def test_calculate_average_confidence(self):
        """Test average confidence calculation"""
        results = [
            {
                'results': [
                    {'confidence': 0.8},
                    {'confidence': 0.9}
                ]
            },
            {
                'results': [
                    {'confidence': 0.7}
                ]
            }
        ]
        
        avg_conf = self.processor._calculate_average_confidence(results)
        assert avg_conf == 0.8  # (0.8 + 0.9 + 0.7) / 3

class TestOCRResult:
    """Test cases for OCRResult dataclass"""
    
    def test_ocr_result_creation(self):
        """Test OCRResult creation"""
        result = OCRResult(
            text="Test text",
            confidence=0.9,
            engine="tesseract",
            processing_time=1.0
        )
        
        assert result.text == "Test text"
        assert result.confidence == 0.9
        assert result.engine == "tesseract"
        assert result.processing_time == 1.0
        assert result.bounding_boxes is None
        assert result.metadata is None

# Integration tests
class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_processing(self):
        """Test end-to-end processing pipeline"""
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, test_image)
            image_path = tmp.name
        
        try:
            with patch('easyocr.Reader'), \
                 patch('paddleocr.PaddleOCR'), \
                 patch('pytesseract.image_to_string', return_value="Test text"), \
                 patch('pytesseract.image_to_data', return_value={'text': ['Test', 'text'], 'conf': [90, 85]}):
                
                extractor = TextExtractor()
                results = extractor.extract_from_image(image_path, engine="tesseract")
                
                assert len(results) == 1
                assert results[0].engine == "Tesseract"
                assert results[0].text == "Test text"
        
        finally:
            os.unlink(image_path)

# Fixtures for pytest
@pytest.fixture
def sample_image():
    """Create a sample test image"""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def temp_image_file(sample_image):
    """Create a temporary image file"""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, sample_image)
        yield tmp.name
        os.unlink(tmp.name)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
