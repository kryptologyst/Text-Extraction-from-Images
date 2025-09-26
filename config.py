"""
Configuration management for OCR application
"""

from pydantic import BaseSettings, Field
from typing import List, Optional
import os
from pathlib import Path

class OCRSettings(BaseSettings):
    """Configuration settings for OCR application"""
    
    # OCR Engine Settings
    default_engine: str = Field(default="all", description="Default OCR engine to use")
    available_engines: List[str] = Field(default=["easyocr", "paddleocr", "tesseract"], 
                                       description="Available OCR engines")
    
    # Image Processing Settings
    enable_preprocessing: bool = Field(default=True, description="Enable image preprocessing")
    enable_denoising: bool = Field(default=True, description="Enable denoising")
    enable_deskewing: bool = Field(default=True, description="Enable deskewing")
    enable_contrast_enhancement: bool = Field(default=True, description="Enable contrast enhancement")
    
    # Database Settings
    database_url: str = Field(default="sqlite:///ocr_results.db", description="Database URL")
    enable_database: bool = Field(default=True, description="Enable database storage")
    
    # File Settings
    upload_folder: str = Field(default="uploads", description="Folder for uploaded images")
    output_folder: str = Field(default="outputs", description="Folder for output files")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes (10MB)")
    allowed_extensions: List[str] = Field(default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"], 
                                        description="Allowed image extensions")
    
    # Tesseract Settings
    tesseract_config: str = Field(default="--psm 6", description="Tesseract configuration")
    tesseract_language: str = Field(default="eng", description="Tesseract language")
    
    # EasyOCR Settings
    easyocr_languages: List[str] = Field(default=["en"], description="EasyOCR languages")
    easyocr_gpu: bool = Field(default=False, description="Use GPU for EasyOCR")
    
    # PaddleOCR Settings
    paddleocr_language: str = Field(default="en", description="PaddleOCR language")
    paddleocr_use_angle_cls: bool = Field(default=True, description="Use angle classification")
    paddleocr_use_gpu: bool = Field(default=False, description="Use GPU for PaddleOCR")
    
    # Web UI Settings
    streamlit_port: int = Field(default=8501, description="Streamlit port")
    streamlit_host: str = Field(default="localhost", description="Streamlit host")
    debug_mode: bool = Field(default=False, description="Enable debug mode")
    
    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class ConfigManager:
    """Configuration manager singleton"""
    
    _instance = None
    _settings = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._settings is None:
            self._settings = OCRSettings()
            self._create_directories()
    
    @property
    def settings(self) -> OCRSettings:
        """Get current settings"""
        return self._settings
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self._settings.upload_folder,
            self._settings.output_folder,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def update_setting(self, key: str, value):
        """Update a setting dynamically"""
        if hasattr(self._settings, key):
            setattr(self._settings, key, value)
        else:
            raise ValueError(f"Setting '{key}' does not exist")
    
    def get_engine_config(self, engine: str) -> dict:
        """Get configuration for specific OCR engine"""
        configs = {
            "tesseract": {
                "config": self._settings.tesseract_config,
                "language": self._settings.tesseract_language
            },
            "easyocr": {
                "languages": self._settings.easyocr_languages,
                "gpu": self._settings.easyocr_gpu
            },
            "paddleocr": {
                "language": self._settings.paddleocr_language,
                "use_angle_cls": self._settings.paddleocr_use_angle_cls,
                "use_gpu": self._settings.paddleocr_use_gpu
            }
        }
        
        return configs.get(engine, {})
    
    def validate_image_file(self, file_path: str) -> bool:
        """Validate if image file meets requirements"""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            return False
        
        # Check file extension
        if path.suffix.lower() not in self._settings.allowed_extensions:
            return False
        
        # Check file size
        if path.stat().st_size > self._settings.max_file_size:
            return False
        
        return True

# Global configuration instance
config = ConfigManager()
