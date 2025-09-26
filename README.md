# Text Extraction from Images

A comprehensive, modern text extraction application that uses state-of-the-art OCR engines to extract text from images with advanced preprocessing capabilities, batch processing, and a beautiful web interface.

## Features

### Multiple OCR Engines
- **EasyOCR**: Deep learning-based OCR with excellent accuracy
- **PaddleOCR**: High-performance OCR with angle detection
- **Tesseract**: Traditional OCR with extensive language support
- **Engine Comparison**: Compare results from multiple engines simultaneously

### Advanced Image Processing
- **Denoising**: Remove noise from images for better OCR accuracy
- **Deskewing**: Correct skewed text orientation
- **Contrast Enhancement**: Improve text visibility using CLAHE
- **Preprocessing Pipeline**: Configurable preprocessing options

### Modern Web Interface
- **Streamlit UI**: Beautiful, responsive web interface
- **Single Image Processing**: Upload and process individual images
- **Batch Processing**: Process multiple images simultaneously
- **Real-time Progress**: Live progress tracking for batch operations
- **Results Visualization**: Interactive charts and statistics

### Database Integration
- **SQLite Database**: Store extraction results and metadata
- **Search Functionality**: Search through extracted text
- **Statistics Dashboard**: Performance metrics and analytics
- **Export Options**: JSON and CSV export formats

### Configuration Management
- **Pydantic Settings**: Type-safe configuration management
- **Environment Variables**: Easy deployment configuration
- **Dynamic Settings**: Runtime configuration updates

## 🛠️ Installation

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR** (for Tesseract engine):
   ```bash
   # macOS
   brew install tesseract
   
   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   
   # Windows
   # Download from: https://github.com/tesseract-ocr/tesseract
   ```

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd 0120_Text_extraction_from_images
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## Usage

### Web Interface

1. **Single Image Processing**:
   - Upload an image file (JPG, PNG, BMP, TIFF)
   - Select OCR engine(s) and preprocessing options
   - Click "Extract Text" to process
   - View results with confidence scores and processing time

2. **Batch Processing**:
   - Upload multiple image files
   - Configure processing options
   - Monitor progress in real-time
   - Export results as JSON or CSV

3. **Statistics Dashboard**:
   - View processing statistics
   - Compare engine performance
   - Browse recent extractions
   - Search through stored results

### Command Line Usage

```python
from modern_ocr import TextExtractor

# Initialize extractor
extractor = TextExtractor()

# Process single image
results = extractor.extract_from_image(
    "path/to/image.jpg",
    preprocess=True,
    engine="all"  # or "easyocr", "paddleocr", "tesseract"
)

# Display results
for result in results:
    print(f"Engine: {result.engine}")
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Processing Time: {result.processing_time:.2f}s")
    print("-" * 50)
```

### Batch Processing

```python
from batch_processor import BatchProcessor

# Initialize processor
processor = BatchProcessor(max_workers=4)

# Process directory
results = processor.process_directory(
    "path/to/images/",
    recursive=True,
    engine="all",
    preprocess=True
)

# Export results
processor.export_results(results, "output.json", format="json")
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# OCR Settings
DEFAULT_ENGINE=all
ENABLE_PREPROCESSING=true

# Database
DATABASE_URL=sqlite:///ocr_results.db
ENABLE_DATABASE=true

# File Settings
MAX_FILE_SIZE=10485760  # 10MB
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs

# Tesseract
TESSERACT_CONFIG=--psm 6
TESSERACT_LANGUAGE=eng

# EasyOCR
EASYOCR_LANGUAGES=["en"]
EASYOCR_GPU=false

# PaddleOCR
PADDLEOCR_LANGUAGE=en
PADDLEOCR_USE_ANGLE_CLS=true
PADDLEOCR_USE_GPU=false

# Web UI
STREAMLIT_PORT=8501
STREAMLIT_HOST=localhost
DEBUG_MODE=false
```

### Configuration File

You can also modify settings directly in `config.py`:

```python
from config import config

# Update settings
config.update_setting('default_engine', 'easyocr')
config.update_setting('enable_preprocessing', True)
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests.py -v

# Run specific test class
pytest tests.py::TestImagePreprocessor -v

# Run with coverage
pytest tests.py --cov=modern_ocr --cov=config --cov=database
```

## 📁 Project Structure

```
0120_Text_extraction_from_images/
├── app.py                 # Streamlit web application
├── modern_ocr.py          # Core OCR functionality
├── batch_processor.py     # Batch processing
├── config.py              # Configuration management
├── database.py            # Database models and operations
├── tests.py               # Unit tests
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (create this)
├── .gitignore             # Git ignore rules
├── uploads/               # Uploaded images (auto-created)
├── outputs/               # Output files (auto-created)
└── logs/                  # Log files (auto-created)
```

## 🔧 API Reference

### TextExtractor

Main class for text extraction operations.

```python
class TextExtractor:
    def extract_from_image(self, image_path: str, preprocess: bool = True, 
                          engine: str = "all") -> List[OCRResult]
    def visualize_results(self, image_path: str, results: List[OCRResult])
```

### BatchProcessor

Handles batch processing of multiple images.

```python
class BatchProcessor:
    def process_images_batch(self, image_paths: List[str], engine: str = "all", 
                           preprocess: bool = True, job_name: Optional[str] = None) -> Dict[str, Any]
    def process_directory(self, directory_path: str, recursive: bool = False, 
                         engine: str = "all", preprocess: bool = True) -> Dict[str, Any]
    def export_results(self, results: Dict[str, Any], output_path: str, 
                      format: str = "json") -> str
```

### DatabaseManager

Manages database operations for storing and retrieving OCR results.

```python
class DatabaseManager:
    def save_ocr_result(self, result_data: Dict[str, Any]) -> int
    def get_ocr_results(self, limit: int = 100, offset: int = 0, 
                       engine: Optional[str] = None) -> List[Dict[str, Any]]
    def search_results(self, search_term: str, engine: Optional[str] = None) -> List[Dict[str, Any]]
    def get_statistics(self) -> Dict[str, Any]
```

## Performance Tips

1. **GPU Acceleration**: Enable GPU support for EasyOCR and PaddleOCR for faster processing
2. **Batch Processing**: Use batch processing for multiple images to improve efficiency
3. **Preprocessing**: Enable preprocessing for better accuracy on noisy or skewed images
4. **Engine Selection**: Choose the most appropriate engine for your use case:
   - **EasyOCR**: Best for general text extraction
   - **PaddleOCR**: Best for rotated text and complex layouts
   - **Tesseract**: Best for clean, well-formatted documents

## Troubleshooting

### Common Issues

1. **Tesseract not found**:
   ```bash
   # Install Tesseract OCR
   brew install tesseract  # macOS
   sudo apt-get install tesseract-ocr  # Ubuntu
   ```

2. **CUDA/GPU issues**:
   - Ensure CUDA is properly installed
   - Set `EASYOCR_GPU=false` and `PADDLEOCR_USE_GPU=false` if GPU is not available

3. **Memory issues with large images**:
   - Reduce image size before processing
   - Use batch processing with smaller batches
   - Increase system memory or use cloud processing

4. **Database connection issues**:
   - Check database URL in configuration
   - Ensure write permissions for database file
   - Reset database by deleting `ocr_results.db`

### Logging

Enable debug logging by setting `DEBUG_MODE=true` in your `.env` file or configuration.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [EasyOCR](https://github.com/JaidedAI/EasyOCR) - Deep learning OCR
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - High-performance OCR
- [Tesseract](https://github.com/tesseract-ocr/tesseract) - Traditional OCR engine
- [Streamlit](https://streamlit.io/) - Web application framework
- [OpenCV](https://opencv.org/) - Computer vision library


# Text-Extraction-from-Images
