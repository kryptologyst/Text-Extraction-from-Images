# Project 120. Text extraction from images - MODERNIZED VERSION
# Description:
# This is the original simple implementation. For the full modern application with
# multiple OCR engines, web UI, batch processing, and database integration,
# please use the other modules in this project.

# 🚀 MODERN IMPLEMENTATION AVAILABLE:
# - Run 'python run.py' or 'streamlit run app.py' for the web interface
# - Use 'modern_ocr.py' for advanced OCR with multiple engines
# - Use 'batch_processor.py' for processing multiple images
# - See README.md for full documentation

# Original Python Implementation Using OpenCV + Tesseract

# Install if not already: pip install pytesseract opencv-python pillow
# Install Tesseract OCR binary from https://github.com/tesseract-ocr/tesseract
 
import cv2
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
 
# Load the image
image_path = "scene_text.jpg"  # Replace with your own image path
image = cv2.imread(image_path)
 
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Optional: apply thresholding to improve OCR
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
 
# Optional: noise removal
gray = cv2.medianBlur(gray, 3)
 
# Perform OCR using pytesseract
text = pytesseract.image_to_string(gray)
 
# Show the original image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("🖼️ Input Image")
plt.axis("off")
plt.show()
 
# Print the extracted text
print("📝 Extracted Text:\n")
print(text)

# 🔄 MODERN ALTERNATIVE:
# For better results, use the modern implementation:
# from modern_ocr import TextExtractor
# extractor = TextExtractor()
# results = extractor.extract_from_image("your_image.jpg", engine="all")
# for result in results:
#     print(f"{result.engine}: {result.text} (confidence: {result.confidence:.2%})")

# 📄 What This Project Demonstrates:
# - Detects and extracts text from images using OCR
# - Uses image preprocessing (thresholding, noise reduction) to enhance accuracy
# - Ideal for scene text, business cards, IDs, signboards, etc.

# 🚀 UPGRADE TO MODERN VERSION:
# The modern version includes:
# - Multiple OCR engines (EasyOCR, PaddleOCR, Tesseract)
# - Advanced image preprocessing
# - Web interface with Streamlit
# - Batch processing capabilities
# - Database integration
# - Configuration management
# - Comprehensive testing