"""
Modern Web UI for Text Extraction from Images using Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time
from typing import List, Dict, Any
import logging

# Import our modules
from modern_ocr import TextExtractor, OCRResult
from batch_processor import BatchProcessor
from database import db_manager
from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🔤 Modern Text Extraction",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .success-text {
        color: #28a745;
        font-weight: bold;
    }
    .error-text {
        color: #dc3545;
        font-weight: bold;
    }
    .info-text {
        color: #17a2b8;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'extractor' not in st.session_state:
        st.session_state.extractor = TextExtractor()
    
    if 'batch_processor' not in st.session_state:
        st.session_state.batch_processor = BatchProcessor()
    
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    
    if 'batch_results' not in st.session_state:
        st.session_state.batch_results = None

def display_header():
    """Display the main header"""
    st.markdown('<h1 class="main-header">🔤 Modern Text Extraction from Images</h1>', 
                unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Extract text from images using state-of-the-art OCR engines: EasyOCR, PaddleOCR, and Tesseract
        </p>
    </div>
    """, unsafe_allow_html=True)

def sidebar_configuration():
    """Sidebar configuration options"""
    st.sidebar.header("⚙️ Configuration")
    
    # OCR Engine Selection
    engine_options = {
        "All Engines": "all",
        "EasyOCR": "easyocr",
        "PaddleOCR": "paddleocr", 
        "Tesseract": "tesseract"
    }
    
    selected_engine = st.sidebar.selectbox(
        "Select OCR Engine:",
        list(engine_options.keys()),
        index=0
    )
    
    # Image Processing Options
    st.sidebar.subheader("🖼️ Image Processing")
    enable_preprocessing = st.sidebar.checkbox("Enable Preprocessing", value=True)
    
    if enable_preprocessing:
        st.sidebar.write("**Preprocessing Options:**")
        enable_denoising = st.sidebar.checkbox("Denoising", value=True)
        enable_deskewing = st.sidebar.checkbox("Deskewing", value=True)
        enable_enhancement = st.sidebar.checkbox("Contrast Enhancement", value=True)
    else:
        enable_denoising = enable_deskewing = enable_enhancement = False
    
    # Database Options
    st.sidebar.subheader("💾 Database")
    show_database_stats = st.sidebar.checkbox("Show Database Statistics", value=True)
    
    return {
        'engine': engine_options[selected_engine],
        'preprocessing': enable_preprocessing,
        'denoising': enable_denoising,
        'deskewing': enable_deskewing,
        'enhancement': enable_enhancement,
        'show_db_stats': show_database_stats
    }

def single_image_processing():
    """Single image processing interface"""
    st.header("📸 Single Image Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload an image file to extract text from"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("⚙️ Processing Options")
            
            # Get configuration from sidebar
            config_options = sidebar_configuration()
            
            # Process button
            if st.button("🚀 Extract Text", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Save uploaded file temporarily
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Process image
                        results = st.session_state.extractor.extract_from_image(
                            temp_path,
                            preprocess=config_options['preprocessing'],
                            engine=config_options['engine']
                        )
                        
                        # Store results
                        st.session_state.processing_results = results
                        
                        # Clean up temp file
                        Path(temp_path).unlink()
                        
                        st.success("✅ Text extraction completed!")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}")
                        logger.error(f"Error processing image: {e}")
        
        # Display results
        if st.session_state.processing_results:
            display_single_results(st.session_state.processing_results)

def display_single_results(results: List[OCRResult]):
    """Display single image processing results"""
    st.header("📝 Extraction Results")
    
    # Create tabs for each engine result
    if len(results) > 1:
        tab_names = [f"{result.engine} ({result.confidence:.2f})" for result in results]
        tabs = st.tabs(tab_names)
        
        for i, (tab, result) in enumerate(zip(tabs, results)):
            with tab:
                display_engine_result(result)
    else:
        display_engine_result(results[0])

def display_engine_result(result: OCRResult):
    """Display result for a specific OCR engine"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🔤 Extracted Text ({result.engine})")
        st.text_area(
            "Extracted Text:",
            value=result.text,
            height=200,
            disabled=True
        )
    
    with col2:
        st.subheader("📊 Statistics")
        
        # Confidence metric
        confidence_color = "success" if result.confidence > 0.8 else "warning" if result.confidence > 0.5 else "danger"
        st.metric(
            "Confidence",
            f"{result.confidence:.2%}",
            delta=f"{'High' if result.confidence > 0.8 else 'Medium' if result.confidence > 0.5 else 'Low'} confidence"
        )
        
        # Processing time
        st.metric("Processing Time", f"{result.processing_time:.2f}s")
        
        # Text length
        st.metric("Text Length", f"{len(result.text)} characters")
        
        # Word count
        word_count = len(result.text.split()) if result.text.strip() else 0
        st.metric("Word Count", f"{word_count} words")

def batch_processing():
    """Batch processing interface"""
    st.header("📁 Batch Processing")
    
    # File upload for multiple images
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        accept_multiple_files=True,
        help="Upload multiple image files for batch processing"
    )
    
    if uploaded_files:
        st.info(f"📊 {len(uploaded_files)} files selected for processing")
        
        # Display file list
        with st.expander("📋 Selected Files"):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name} ({file.size:,} bytes)")
        
        # Batch processing options
        col1, col2 = st.columns([1, 1])
        
        with col1:
            max_workers = st.slider("Max Workers", 1, 8, 4)
        
        with col2:
            config_options = sidebar_configuration()
        
        # Process batch button
        if st.button("🚀 Process Batch", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded files temporarily
                temp_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    temp_paths.append(temp_path)
                
                # Update processor settings
                st.session_state.batch_processor.max_workers = max_workers
                
                # Progress callback
                def progress_callback(processed, total, current_file):
                    progress = processed / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {processed}/{total}: {current_file}")
                
                st.session_state.batch_processor.set_progress_callback(progress_callback)
                
                # Process batch
                batch_results = st.session_state.batch_processor.process_images_batch(
                    temp_paths,
                    engine=config_options['engine'],
                    preprocess=config_options['preprocessing'],
                    job_name=f"Batch_{int(time.time())}"
                )
                
                # Store results
                st.session_state.batch_results = batch_results
                
                # Clean up temp files
                for temp_path in temp_paths:
                    Path(temp_path).unlink()
                
                progress_bar.progress(1.0)
                status_text.text("✅ Batch processing completed!")
                
                st.success(f"✅ Processed {batch_results['total_images']} images successfully!")
                
            except Exception as e:
                st.error(f"❌ Error in batch processing: {str(e)}")
                logger.error(f"Error in batch processing: {e}")
        
        # Display batch results
        if st.session_state.batch_results:
            display_batch_results(st.session_state.batch_results)

def display_batch_results(results: Dict[str, Any]):
    """Display batch processing results"""
    st.header("📊 Batch Processing Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", results['total_images'])
    
    with col2:
        st.metric("Successful", results['successful'], delta=f"{results['summary']['success_rate']:.1%}")
    
    with col3:
        st.metric("Failed", results['failed'])
    
    with col4:
        st.metric("Avg Confidence", f"{results['summary']['average_confidence']:.2%}")
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Prepare data for table
    table_data = []
    for result in results['results']:
        if 'results' in result:
            for ocr_result in result['results']:
                table_data.append({
                    'Image': result['image_filename'],
                    'Engine': ocr_result['engine'],
                    'Text Length': len(ocr_result['extracted_text']),
                    'Confidence': f"{ocr_result['confidence']:.2%}",
                    'Processing Time': f"{ocr_result['processing_time']:.2f}s",
                    'Status': '✅ Success' if 'error' not in result else '❌ Failed'
                })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export as JSON"):
                json_str = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"batch_results_{int(time.time())}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 Export as CSV"):
                csv_data = []
                for result in results['results']:
                    if 'results' in result:
                        for ocr_result in result['results']:
                            csv_data.append({
                                'image_filename': result['image_filename'],
                                'engine': ocr_result['engine'],
                                'extracted_text': ocr_result['extracted_text'],
                                'confidence': ocr_result['confidence'],
                                'processing_time': ocr_result['processing_time']
                            })
                
                df_csv = pd.DataFrame(csv_data)
                csv_str = df_csv.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_str,
                    file_name=f"batch_results_{int(time.time())}.csv",
                    mime="text/csv"
                )

def database_statistics():
    """Display database statistics"""
    if not config.settings.enable_database:
        st.warning("⚠️ Database is disabled in configuration")
        return
    
    st.header("📊 Database Statistics")
    
    try:
        stats = db_manager.get_statistics()
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Extractions", stats['total_results'])
        
        with col2:
            st.metric("Average Processing Time", f"{stats['average_processing_time']:.2f}s")
        
        with col3:
            st.metric("OCR Engines Used", stats['total_engines'])
        
        # Engine statistics
        st.subheader("🔧 Engine Performance")
        
        engine_data = []
        for engine, data in stats['engine_statistics'].items():
            engine_data.append({
                'Engine': engine,
                'Count': data['count'],
                'Avg Confidence': f"{data['average_confidence']:.2%}"
            })
        
        if engine_data:
            df_engines = pd.DataFrame(engine_data)
            
            # Bar chart for engine usage
            fig = px.bar(df_engines, x='Engine', y='Count', 
                        title="Extractions by Engine",
                        color='Count',
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence comparison
            fig_conf = px.bar(df_engines, x='Engine', y='Avg Confidence',
                             title="Average Confidence by Engine",
                             color='Avg Confidence',
                             color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # Recent extractions
        st.subheader("🕒 Recent Extractions")
        recent_results = db_manager.get_ocr_results(limit=10)
        
        if recent_results:
            recent_df = pd.DataFrame(recent_results)
            st.dataframe(
                recent_df[['image_filename', 'engine', 'confidence', 'created_at']],
                use_container_width=True
            )
        else:
            st.info("No recent extractions found")
    
    except Exception as e:
        st.error(f"❌ Error loading database statistics: {str(e)}")
        logger.error(f"Error loading database statistics: {e}")

def main():
    """Main application function"""
    initialize_session_state()
    display_header()
    
    # Sidebar configuration
    config_options = sidebar_configuration()
    
    # Main navigation
    tab1, tab2, tab3 = st.tabs(["📸 Single Image", "📁 Batch Processing", "📊 Statistics"])
    
    with tab1:
        single_image_processing()
    
    with tab2:
        batch_processing()
    
    with tab3:
        if config_options['show_db_stats']:
            database_statistics()
        else:
            st.info("Enable 'Show Database Statistics' in the sidebar to view statistics")

if __name__ == "__main__":
    main()
