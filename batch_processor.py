"""
Batch processing functionality for multiple images
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
import traceback

from modern_ocr import TextExtractor, OCRResult
from database import db_manager
from config import config

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Batch processor for multiple images"""
    
    def __init__(self, max_workers: int = 4):
        self.extractor = TextExtractor()
        self.max_workers = max_workers
        self.progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set progress callback function"""
        self.progress_callback = callback
    
    def _update_progress(self, processed: int, total: int, current_file: str):
        """Update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(processed, total, current_file)
    
    def process_single_image(self, image_path: str, engine: str = "all", 
                           preprocess: bool = True) -> Dict[str, Any]:
        """Process a single image and return results"""
        try:
            # Validate image file
            if not config.validate_image_file(image_path):
                raise ValueError(f"Invalid image file: {image_path}")
            
            # Extract text
            results = self.extractor.extract_from_image(image_path, preprocess, engine)
            
            # Prepare result data
            result_data = {
                'image_path': image_path,
                'image_filename': Path(image_path).name,
                'engine': engine,
                'preprocessed': preprocess,
                'results': []
            }
            
            # Process each OCR result
            for result in results:
                ocr_data = {
                    'engine': result.engine,
                    'extracted_text': result.text,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time,
                    'bounding_boxes': result.bounding_boxes,
                    'metadata': result.metadata
                }
                
                # Save to database if enabled
                if config.settings.enable_database:
                    try:
                        db_manager.save_ocr_result(ocr_data)
                    except Exception as e:
                        logger.warning(f"Failed to save to database: {e}")
                
                result_data['results'].append(ocr_data)
            
            return result_data
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'image_path': image_path,
                'image_filename': Path(image_path).name,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def process_images_batch(self, image_paths: List[str], engine: str = "all", 
                           preprocess: bool = True, job_name: Optional[str] = None) -> Dict[str, Any]:
        """Process multiple images in batch"""
        
        # Create batch job if database is enabled
        job_id = None
        if config.settings.enable_database and job_name:
            try:
                job_id = db_manager.create_batch_job(job_name, len(image_paths))
            except Exception as e:
                logger.warning(f"Failed to create batch job: {e}")
        
        # Update job status to processing
        if job_id:
            db_manager.update_batch_job(job_id, status="processing")
        
        results = []
        successful = 0
        failed = 0
        
        try:
            # Process images with thread pool
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(self.process_single_image, path, engine, preprocess): path
                    for path in image_paths
                }
                
                # Process completed tasks
                for i, future in enumerate(as_completed(future_to_path)):
                    path = future_to_path[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if 'error' in result:
                            failed += 1
                            logger.error(f"Failed to process {path}: {result['error']}")
                        else:
                            successful += 1
                        
                        # Update progress
                        self._update_progress(i + 1, len(image_paths), path)
                        
                        # Update batch job progress
                        if job_id:
                            db_manager.update_batch_job(
                                job_id,
                                processed_images=i + 1,
                                successful_extractions=successful,
                                failed_extractions=failed
                            )
                    
                    except Exception as e:
                        failed += 1
                        logger.error(f"Unexpected error processing {path}: {e}")
                        results.append({
                            'image_path': path,
                            'image_filename': Path(path).name,
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        })
        
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            if job_id:
                db_manager.update_batch_job(job_id, status="failed", error_message=str(e))
            raise
        
        # Update final job status
        if job_id:
            db_manager.update_batch_job(
                job_id,
                status="completed",
                completed_at=datetime.utcnow(),
                processed_images=len(image_paths),
                successful_extractions=successful,
                failed_extractions=failed
            )
        
        return {
            'job_id': job_id,
            'total_images': len(image_paths),
            'successful': successful,
            'failed': failed,
            'results': results,
            'summary': {
                'success_rate': successful / len(image_paths) if image_paths else 0,
                'average_confidence': self._calculate_average_confidence(results),
                'total_processing_time': sum(
                    sum(r.get('processing_time', 0) for r in result.get('results', []))
                    for result in results if 'results' in result
                )
            }
        }
    
    def process_directory(self, directory_path: str, recursive: bool = False, 
                         engine: str = "all", preprocess: bool = True) -> Dict[str, Any]:
        """Process all images in a directory"""
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Find all image files
        image_extensions = config.settings.allowed_extensions
        image_paths = []
        
        if recursive:
            for ext in image_extensions:
                image_paths.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in image_extensions:
                image_paths.extend(directory.glob(f"*{ext}"))
        
        image_paths = [str(path) for path in image_paths]
        
        if not image_paths:
            logger.warning(f"No image files found in {directory_path}")
            return {
                'total_images': 0,
                'successful': 0,
                'failed': 0,
                'results': [],
                'message': 'No image files found'
            }
        
        job_name = f"Directory: {directory.name} ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        return self.process_images_batch(image_paths, engine, preprocess, job_name)
    
    def _calculate_average_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate average confidence across all results"""
        confidences = []
        
        for result in results:
            if 'results' in result:
                for ocr_result in result['results']:
                    if 'confidence' in ocr_result:
                        confidences.append(ocr_result['confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def export_results(self, results: Dict[str, Any], output_path: str, 
                      format: str = "json") -> str:
        """Export batch processing results to file"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        elif format.lower() == "csv":
            import pandas as pd
            
            # Flatten results for CSV
            csv_data = []
            for result in results.get('results', []):
                if 'results' in result:
                    for ocr_result in result['results']:
                        csv_data.append({
                            'image_path': result['image_path'],
                            'image_filename': result['image_filename'],
                            'engine': ocr_result['engine'],
                            'extracted_text': ocr_result['extracted_text'],
                            'confidence': ocr_result['confidence'],
                            'processing_time': ocr_result['processing_time'],
                            'preprocessed': result.get('preprocessed', False)
                        })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(output_file, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Results exported to {output_file}")
        return str(output_file)
    
    def get_batch_job_status(self, job_id: int) -> Optional[Dict[str, Any]]:
        """Get status of a batch job"""
        try:
            jobs = db_manager.get_batch_jobs(limit=1000)
            for job in jobs:
                if job['id'] == job_id:
                    return job
            return None
        except Exception as e:
            logger.error(f"Failed to get batch job status: {e}")
            return None

def main():
    """Example usage of batch processor"""
    processor = BatchProcessor(max_workers=2)
    
    # Example: Process a directory
    try:
        results = processor.process_directory(
            directory_path="sample_images",
            recursive=True,
            engine="all",
            preprocess=True
        )
        
        print(f"Processed {results['total_images']} images")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
        print(f"Success rate: {results['summary']['success_rate']:.2%}")
        
        # Export results
        processor.export_results(results, "batch_results.json")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")

if __name__ == "__main__":
    main()
