"""
Database models and operations for OCR results storage
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import List, Optional, Dict, Any
import json
import logging
from config import config

logger = logging.getLogger(__name__)

Base = declarative_base()

class OCRResultDB(Base):
    """Database model for OCR results"""
    __tablename__ = "ocr_results"
    
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String, nullable=False, index=True)
    image_filename = Column(String, nullable=False)
    engine = Column(String, nullable=False, index=True)
    extracted_text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    processing_time = Column(Float, nullable=False)
    bounding_boxes = Column(JSON, nullable=True)
    metadata = Column(JSON, nullable=True)
    preprocessed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class BatchJobDB(Base):
    """Database model for batch processing jobs"""
    __tablename__ = "batch_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    job_name = Column(String, nullable=False)
    total_images = Column(Integer, nullable=False)
    processed_images = Column(Integer, default=0)
    successful_extractions = Column(Integer, default=0)
    failed_extractions = Column(Integer, default=0)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

class DatabaseManager:
    """Database manager for OCR operations"""
    
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database connection"""
        try:
            database_url = config.settings.database_url
            self.engine = create_engine(database_url, echo=config.settings.debug_mode)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def save_ocr_result(self, result_data: Dict[str, Any]) -> int:
        """Save OCR result to database"""
        session = self.get_session()
        try:
            ocr_result = OCRResultDB(
                image_path=result_data.get('image_path'),
                image_filename=result_data.get('image_filename'),
                engine=result_data.get('engine'),
                extracted_text=result_data.get('extracted_text'),
                confidence=result_data.get('confidence'),
                processing_time=result_data.get('processing_time'),
                bounding_boxes=result_data.get('bounding_boxes'),
                metadata=result_data.get('metadata'),
                preprocessed=result_data.get('preprocessed', False)
            )
            
            session.add(ocr_result)
            session.commit()
            session.refresh(ocr_result)
            
            logger.info(f"OCR result saved with ID: {ocr_result.id}")
            return ocr_result.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save OCR result: {e}")
            raise
        finally:
            session.close()
    
    def get_ocr_results(self, limit: int = 100, offset: int = 0, 
                       engine: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get OCR results from database"""
        session = self.get_session()
        try:
            query = session.query(OCRResultDB)
            
            if engine:
                query = query.filter(OCRResultDB.engine == engine)
            
            results = query.order_by(OCRResultDB.created_at.desc()).offset(offset).limit(limit).all()
            
            return [self._result_to_dict(result) for result in results]
            
        except Exception as e:
            logger.error(f"Failed to get OCR results: {e}")
            raise
        finally:
            session.close()
    
    def get_result_by_id(self, result_id: int) -> Optional[Dict[str, Any]]:
        """Get OCR result by ID"""
        session = self.get_session()
        try:
            result = session.query(OCRResultDB).filter(OCRResultDB.id == result_id).first()
            return self._result_to_dict(result) if result else None
            
        except Exception as e:
            logger.error(f"Failed to get OCR result by ID: {e}")
            raise
        finally:
            session.close()
    
    def search_results(self, search_term: str, engine: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search OCR results by text content"""
        session = self.get_session()
        try:
            query = session.query(OCRResultDB).filter(
                OCRResultDB.extracted_text.contains(search_term)
            )
            
            if engine:
                query = query.filter(OCRResultDB.engine == engine)
            
            results = query.order_by(OCRResultDB.created_at.desc()).all()
            return [self._result_to_dict(result) for result in results]
            
        except Exception as e:
            logger.error(f"Failed to search OCR results: {e}")
            raise
        finally:
            session.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get OCR processing statistics"""
        session = self.get_session()
        try:
            total_results = session.query(OCRResultDB).count()
            
            # Results by engine
            engine_stats = {}
            engines = session.query(OCRResultDB.engine).distinct().all()
            for engine_tuple in engines:
                engine = engine_tuple[0]
                count = session.query(OCRResultDB).filter(OCRResultDB.engine == engine).count()
                avg_confidence = session.query(OCRResultDB.confidence).filter(
                    OCRResultDB.engine == engine
                ).all()
                avg_conf = sum([r[0] for r in avg_confidence]) / len(avg_confidence) if avg_confidence else 0
                
                engine_stats[engine] = {
                    'count': count,
                    'average_confidence': avg_conf
                }
            
            # Average processing time
            processing_times = session.query(OCRResultDB.processing_time).all()
            avg_processing_time = sum([r[0] for r in processing_times]) / len(processing_times) if processing_times else 0
            
            return {
                'total_results': total_results,
                'engine_statistics': engine_stats,
                'average_processing_time': avg_processing_time,
                'total_engines': len(engines)
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            raise
        finally:
            session.close()
    
    def create_batch_job(self, job_name: str, total_images: int) -> int:
        """Create a new batch job"""
        session = self.get_session()
        try:
            batch_job = BatchJobDB(
                job_name=job_name,
                total_images=total_images
            )
            
            session.add(batch_job)
            session.commit()
            session.refresh(batch_job)
            
            logger.info(f"Batch job created with ID: {batch_job.id}")
            return batch_job.id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to create batch job: {e}")
            raise
        finally:
            session.close()
    
    def update_batch_job(self, job_id: int, **kwargs) -> bool:
        """Update batch job status"""
        session = self.get_session()
        try:
            job = session.query(BatchJobDB).filter(BatchJobDB.id == job_id).first()
            if not job:
                return False
            
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update batch job: {e}")
            raise
        finally:
            session.close()
    
    def get_batch_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get batch jobs"""
        session = self.get_session()
        try:
            jobs = session.query(BatchJobDB).order_by(BatchJobDB.created_at.desc()).limit(limit).all()
            return [self._batch_job_to_dict(job) for job in jobs]
            
        except Exception as e:
            logger.error(f"Failed to get batch jobs: {e}")
            raise
        finally:
            session.close()
    
    def _result_to_dict(self, result: OCRResultDB) -> Dict[str, Any]:
        """Convert OCRResultDB to dictionary"""
        return {
            'id': result.id,
            'image_path': result.image_path,
            'image_filename': result.image_filename,
            'engine': result.engine,
            'extracted_text': result.extracted_text,
            'confidence': result.confidence,
            'processing_time': result.processing_time,
            'bounding_boxes': result.bounding_boxes,
            'metadata': result.metadata,
            'preprocessed': result.preprocessed,
            'created_at': result.created_at.isoformat(),
            'updated_at': result.updated_at.isoformat()
        }
    
    def _batch_job_to_dict(self, job: BatchJobDB) -> Dict[str, Any]:
        """Convert BatchJobDB to dictionary"""
        return {
            'id': job.id,
            'job_name': job.job_name,
            'total_images': job.total_images,
            'processed_images': job.processed_images,
            'successful_extractions': job.successful_extractions,
            'failed_extractions': job.failed_extractions,
            'status': job.status,
            'created_at': job.created_at.isoformat(),
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'error_message': job.error_message
        }

# Global database manager instance
db_manager = DatabaseManager()
