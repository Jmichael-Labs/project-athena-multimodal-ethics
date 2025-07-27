"""
Image Ethics Analyzer for Project Athena

Advanced computer vision-based ethical evaluation for image content
with Meta integration and comprehensive safety assessment.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from pathlib import Path
import io
import base64

# Computer Vision and ML imports
try:
    import torch
    import torchvision.transforms as transforms
    from transformers import (
        AutoFeatureExtractor, AutoModelForImageClassification,
        CLIPProcessor, CLIPModel, pipeline
    )
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    import face_recognition
except ImportError as e:
    logging.warning(f"Some CV dependencies not available: {e}")

from ..core.evaluator import EthicsIssue, EthicsCategory

logger = logging.getLogger(__name__)

@dataclass
class ImageAnalysisResult:
    """Result of comprehensive image analysis."""
    violence_score: float = 0.0
    sexual_content_score: float = 0.0
    child_safety_score: float = 0.0
    hate_symbols_score: float = 0.0
    privacy_violation_score: float = 0.0
    copyright_risk_score: float = 0.0
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    detected_faces: List[Dict[str, Any]] = field(default_factory=list)
    image_quality: Dict[str, float] = field(default_factory=dict)
    content_categories: Dict[str, float] = field(default_factory=dict)
    metadata_analysis: Dict[str, Any] = field(default_factory=dict)
    issues: List[EthicsIssue] = field(default_factory=list)

@dataclass
class DetectedObject:
    """Information about a detected object in the image."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DetectedFace:
    """Information about a detected face in the image."""
    bbox: Tuple[int, int, int, int]
    confidence: float
    age_estimate: Optional[int] = None
    gender_estimate: Optional[str] = None
    emotion: Optional[str] = None
    identifiable: bool = False

class ImageEthics:
    """
    Advanced image ethics analyzer with Meta integration.
    
    Provides comprehensive ethical evaluation of image content including
    violence detection, sexual content analysis, child safety assessment,
    and privacy violation detection.
    """
    
    def __init__(self, config):
        """
        Initialize image ethics analyzer.
        
        Args:
            config: EthicsConfig instance
        """
        self.config = config
        self.model_cache = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize computer vision models
        self._initialize_models()
        
        # Load ethical content databases
        self._load_content_databases()
        
        # Setup image preprocessing
        self._setup_preprocessing()
        
        logger.info("Image Ethics analyzer initialized")
    
    def _initialize_models(self) -> None:
        """Initialize computer vision models for image analysis."""
        try:
            # CLIP model for general image understanding
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            
            # NSFW detection model
            self.nsfw_detector = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Violence detection model
            self.violence_detector = pipeline(
                "image-classification", 
                model="nateraw/vit-age-classifier",  # Placeholder - would use violence-specific model
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Object detection model
            self.object_detector = pipeline(
                "object-detection",
                model="facebook/detr-resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Meta integration models (placeholder)
            if self.config.meta_integration.use_pytorch_integration:
                self._initialize_meta_models()
            
        except Exception as e:
            logger.error(f"Error initializing image models: {e}")
            self._initialize_fallback_models()
    
    def _initialize_meta_models(self) -> None:
        """Initialize Meta-specific models for enhanced analysis."""
        # This would integrate with actual Meta computer vision models
        # For now, using placeholder implementation
        logger.info("Meta CV integration models initialized (placeholder)")
    
    def _initialize_fallback_models(self) -> None:
        """Initialize fallback models if primary models fail."""
        logger.warning("Using fallback image analysis models")
        self.clip_processor = None
        self.clip_model = None
        self.nsfw_detector = None
        self.violence_detector = None
        self.object_detector = None
    
    def _load_content_databases(self) -> None:
        """Load databases for content classification."""
        self.content_databases = {
            "hate_symbols": [
                # Would contain actual hate symbol patterns/hashes
                "swastika", "confederate_flag", "kkk_symbols"
            ],
            "violence_indicators": [
                "weapon", "blood", "fight", "war", "explosion", "injury"
            ],
            "child_safety_keywords": [
                "child", "minor", "underage", "school", "playground"
            ],
            "inappropriate_content": [
                "nudity", "sexual", "explicit", "adult_content"
            ]
        }
    
    def _setup_preprocessing(self) -> None:
        """Setup image preprocessing pipelines."""
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.clip_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    
    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Load and standardize image input."""
        if isinstance(image_input, str):
            # File path
            return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # NumPy array
            return Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            # PIL Image
            return image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    async def analyze(
        self, 
        image_input: Union[str, np.ndarray, Image.Image], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> ImageAnalysisResult:
        """
        Perform comprehensive ethical analysis of image content.
        
        Args:
            image_input: Image to analyze (file path, array, or PIL Image)
            metadata: Optional metadata for context
        
        Returns:
            ImageAnalysisResult: Comprehensive analysis result
        """
        try:
            # Load and preprocess image
            image = self._load_image(image_input)
            result = ImageAnalysisResult()
            
            # Perform parallel analysis
            analysis_tasks = [
                self._analyze_violence(image),
                self._analyze_sexual_content(image),
                self._analyze_child_safety(image),
                self._analyze_hate_symbols(image),
                self._analyze_privacy_violations(image),
                self._analyze_copyright_risk(image),
                self._detect_objects(image),
                self._detect_faces(image),
                self._analyze_image_quality(image),
                self._classify_content(image),
                self._analyze_metadata(image, metadata)
            ]
            
            analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            result.violence_score = analyses[0] if not isinstance(analyses[0], Exception) else 0.0
            result.sexual_content_score = analyses[1] if not isinstance(analyses[1], Exception) else 0.0
            result.child_safety_score = analyses[2] if not isinstance(analyses[2], Exception) else 0.0
            result.hate_symbols_score = analyses[3] if not isinstance(analyses[3], Exception) else 0.0
            result.privacy_violation_score = analyses[4] if not isinstance(analyses[4], Exception) else 0.0
            result.copyright_risk_score = analyses[5] if not isinstance(analyses[5], Exception) else 0.0
            result.detected_objects = analyses[6] if not isinstance(analyses[6], Exception) else []
            result.detected_faces = analyses[7] if not isinstance(analyses[7], Exception) else []
            result.image_quality = analyses[8] if not isinstance(analyses[8], Exception) else {}
            result.content_categories = analyses[9] if not isinstance(analyses[9], Exception) else {}
            result.metadata_analysis = analyses[10] if not isinstance(analyses[10], Exception) else {}
            
            # Generate ethics issues based on analysis
            result.issues = self._generate_ethics_issues(result, image)
            
            logger.debug(f"Image analysis completed. Issues found: {len(result.issues)}")
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            result = ImageAnalysisResult()
            result.issues.append(EthicsIssue(
                category=EthicsCategory.HARMFUL_CONTENT,
                severity=0.5,
                confidence=0.8,
                description=f"Image analysis failed: {str(e)}",
                recommendation="Manual review required due to analysis failure"
            ))
        
        return result
    
    async def _analyze_violence(self, image: Image.Image) -> float:
        """Analyze image for violent content."""
        if not self.violence_detector:
            return await self._fallback_violence_analysis(image)
        
        try:
            # Use CLIP for violence detection
            if self.clip_model and self.clip_processor:
                violence_prompts = [
                    "a violent scene", "fighting", "weapons", "blood", "war",
                    "aggression", "conflict", "destruction", "injury"
                ]
                
                inputs = self.clip_processor(
                    text=violence_prompts, 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    max_violence_prob = torch.max(probs).item()
                
                return max_violence_prob
            
            return 0.1  # Default low score
            
        except Exception as e:
            logger.warning(f"Violence analysis failed: {e}")
            return await self._fallback_violence_analysis(image)
    
    async def _analyze_sexual_content(self, image: Image.Image) -> float:
        """Analyze image for sexual/NSFW content."""
        if not self.nsfw_detector:
            return await self._fallback_nsfw_analysis(image)
        
        try:
            results = self.nsfw_detector(image)
            
            # Find NSFW/sexual content score
            for result in results:
                if result['label'] in ['NSFW', 'sexual', 'explicit', 'porn']:
                    return result['score']
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"NSFW analysis failed: {e}")
            return await self._fallback_nsfw_analysis(image)
    
    async def _analyze_child_safety(self, image: Image.Image) -> float:
        """Analyze image for child safety concerns."""
        try:
            # Detect children in image
            child_indicators = 0.0
            
            # Use face detection to estimate ages
            faces = await self._detect_faces(image)
            for face in faces:
                if face.age_estimate and face.age_estimate < 18:
                    child_indicators += 0.3
            
            # Use object detection for child-related objects
            objects = await self._detect_objects(image)
            child_objects = ["person", "child", "toy", "school", "playground"]
            
            for obj in objects:
                if any(child_obj in obj.get('label', '').lower() for child_obj in child_objects):
                    child_indicators += 0.2
            
            return min(child_indicators, 1.0)
            
        except Exception as e:
            logger.warning(f"Child safety analysis failed: {e}")
            return 0.0
    
    async def _analyze_hate_symbols(self, image: Image.Image) -> float:
        """Analyze image for hate symbols and extremist content."""
        if not self.clip_model or not self.clip_processor:
            return 0.0
        
        try:
            hate_prompts = [
                "hate symbol", "swastika", "confederate flag", "extremist symbol",
                "racist imagery", "nazi symbol", "white supremacist"
            ]
            
            inputs = self.clip_processor(
                text=hate_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
                max_hate_prob = torch.max(probs).item()
            
            return max_hate_prob
            
        except Exception as e:
            logger.warning(f"Hate symbol analysis failed: {e}")
            return 0.0
    
    async def _analyze_privacy_violations(self, image: Image.Image) -> float:
        """Analyze image for privacy violations."""
        privacy_score = 0.0
        
        try:
            # Check for faces (potential privacy violation)
            faces = await self._detect_faces(image)
            if faces:
                privacy_score += len(faces) * 0.2
            
            # Check for text that might contain personal information
            # (Would use OCR in production)
            
            # Check for private settings/locations
            if self.clip_model and self.clip_processor:
                private_prompts = [
                    "private home", "bedroom", "bathroom", "personal documents",
                    "license plate", "address", "phone number"
                ]
                
                inputs = self.clip_processor(
                    text=private_prompts,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    max_private_prob = torch.max(probs).item()
                
                privacy_score += max_private_prob * 0.5
            
            return min(privacy_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Privacy analysis failed: {e}")
            return 0.0
    
    async def _analyze_copyright_risk(self, image: Image.Image) -> float:
        """Analyze image for potential copyright issues."""
        try:
            copyright_score = 0.0
            
            # Check for copyrighted content indicators
            if self.clip_model and self.clip_processor:
                copyright_prompts = [
                    "movie scene", "tv show", "cartoon character", "logo",
                    "brand", "copyrighted artwork", "famous painting"
                ]
                
                inputs = self.clip_processor(
                    text=copyright_prompts,
                    images=image,
                    return_tensors="pt",
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    max_copyright_prob = torch.max(probs).item()
                
                copyright_score = max_copyright_prob
            
            return copyright_score
            
        except Exception as e:
            logger.warning(f"Copyright analysis failed: {e}")
            return 0.0
    
    async def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in the image."""
        if not self.object_detector:
            return []
        
        try:
            results = self.object_detector(image)
            
            objects = []
            for result in results:
                objects.append({
                    'label': result['label'],
                    'score': result['score'],
                    'box': result['box']
                })
            
            return objects
            
        except Exception as e:
            logger.warning(f"Object detection failed: {e}")
            return []
    
    async def _detect_faces(self, image: Image.Image) -> List[DetectedFace]:
        """Detect faces in the image."""
        try:
            # Convert PIL to OpenCV format
            image_array = np.array(image)
            
            # Use face_recognition library if available
            try:
                import face_recognition
                face_locations = face_recognition.face_locations(image_array)
                
                faces = []
                for (top, right, bottom, left) in face_locations:
                    face = DetectedFace(
                        bbox=(left, top, right, bottom),
                        confidence=0.8,  # face_recognition doesn't provide confidence
                        identifiable=True  # Assume identifiable if detected
                    )
                    faces.append(face)
                
                return faces
                
            except ImportError:
                # Fallback to OpenCV Haar cascades
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                
                faces_cv = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                faces = []
                for (x, y, w, h) in faces_cv:
                    face = DetectedFace(
                        bbox=(x, y, x+w, y+h),
                        confidence=0.7,
                        identifiable=True
                    )
                    faces.append(face)
                
                return faces
                
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []
    
    async def _analyze_image_quality(self, image: Image.Image) -> Dict[str, float]:
        """Analyze image quality metrics."""
        try:
            image_array = np.array(image)
            
            quality_metrics = {
                "resolution": image.width * image.height,
                "aspect_ratio": image.width / image.height,
                "brightness": np.mean(image_array),
                "contrast": np.std(image_array),
                "sharpness": self._calculate_sharpness(image_array)
            }
            
            return quality_metrics
            
        except Exception as e:
            logger.warning(f"Image quality analysis failed: {e}")
            return {}
    
    def _calculate_sharpness(self, image_array: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except:
            return 0.0
    
    async def _classify_content(self, image: Image.Image) -> Dict[str, float]:
        """Classify image content into categories."""
        if not self.clip_model or not self.clip_processor:
            return {}
        
        try:
            content_categories = [
                "nature", "people", "animals", "food", "technology",
                "art", "architecture", "vehicles", "sports", "entertainment"
            ]
            
            inputs = self.clip_processor(
                text=content_categories,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1)
            
            categories = {}
            for i, category in enumerate(content_categories):
                categories[category] = probs[0][i].item()
            
            return categories
            
        except Exception as e:
            logger.warning(f"Content classification failed: {e}")
            return {}
    
    async def _analyze_metadata(
        self, 
        image: Image.Image, 
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze image metadata for additional context."""
        metadata_analysis = {}
        
        try:
            # Extract EXIF data
            exif_data = image.getexif()
            if exif_data:
                metadata_analysis["has_exif"] = True
                metadata_analysis["exif_tags"] = len(exif_data)
            else:
                metadata_analysis["has_exif"] = False
            
            # Add provided metadata
            if metadata:
                metadata_analysis["provided_metadata"] = metadata
            
            return metadata_analysis
            
        except Exception as e:
            logger.warning(f"Metadata analysis failed: {e}")
            return {}
    
    def _generate_ethics_issues(
        self, 
        analysis: ImageAnalysisResult, 
        image: Image.Image
    ) -> List[EthicsIssue]:
        """Generate ethics issues based on analysis results."""
        issues = []
        
        # Violence issues
        if analysis.violence_score > self.config.ethics_thresholds.violence:
            issues.append(EthicsIssue(
                category=EthicsCategory.VIOLENCE,
                severity=analysis.violence_score,
                confidence=0.85,
                description=f"Violent content detected (score: {analysis.violence_score:.2f})",
                recommendation="Review and potentially remove violent imagery"
            ))
        
        # Sexual content issues
        if analysis.sexual_content_score > self.config.ethics_thresholds.sexual_content:
            issues.append(EthicsIssue(
                category=EthicsCategory.SEXUAL_CONTENT,
                severity=analysis.sexual_content_score,
                confidence=0.9,
                description=f"Sexual/NSFW content detected (score: {analysis.sexual_content_score:.2f})",
                recommendation="Apply content warnings or age restrictions"
            ))
        
        # Child safety issues
        if analysis.child_safety_score > self.config.ethics_thresholds.child_safety:
            issues.append(EthicsIssue(
                category=EthicsCategory.CHILD_SAFETY,
                severity=analysis.child_safety_score,
                confidence=0.95,
                description=f"Child safety concerns detected (score: {analysis.child_safety_score:.2f})",
                recommendation="Immediate review required for child safety"
            ))
        
        # Hate symbols issues
        if analysis.hate_symbols_score > 0.3:  # Lower threshold for hate symbols
            issues.append(EthicsIssue(
                category=EthicsCategory.HATE_SPEECH,
                severity=analysis.hate_symbols_score,
                confidence=0.8,
                description=f"Potential hate symbols detected (score: {analysis.hate_symbols_score:.2f})",
                recommendation="Review for hate symbols and extremist content"
            ))
        
        # Privacy violation issues
        if analysis.privacy_violation_score > self.config.ethics_thresholds.privacy_violation:
            issues.append(EthicsIssue(
                category=EthicsCategory.PRIVACY_VIOLATION,
                severity=analysis.privacy_violation_score,
                confidence=0.75,
                description=f"Privacy concerns detected (score: {analysis.privacy_violation_score:.2f})",
                recommendation="Review for personal information and obtain consent"
            ))
        
        # Copyright issues
        if analysis.copyright_risk_score > self.config.ethics_thresholds.copyright_infringement:
            issues.append(EthicsIssue(
                category=EthicsCategory.COPYRIGHT_INFRINGEMENT,
                severity=analysis.copyright_risk_score,
                confidence=0.7,
                description=f"Potential copyright infringement (score: {analysis.copyright_risk_score:.2f})",
                recommendation="Verify usage rights and permissions"
            ))
        
        return issues
    
    # Fallback methods for when models are not available
    
    async def _fallback_violence_analysis(self, image: Image.Image) -> float:
        """Fallback violence analysis using basic image properties."""
        # Very basic analysis - would be enhanced in production
        return 0.05
    
    async def _fallback_nsfw_analysis(self, image: Image.Image) -> float:
        """Fallback NSFW analysis using basic image properties."""
        # Very basic analysis - would be enhanced in production
        return 0.03