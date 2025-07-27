"""
Text Ethics Analyzer for Project Athena

Advanced NLP-based ethical evaluation for text content with
Meta Llama integration and sophisticated bias detection.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import json

# NLP and ML imports
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        pipeline, AutoModel
    )
    import spacy
    from textblob import TextBlob
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
except ImportError as e:
    logging.warning(f"Some NLP dependencies not available: {e}")

from ..core.evaluator import EthicsIssue, EthicsCategory

logger = logging.getLogger(__name__)

@dataclass
class TextAnalysisResult:
    """Result of comprehensive text analysis."""
    toxicity_score: float = 0.0
    bias_score: float = 0.0
    hate_speech_score: float = 0.0
    misinformation_score: float = 0.0
    sentiment_analysis: Dict[str, float] = field(default_factory=dict)
    linguistic_features: Dict[str, Any] = field(default_factory=dict)
    cultural_sensitivity: Dict[str, float] = field(default_factory=dict)
    detected_entities: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[EthicsIssue] = field(default_factory=list)

class TextEthics:
    """
    Advanced text ethics analyzer with Meta Llama integration.
    
    Provides comprehensive ethical evaluation of text content including
    toxicity detection, bias analysis, hate speech detection, and
    cultural sensitivity assessment.
    """
    
    def __init__(self, config):
        """
        Initialize text ethics analyzer.
        
        Args:
            config: EthicsConfig instance
        """
        self.config = config
        self.model_cache = {}
        self.linguistic_patterns = self._load_linguistic_patterns()
        
        # Initialize NLP models
        self._initialize_models()
        
        # Load bias lexicons and cultural sensitivity data
        self._load_bias_lexicons()
        self._load_cultural_sensitivity_data()
        
        logger.info("Text Ethics analyzer initialized")
    
    def _initialize_models(self) -> None:
        """Initialize NLP models for text analysis."""
        try:
            # Toxicity detection model
            self.toxicity_model = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Hate speech detection model
            self.hate_speech_model = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-hate-latest",
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Sentiment analyzer
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
            except:
                self.sentiment_analyzer = None
            
            # SpaCy model for entity recognition
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("SpaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Meta Llama integration (placeholder for actual Meta integration)
            if self.config.meta_integration.use_pytorch_integration:
                self._initialize_meta_models()
            
        except Exception as e:
            logger.error(f"Error initializing text models: {e}")
            self._initialize_fallback_models()
    
    def _initialize_meta_models(self) -> None:
        """Initialize Meta-specific models for enhanced analysis."""
        # This would integrate with actual Meta Llama models
        # For now, using placeholder implementation
        logger.info("Meta integration models initialized (placeholder)")
    
    def _initialize_fallback_models(self) -> None:
        """Initialize fallback models if primary models fail."""
        logger.warning("Using fallback text analysis models")
        self.toxicity_model = None
        self.hate_speech_model = None
        self.sentiment_analyzer = None
        self.nlp = None
    
    def _load_linguistic_patterns(self) -> Dict[str, List[str]]:
        """Load linguistic patterns for bias detection."""
        return {
            "gender_bias": [
                r'\b(he|she)\s+is\s+(emotional|irrational|hysterical)',
                r'\b(men|women)\s+are\s+(naturally|obviously|typically)',
                r'\b(boys|girls)\s+should\s+(not|never)',
            ],
            "racial_bias": [
                r'\b(all|most)\s+[A-Z][a-z]+\s+(people|men|women)\s+are',
                r'\b(typical|stereotypical)\s+[A-Z][a-z]+\s+(behavior|attitude)',
            ],
            "age_bias": [
                r'\b(young|old)\s+people\s+(are|can\'t|cannot)',
                r'\b(millennials|boomers)\s+are\s+(lazy|entitled|stubborn)',
            ],
            "religious_bias": [
                r'\b(all|most)\s+(christians|muslims|jews|hindus|buddhists)\s+are',
                r'\b(typical|stereotypical)\s+(christian|muslim|jewish|hindu|buddhist)',
            ]
        }
    
    def _load_bias_lexicons(self) -> None:
        """Load bias detection lexicons and word lists."""
        self.bias_lexicons = {
            "gender_terms": {
                "male_coded": ["aggressive", "ambitious", "analytical", "assertive", "athletic"],
                "female_coded": ["affectionate", "cheerful", "childlike", "compassionate", "considerate"]
            },
            "stereotype_indicators": [
                "obviously", "naturally", "typically", "clearly", "certainly",
                "all women", "all men", "typical woman", "typical man"
            ]
        }
    
    def _load_cultural_sensitivity_data(self) -> None:
        """Load cultural sensitivity indicators and guidelines."""
        self.cultural_indicators = {
            "potentially_insensitive": [
                "exotic", "primitive", "civilized", "backwards", "third world",
                "tribal", "savage", "barbarian", "uncivilized"
            ],
            "religious_sensitivity": [
                "cult", "extremist", "radical", "fundamentalist"
            ],
            "cultural_appropriation": [
                "spirit animal", "tribe", "pow wow", "gypsy"
            ]
        }
    
    async def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> TextAnalysisResult:
        """
        Perform comprehensive ethical analysis of text content.
        
        Args:
            text: Text content to analyze
            metadata: Optional metadata for context
        
        Returns:
            TextAnalysisResult: Comprehensive analysis result
        """
        if not text or not text.strip():
            return TextAnalysisResult()
        
        result = TextAnalysisResult()
        
        try:
            # Perform parallel analysis
            analysis_tasks = [
                self._analyze_toxicity(text),
                self._analyze_bias(text),
                self._analyze_hate_speech(text),
                self._analyze_misinformation(text),
                self._analyze_sentiment(text),
                self._analyze_linguistic_features(text),
                self._analyze_cultural_sensitivity(text),
                self._extract_entities(text)
            ]
            
            analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process results
            result.toxicity_score = analyses[0] if not isinstance(analyses[0], Exception) else 0.0
            result.bias_score = analyses[1] if not isinstance(analyses[1], Exception) else 0.0
            result.hate_speech_score = analyses[2] if not isinstance(analyses[2], Exception) else 0.0
            result.misinformation_score = analyses[3] if not isinstance(analyses[3], Exception) else 0.0
            result.sentiment_analysis = analyses[4] if not isinstance(analyses[4], Exception) else {}
            result.linguistic_features = analyses[5] if not isinstance(analyses[5], Exception) else {}
            result.cultural_sensitivity = analyses[6] if not isinstance(analyses[6], Exception) else {}
            result.detected_entities = analyses[7] if not isinstance(analyses[7], Exception) else []
            
            # Generate ethics issues based on analysis
            result.issues = self._generate_ethics_issues(result, text)
            
            logger.debug(f"Text analysis completed. Issues found: {len(result.issues)}")
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            result.issues.append(EthicsIssue(
                category=EthicsCategory.HARMFUL_CONTENT,
                severity=0.5,
                confidence=0.8,
                description=f"Text analysis failed: {str(e)}",
                recommendation="Manual review required due to analysis failure"
            ))
        
        return result
    
    async def _analyze_toxicity(self, text: str) -> float:
        """Analyze text for toxic content."""
        if not self.toxicity_model:
            return await self._fallback_toxicity_analysis(text)
        
        try:
            results = self.toxicity_model(text)
            # Find toxic score from results
            for result in results[0]:
                if result['label'] in ['TOXIC', 'toxic', '1']:
                    return result['score']
            return 0.0
        except Exception as e:
            logger.warning(f"Toxicity analysis failed: {e}")
            return await self._fallback_toxicity_analysis(text)
    
    async def _analyze_bias(self, text: str) -> float:
        """Analyze text for various forms of bias."""
        bias_score = 0.0
        text_lower = text.lower()
        
        # Check for linguistic bias patterns
        for bias_type, patterns in self.linguistic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    bias_score += 0.2
        
        # Check for gendered language bias
        gendered_score = self._analyze_gendered_language(text)
        bias_score += gendered_score
        
        # Check for stereotype indicators
        for indicator in self.bias_lexicons["stereotype_indicators"]:
            if indicator in text_lower:
                bias_score += 0.1
        
        return min(bias_score, 1.0)
    
    async def _analyze_hate_speech(self, text: str) -> float:
        """Analyze text for hate speech."""
        if not self.hate_speech_model:
            return await self._fallback_hate_speech_analysis(text)
        
        try:
            results = self.hate_speech_model(text)
            # Find hate speech score from results
            for result in results[0]:
                if result['label'] in ['HATE', 'hate', 'OFFENSIVE']:
                    return result['score']
            return 0.0
        except Exception as e:
            logger.warning(f"Hate speech analysis failed: {e}")
            return await self._fallback_hate_speech_analysis(text)
    
    async def _analyze_misinformation(self, text: str) -> float:
        """Analyze text for potential misinformation indicators."""
        misinformation_score = 0.0
        text_lower = text.lower()
        
        # Check for absolute claims without evidence
        absolute_patterns = [
            r'\b(always|never|all|none|every|no one)\b.*\b(are|is|does|do)\b',
            r'\bproven fact\b',
            r'\bscientists say\b',
            r'\bstudies show\b'
        ]
        
        for pattern in absolute_patterns:
            if re.search(pattern, text_lower):
                misinformation_score += 0.15
        
        # Check for conspiracy indicators
        conspiracy_terms = [
            "cover up", "hidden truth", "they don't want you to know",
            "mainstream media lies", "wake up", "sheeple"
        ]
        
        for term in conspiracy_terms:
            if term in text_lower:
                misinformation_score += 0.2
        
        return min(misinformation_score, 1.0)
    
    async def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        if not self.sentiment_analyzer:
            return self._fallback_sentiment_analysis(text)
        
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return {
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu'],
                "compound": scores['compound']
            }
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return self._fallback_sentiment_analysis(text)
    
    async def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features of text."""
        features = {
            "word_count": len(text.split()),
            "sentence_count": len(text.split('.')),
            "avg_word_length": np.mean([len(word) for word in text.split()]) if text.split() else 0,
            "question_count": text.count('?'),
            "exclamation_count": text.count('!'),
            "capitalization_ratio": sum(1 for c in text if c.isupper()) / len(text) if text else 0
        }
        
        # Add complexity measures
        if self.nlp:
            try:
                doc = self.nlp(text)
                features["named_entities"] = len(doc.ents)
                features["pos_tags"] = len(set([token.pos_ for token in doc]))
            except:
                pass
        
        return features
    
    async def _analyze_cultural_sensitivity(self, text: str) -> Dict[str, float]:
        """Analyze text for cultural sensitivity issues."""
        sensitivity_scores = {
            "cultural_appropriation": 0.0,
            "religious_sensitivity": 0.0,
            "general_sensitivity": 0.0
        }
        
        text_lower = text.lower()
        
        # Check for cultural appropriation terms
        for term in self.cultural_indicators["cultural_appropriation"]:
            if term in text_lower:
                sensitivity_scores["cultural_appropriation"] += 0.3
        
        # Check for religious sensitivity
        for term in self.cultural_indicators["religious_sensitivity"]:
            if term in text_lower:
                sensitivity_scores["religious_sensitivity"] += 0.25
        
        # Check for generally insensitive terms
        for term in self.cultural_indicators["potentially_insensitive"]:
            if term in text_lower:
                sensitivity_scores["general_sensitivity"] += 0.2
        
        # Normalize scores
        for key in sensitivity_scores:
            sensitivity_scores[key] = min(sensitivity_scores[key], 1.0)
        
        return sensitivity_scores
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        entities = []
        
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "description": spacy.explain(ent.label_)
                    })
            except Exception as e:
                logger.warning(f"Entity extraction failed: {e}")
        
        return entities
    
    def _analyze_gendered_language(self, text: str) -> float:
        """Analyze for gendered language bias."""
        text_lower = text.lower()
        
        male_coded_count = sum(1 for term in self.bias_lexicons["gender_terms"]["male_coded"] 
                              if term in text_lower)
        female_coded_count = sum(1 for term in self.bias_lexicons["gender_terms"]["female_coded"] 
                                if term in text_lower)
        
        total_gendered = male_coded_count + female_coded_count
        if total_gendered == 0:
            return 0.0
        
        # Calculate imbalance
        imbalance = abs(male_coded_count - female_coded_count) / total_gendered
        return imbalance * 0.5  # Scale to reasonable range
    
    def _generate_ethics_issues(
        self, 
        analysis: TextAnalysisResult, 
        text: str
    ) -> List[EthicsIssue]:
        """Generate ethics issues based on analysis results."""
        issues = []
        
        # Toxicity issues
        if analysis.toxicity_score > self.config.ethics_thresholds.toxicity:
            issues.append(EthicsIssue(
                category=EthicsCategory.TOXICITY,
                severity=analysis.toxicity_score,
                confidence=0.85,
                description=f"High toxicity detected (score: {analysis.toxicity_score:.2f})",
                recommendation="Review and moderate toxic language"
            ))
        
        # Bias issues
        if analysis.bias_score > self.config.ethics_thresholds.bias_detection:
            issues.append(EthicsIssue(
                category=EthicsCategory.BIAS_DETECTION,
                severity=analysis.bias_score,
                confidence=0.75,
                description=f"Potential bias detected (score: {analysis.bias_score:.2f})",
                recommendation="Review for biased language and stereotypes"
            ))
        
        # Hate speech issues
        if analysis.hate_speech_score > self.config.ethics_thresholds.hate_speech:
            issues.append(EthicsIssue(
                category=EthicsCategory.HATE_SPEECH,
                severity=analysis.hate_speech_score,
                confidence=0.9,
                description=f"Hate speech detected (score: {analysis.hate_speech_score:.2f})",
                recommendation="Remove or modify hate speech content"
            ))
        
        # Misinformation issues
        if analysis.misinformation_score > self.config.ethics_thresholds.misinformation:
            issues.append(EthicsIssue(
                category=EthicsCategory.MISINFORMATION,
                severity=analysis.misinformation_score,
                confidence=0.7,
                description=f"Potential misinformation (score: {analysis.misinformation_score:.2f})",
                recommendation="Verify claims and provide sources"
            ))
        
        # Cultural sensitivity issues
        for category, score in analysis.cultural_sensitivity.items():
            if score > 0.5:
                issues.append(EthicsIssue(
                    category=EthicsCategory.HARMFUL_CONTENT,
                    severity=score,
                    confidence=0.65,
                    description=f"Cultural sensitivity concern: {category} (score: {score:.2f})",
                    recommendation="Review for cultural appropriateness"
                ))
        
        return issues
    
    # Fallback methods for when models are not available
    
    async def _fallback_toxicity_analysis(self, text: str) -> float:
        """Fallback toxicity analysis using keyword matching."""
        toxic_keywords = [
            "hate", "kill", "die", "stupid", "idiot", "retard", "loser",
            "shut up", "worthless", "pathetic", "disgusting"
        ]
        
        text_lower = text.lower()
        toxic_count = sum(1 for keyword in toxic_keywords if keyword in text_lower)
        return min(toxic_count * 0.2, 1.0)
    
    async def _fallback_hate_speech_analysis(self, text: str) -> float:
        """Fallback hate speech analysis using keyword matching."""
        hate_keywords = [
            "racial slurs", "homophobic terms", "transphobic terms",
            # Note: Actual implementation would include specific terms
            # but avoiding them here for safety
        ]
        
        # This is a simplified fallback - real implementation would be more sophisticated
        return 0.0
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict[str, float]:
        """Fallback sentiment analysis using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0:
                return {"positive": polarity, "negative": 0, "neutral": 1-polarity, "compound": polarity}
            elif polarity < 0:
                return {"positive": 0, "negative": abs(polarity), "neutral": 1-abs(polarity), "compound": polarity}
            else:
                return {"positive": 0, "negative": 0, "neutral": 1.0, "compound": 0}
        except:
            return {"positive": 0, "negative": 0, "neutral": 1.0, "compound": 0}