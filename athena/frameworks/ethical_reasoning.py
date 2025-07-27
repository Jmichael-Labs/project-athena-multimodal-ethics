"""
Ethical Reasoning Framework for Project Athena

Advanced ethical reasoning system integrating multiple ethical frameworks
(utilitarian, deontological, virtue ethics) with Meta AI integration.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

from ..core.evaluator import EvaluationResult, EthicsIssue, EthicsCategory, ComplianceStatus
from .utilitarian_framework import UtilitarianFramework
from .deontological_framework import DeontologicalFramework
from .virtue_ethics_framework import VirtueEthicsFramework

logger = logging.getLogger(__name__)

class EthicalFrameworkType(Enum):
    """Types of ethical frameworks."""
    UTILITARIAN = "utilitarian"
    DEONTOLOGICAL = "deontological"
    VIRTUE_ETHICS = "virtue_ethics"
    HYBRID = "hybrid"

@dataclass
class EthicalDecision:
    """Result of ethical reasoning process."""
    recommended_action: str
    framework_scores: Dict[EthicalFrameworkType, float] = field(default_factory=dict)
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0
    ethical_justification: str = ""
    potential_consequences: List[str] = field(default_factory=list)
    stakeholder_analysis: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EthicalDilemma:
    """Representation of an ethical dilemma."""
    scenario: str
    stakeholders: List[str]
    potential_actions: List[str]
    context: Dict[str, Any] = field(default_factory=dict)
    urgency: float = 0.5  # 0.0 to 1.0

class EthicalReasoning:
    """
    Comprehensive ethical reasoning system.
    
    Integrates multiple ethical frameworks to provide robust
    ethical decision-making and evaluation enhancement.
    """
    
    def __init__(self, config):
        """
        Initialize ethical reasoning system.
        
        Args:
            config: EthicsConfig instance
        """
        self.config = config
        
        # Initialize individual frameworks
        self.utilitarian = UtilitarianFramework(config)
        self.deontological = DeontologicalFramework(config)
        self.virtue_ethics = VirtueEthicsFramework(config)
        
        # Framework weights for hybrid decisions
        self.framework_weights = {
            EthicalFrameworkType.UTILITARIAN: 0.4,
            EthicalFrameworkType.DEONTOLOGICAL: 0.35,
            EthicalFrameworkType.VIRTUE_ETHICS: 0.25
        }
        
        # Decision history
        self.decision_history = []
        self.framework_performance = {
            EthicalFrameworkType.UTILITARIAN: {"correct_predictions": 0, "total_predictions": 0},
            EthicalFrameworkType.DEONTOLOGICAL: {"correct_predictions": 0, "total_predictions": 0},
            EthicalFrameworkType.VIRTUE_ETHICS: {"correct_predictions": 0, "total_predictions": 0}
        }
        
        logger.info("Ethical Reasoning system initialized")
    
    async def enhance_evaluation(
        self, 
        evaluation_result: EvaluationResult, 
        content: Any
    ) -> EvaluationResult:
        """
        Enhance evaluation using comprehensive ethical reasoning.
        
        Args:
            evaluation_result: Original evaluation result
            content: Content that was evaluated
        
        Returns:
            Enhanced evaluation result with ethical reasoning
        """
        try:
            # Create ethical dilemma from evaluation
            dilemma = self._create_ethical_dilemma(evaluation_result, content)
            
            # Perform ethical reasoning
            decision = await self._perform_ethical_reasoning(dilemma)
            
            # Apply reasoning to enhance evaluation
            enhanced_result = self._apply_ethical_decision(evaluation_result, decision)
            
            # Add reasoning metadata
            enhanced_result.metadata["ethical_reasoning_applied"] = True
            enhanced_result.metadata["ethical_decision"] = decision.__dict__
            enhanced_result.metadata["dominant_framework"] = self._get_dominant_framework(decision)
            
            self.decision_history.append(decision)
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Ethical reasoning enhancement failed: {e}")
            return evaluation_result
    
    def _create_ethical_dilemma(
        self, 
        evaluation_result: EvaluationResult, 
        content: Any
    ) -> EthicalDilemma:
        """Create an ethical dilemma from evaluation result."""
        
        # Determine scenario description
        scenario = f"Content evaluation with score {evaluation_result.overall_score:.2f} "
        scenario += f"and {len(evaluation_result.issues)} ethical issues"
        
        if evaluation_result.issues:
            major_categories = list(set(issue.category.value for issue in evaluation_result.issues))
            scenario += f" involving: {', '.join(major_categories)}"
        
        # Identify stakeholders
        stakeholders = ["content_users", "platform_operators", "society", "content_creators"]
        
        if any(issue.category == EthicsCategory.CHILD_SAFETY for issue in evaluation_result.issues):
            stakeholders.append("children_and_families")
        
        if any(issue.category == EthicsCategory.PRIVACY_VIOLATION for issue in evaluation_result.issues):
            stakeholders.append("privacy_advocates")
        
        # Define potential actions
        potential_actions = [
            "approve_content_as_is",
            "approve_with_warnings",
            "request_modifications",
            "reject_content",
            "escalate_for_human_review"
        ]
        
        # Determine urgency based on severity
        max_severity = max([issue.severity for issue in evaluation_result.issues], default=0.0)
        urgency = min(max_severity * 1.2, 1.0)
        
        return EthicalDilemma(
            scenario=scenario,
            stakeholders=stakeholders,
            potential_actions=potential_actions,
            context={
                "evaluation_result": evaluation_result,
                "content_type": type(content).__name__ if content else "unknown"
            },
            urgency=urgency
        )
    
    async def _perform_ethical_reasoning(self, dilemma: EthicalDilemma) -> EthicalDecision:
        """Perform comprehensive ethical reasoning on dilemma."""
        
        decision = EthicalDecision(recommended_action="")
        
        try:
            # Analyze using each framework
            analysis_tasks = [
                self._analyze_utilitarian(dilemma),
                self._analyze_deontological(dilemma),
                self._analyze_virtue_ethics(dilemma)
            ]
            
            analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Process framework results
            if not isinstance(analyses[0], Exception):
                decision.framework_scores[EthicalFrameworkType.UTILITARIAN] = analyses[0]["score"]
                decision.reasoning_chain.extend(analyses[0]["reasoning"])
            
            if not isinstance(analyses[1], Exception):
                decision.framework_scores[EthicalFrameworkType.DEONTOLOGICAL] = analyses[1]["score"]
                decision.reasoning_chain.extend(analyses[1]["reasoning"])
            
            if not isinstance(analyses[2], Exception):
                decision.framework_scores[EthicalFrameworkType.VIRTUE_ETHICS] = analyses[2]["score"]
                decision.reasoning_chain.extend(analyses[2]["reasoning"])
            
            # Make hybrid decision
            decision.recommended_action = self._make_hybrid_decision(decision, dilemma)
            decision.confidence = self._calculate_decision_confidence(decision)
            decision.ethical_justification = self._generate_ethical_justification(decision, dilemma)
            decision.potential_consequences = self._analyze_consequences(decision, dilemma)
            decision.stakeholder_analysis = self._analyze_stakeholders(decision, dilemma)
            
        except Exception as e:
            logger.error(f"Ethical reasoning failed: {e}")
            decision.recommended_action = "escalate_for_human_review"
            decision.confidence = 0.0
            decision.reasoning_chain.append(f"Reasoning failed: {e}")
        
        return decision
    
    async def _analyze_utilitarian(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Analyze dilemma using utilitarian framework."""
        return await self.utilitarian.analyze_dilemma(dilemma)
    
    async def _analyze_deontological(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Analyze dilemma using deontological framework."""
        return await self.deontological.analyze_dilemma(dilemma)
    
    async def _analyze_virtue_ethics(self, dilemma: EthicalDilemma) -> Dict[str, Any]:
        """Analyze dilemma using virtue ethics framework."""
        return await self.virtue_ethics.analyze_dilemma(dilemma)
    
    def _make_hybrid_decision(
        self, 
        decision: EthicalDecision, 
        dilemma: EthicalDilemma
    ) -> str:
        """Make final decision using hybrid approach."""
        
        # Calculate weighted average of framework scores
        weighted_score = 0.0
        total_weight = 0.0
        
        for framework_type, score in decision.framework_scores.items():
            weight = self.framework_weights.get(framework_type, 0.0)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_score /= total_weight
        
        # Make decision based on weighted score and urgency
        evaluation_result = dilemma.context.get("evaluation_result")
        
        if weighted_score < 0.3 or dilemma.urgency > 0.8:
            return "reject_content"
        elif weighted_score < 0.5:
            return "request_modifications"
        elif weighted_score < 0.7:
            return "approve_with_warnings"
        elif weighted_score < 0.9:
            return "approve_content_as_is"
        else:
            return "approve_content_as_is"
    
    def _calculate_decision_confidence(self, decision: EthicalDecision) -> float:
        """Calculate confidence in the ethical decision."""
        
        confidence_factors = []
        
        # Factor 1: Agreement between frameworks
        if len(decision.framework_scores) > 1:
            scores = list(decision.framework_scores.values())
            agreement = 1.0 - np.std(scores)  # Higher agreement = lower std dev
            confidence_factors.append(max(0.0, agreement))
        
        # Factor 2: Number of frameworks that provided scores
        framework_coverage = len(decision.framework_scores) / 3.0  # We have 3 frameworks
        confidence_factors.append(framework_coverage)
        
        # Factor 3: Clarity of reasoning
        reasoning_clarity = min(len(decision.reasoning_chain) / 5.0, 1.0)
        confidence_factors.append(reasoning_clarity)
        
        # Factor 4: Extremity of scores (more extreme = higher confidence)
        if decision.framework_scores:
            avg_score = np.mean(list(decision.framework_scores.values()))
            extremity = abs(avg_score - 0.5) * 2  # Distance from neutral (0.5)
            confidence_factors.append(extremity)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _generate_ethical_justification(
        self, 
        decision: EthicalDecision, 
        dilemma: EthicalDilemma
    ) -> str:
        """Generate ethical justification for the decision."""
        
        justification_parts = []
        
        # Framework-based justification
        for framework_type, score in decision.framework_scores.items():
            framework_name = framework_type.value.replace("_", " ").title()
            justification_parts.append(
                f"{framework_name} analysis yields score {score:.2f}"
            )
        
        # Decision rationale
        action_rationales = {
            "approve_content_as_is": "Content meets ethical standards for unrestricted distribution",
            "approve_with_warnings": "Content is acceptable but requires user warnings",
            "request_modifications": "Content has addressable ethical concerns",
            "reject_content": "Content violates fundamental ethical principles",
            "escalate_for_human_review": "Situation requires human ethical judgment"
        }
        
        rationale = action_rationales.get(
            decision.recommended_action, 
            "Action determined by ethical analysis"
        )
        justification_parts.append(rationale)
        
        # Urgency consideration
        if dilemma.urgency > 0.7:
            justification_parts.append("High urgency requires immediate action")
        
        return ". ".join(justification_parts)
    
    def _analyze_consequences(
        self, 
        decision: EthicalDecision, 
        dilemma: EthicalDilemma
    ) -> List[str]:
        """Analyze potential consequences of the decision."""
        
        consequences = []
        action = decision.recommended_action
        
        consequence_mapping = {
            "approve_content_as_is": [
                "Content reaches intended audience without restrictions",
                "Platform maintains user engagement",
                "Potential for unforeseen negative impacts if analysis is incorrect"
            ],
            "approve_with_warnings": [
                "Users receive important safety information",
                "Content remains accessible with informed consent",
                "May reduce engagement but increases safety"
            ],
            "request_modifications": [
                "Content creator has opportunity to address issues",
                "Improved version better serves all stakeholders",
                "Delays content distribution"
            ],
            "reject_content": [
                "Prevents potential harm to users and society",
                "Content creator may lose investment in content",
                "Platform avoids liability and reputational damage"
            ],
            "escalate_for_human_review": [
                "Ensures complex ethical issues receive proper attention",
                "Increases processing time and costs",
                "May identify nuances missed by automated analysis"
            ]
        }
        
        return consequence_mapping.get(action, ["Consequences require further analysis"])
    
    def _analyze_stakeholders(
        self, 
        decision: EthicalDecision, 
        dilemma: EthicalDilemma
    ) -> Dict[str, Any]:
        """Analyze impact on different stakeholders."""
        
        stakeholder_impact = {}
        action = decision.recommended_action
        
        # Define impact scoring (-1 to 1, negative = harmful, positive = beneficial)
        impact_matrix = {
            "approve_content_as_is": {
                "content_users": 0.5,
                "platform_operators": 0.7,
                "society": 0.3,
                "content_creators": 0.8
            },
            "approve_with_warnings": {
                "content_users": 0.7,
                "platform_operators": 0.5,
                "society": 0.6,
                "content_creators": 0.3
            },
            "request_modifications": {
                "content_users": 0.6,
                "platform_operators": 0.2,
                "society": 0.7,
                "content_creators": -0.2
            },
            "reject_content": {
                "content_users": 0.8,
                "platform_operators": 0.4,
                "society": 0.9,
                "content_creators": -0.8
            },
            "escalate_for_human_review": {
                "content_users": 0.5,
                "platform_operators": -0.3,
                "society": 0.6,
                "content_creators": -0.1
            }
        }
        
        action_impacts = impact_matrix.get(action, {})
        
        for stakeholder in dilemma.stakeholders:
            impact_score = action_impacts.get(stakeholder, 0.0)
            
            if impact_score > 0.5:
                impact_description = "Highly beneficial"
            elif impact_score > 0.0:
                impact_description = "Beneficial"
            elif impact_score == 0.0:
                impact_description = "Neutral"
            elif impact_score > -0.5:
                impact_description = "Slightly harmful"
            else:
                impact_description = "Significantly harmful"
            
            stakeholder_impact[stakeholder] = {
                "impact_score": impact_score,
                "description": impact_description
            }
        
        return stakeholder_impact
    
    def _apply_ethical_decision(
        self, 
        evaluation_result: EvaluationResult, 
        decision: EthicalDecision
    ) -> EvaluationResult:
        """Apply ethical decision to enhance evaluation result."""
        
        enhanced_result = EvaluationResult(
            overall_score=evaluation_result.overall_score,
            compliance_status=evaluation_result.compliance_status,
            issues=evaluation_result.issues.copy(),
            modality_scores=evaluation_result.modality_scores.copy(),
            evaluation_time=evaluation_result.evaluation_time,
            metadata=evaluation_result.metadata.copy()
        )
        
        # Adjust based on ethical decision
        action = decision.recommended_action
        
        if action == "reject_content":
            enhanced_result.overall_score = min(enhanced_result.overall_score, 0.3)
            enhanced_result.compliance_status = ComplianceStatus.CRITICAL
        elif action == "request_modifications":
            enhanced_result.overall_score = min(enhanced_result.overall_score, 0.6)
            if enhanced_result.compliance_status == ComplianceStatus.COMPLIANT:
                enhanced_result.compliance_status = ComplianceStatus.WARNING
        elif action == "approve_with_warnings":
            if enhanced_result.compliance_status == ComplianceStatus.COMPLIANT:
                enhanced_result.compliance_status = ComplianceStatus.WARNING
        elif action == "escalate_for_human_review":
            # Add metadata flag for human review
            enhanced_result.metadata["requires_human_review"] = True
            enhanced_result.metadata["escalation_reason"] = "Ethical complexity requires human judgment"
        
        # Adjust confidence based on ethical reasoning confidence
        enhanced_result.metadata["ethical_confidence"] = decision.confidence
        
        return enhanced_result
    
    def _get_dominant_framework(self, decision: EthicalDecision) -> str:
        """Identify the dominant ethical framework in the decision."""
        
        if not decision.framework_scores:
            return "none"
        
        # Find framework with highest weighted contribution
        max_contribution = 0.0
        dominant_framework = "hybrid"
        
        for framework_type, score in decision.framework_scores.items():
            weight = self.framework_weights.get(framework_type, 0.0)
            contribution = score * weight
            
            if contribution > max_contribution:
                max_contribution = contribution
                dominant_framework = framework_type.value
        
        return dominant_framework
    
    def update_framework_weights(self, feedback: Dict[EthicalFrameworkType, float]) -> None:
        """Update framework weights based on performance feedback."""
        
        # Update performance tracking
        for framework_type, performance in feedback.items():
            if framework_type in self.framework_performance:
                self.framework_performance[framework_type]["total_predictions"] += 1
                if performance > 0.5:  # Consider >0.5 as correct
                    self.framework_performance[framework_type]["correct_predictions"] += 1
        
        # Recalculate weights based on performance
        total_accuracy = 0.0
        framework_accuracies = {}
        
        for framework_type, stats in self.framework_performance.items():
            if stats["total_predictions"] > 0:
                accuracy = stats["correct_predictions"] / stats["total_predictions"]
                framework_accuracies[framework_type] = accuracy
                total_accuracy += accuracy
        
        # Normalize weights
        if total_accuracy > 0:
            for framework_type in self.framework_weights:
                if framework_type in framework_accuracies:
                    self.framework_weights[framework_type] = (
                        framework_accuracies[framework_type] / total_accuracy
                    )
        
        logger.info(f"Updated framework weights: {self.framework_weights}")
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """Get statistics about ethical reasoning performance."""
        
        return {
            "total_decisions": len(self.decision_history),
            "framework_performance": self.framework_performance.copy(),
            "current_weights": self.framework_weights.copy(),
            "avg_confidence": np.mean([d.confidence for d in self.decision_history]) if self.decision_history else 0.0,
            "decision_distribution": self._get_decision_distribution()
        }
    
    def _get_decision_distribution(self) -> Dict[str, int]:
        """Get distribution of decision types."""
        
        distribution = {}
        for decision in self.decision_history:
            action = decision.recommended_action
            distribution[action] = distribution.get(action, 0) + 1
        
        return distribution
    
    async def shutdown(self) -> None:
        """Gracefully shutdown ethical reasoning system."""
        logger.info("Shutting down Ethical Reasoning system...")
        
        # Shutdown individual frameworks
        await self.utilitarian.shutdown()
        await self.deontological.shutdown()
        await self.virtue_ethics.shutdown()
        
        logger.info("Ethical Reasoning system shutdown complete")