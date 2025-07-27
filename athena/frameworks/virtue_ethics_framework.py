"""
Virtue Ethics Framework for Project Athena

Implementation of virtue ethics framework focusing on character traits,
moral virtues, and excellence in ethical decision-making.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class Virtue(Enum):
    """Core virtues in virtue ethics framework."""
    WISDOM = "wisdom"
    COURAGE = "courage"
    TEMPERANCE = "temperance"
    JUSTICE = "justice"
    COMPASSION = "compassion"
    HONESTY = "honesty"
    HUMILITY = "humility"
    PRUDENCE = "prudence"
    INTEGRITY = "integrity"
    RESPONSIBILITY = "responsibility"

@dataclass
class VirtueAssessment:
    """Assessment of how well an action embodies a virtue."""
    virtue: Virtue
    embodiment_score: float  # 0.0 to 1.0
    vice_score: float  # 0.0 to 1.0 (opposite of virtue)
    balance_score: float  # How well virtue is balanced (avoiding extremes)
    justification: str

class VirtueEthicsFramework:
    """
    Virtue ethics framework implementation.
    
    Evaluates actions based on how well they embody moral virtues
    and contribute to human flourishing (eudaimonia).
    """
    
    def __init__(self, config):
        """Initialize virtue ethics framework."""
        self.config = config
        
        # Define virtues and their characteristics
        self.virtue_definitions = {
            Virtue.WISDOM: {
                "description": "Good judgment and understanding in ethical matters",
                "weight": 1.0,
                "related_vices": ["foolishness", "ignorance", "recklessness"],
                "excess": "overthinking",
                "deficiency": "thoughtlessness"
            },
            Virtue.COURAGE: {
                "description": "Moral bravery to do what is right despite difficulties",
                "weight": 0.9,
                "related_vices": ["cowardice", "recklessness"],
                "excess": "rashness",
                "deficiency": "cowardice"
            },
            Virtue.TEMPERANCE: {
                "description": "Moderation and self-control in actions and decisions",
                "weight": 0.8,
                "related_vices": ["excess", "deficiency"],
                "excess": "over-restriction",
                "deficiency": "indulgence"
            },
            Virtue.JUSTICE: {
                "description": "Fairness and giving each their due",
                "weight": 1.0,
                "related_vices": ["unfairness", "bias", "discrimination"],
                "excess": "rigidity",
                "deficiency": "permissiveness"
            },
            Virtue.COMPASSION: {
                "description": "Empathy and care for the suffering of others",
                "weight": 0.9,
                "related_vices": ["cruelty", "indifference"],
                "excess": "enabling",
                "deficiency": "callousness"
            },
            Virtue.HONESTY: {
                "description": "Truthfulness and transparency in communication",
                "weight": 0.9,
                "related_vices": ["deception", "lying", "manipulation"],
                "excess": "brutal_honesty",
                "deficiency": "dishonesty"
            },
            Virtue.HUMILITY: {
                "description": "Modest view of one's importance and limitations",
                "weight": 0.7,
                "related_vices": ["arrogance", "pride"],
                "excess": "self-deprecation",
                "deficiency": "arrogance"
            },
            Virtue.PRUDENCE: {
                "description": "Careful judgment and practical wisdom",
                "weight": 0.8,
                "related_vices": ["imprudence", "carelessness"],
                "excess": "over-caution",
                "deficiency": "recklessness"
            },
            Virtue.INTEGRITY: {
                "description": "Consistency between values, words, and actions",
                "weight": 0.9,
                "related_vices": ["hypocrisy", "inconsistency"],
                "excess": "inflexibility",
                "deficiency": "inconsistency"
            },
            Virtue.RESPONSIBILITY: {
                "description": "Accountability for one's actions and their consequences",
                "weight": 0.8,
                "related_vices": ["irresponsibility", "blame-shifting"],
                "excess": "over-responsibility",
                "deficiency": "irresponsibility"
            }
        }
        
        logger.info("Virtue Ethics Framework initialized")
    
    async def analyze_dilemma(self, dilemma) -> Dict[str, Any]:
        """
        Analyze ethical dilemma using virtue ethics principles.
        
        Args:
            dilemma: EthicalDilemma instance
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Assess virtues for each potential action
            action_assessments = {}
            
            for action in dilemma.potential_actions:
                virtue_assessments = await self._assess_action_virtues(action, dilemma)
                overall_score = self._calculate_virtue_score(virtue_assessments)
                action_assessments[action] = {
                    "score": overall_score,
                    "virtue_assessments": virtue_assessments
                }
            
            # Find most virtuous action
            best_action = max(action_assessments, key=lambda x: action_assessments[x]["score"])
            best_score = action_assessments[best_action]["score"]
            
            # Generate reasoning
            reasoning = self._generate_virtue_reasoning(
                action_assessments, best_action, dilemma
            )
            
            return {
                "score": best_score,
                "reasoning": reasoning,
                "best_action": best_action,
                "action_assessments": action_assessments
            }
            
        except Exception as e:
            logger.error(f"Virtue ethics analysis failed: {e}")
            return {
                "score": 0.5,
                "reasoning": [f"Analysis failed: {e}"],
                "best_action": "escalate_for_human_review",
                "action_assessments": {}
            }
    
    async def _assess_action_virtues(self, action: str, dilemma) -> List[VirtueAssessment]:
        """Assess how well an action embodies each virtue."""
        
        virtue_assessments = []
        evaluation_result = dilemma.context.get("evaluation_result")
        
        for virtue in Virtue:
            assessment = self._assess_specific_virtue(virtue, action, dilemma, evaluation_result)
            virtue_assessments.append(assessment)
        
        return virtue_assessments
    
    def _assess_specific_virtue(
        self, 
        virtue: Virtue, 
        action: str, 
        dilemma, 
        evaluation_result
    ) -> VirtueAssessment:
        """Assess how well an action embodies a specific virtue."""
        
        if virtue == Virtue.WISDOM:
            return self._assess_wisdom(action, dilemma, evaluation_result)
        elif virtue == Virtue.COURAGE:
            return self._assess_courage(action, dilemma, evaluation_result)
        elif virtue == Virtue.TEMPERANCE:
            return self._assess_temperance(action, dilemma, evaluation_result)
        elif virtue == Virtue.JUSTICE:
            return self._assess_justice(action, dilemma, evaluation_result)
        elif virtue == Virtue.COMPASSION:
            return self._assess_compassion(action, dilemma, evaluation_result)
        elif virtue == Virtue.HONESTY:
            return self._assess_honesty(action, dilemma, evaluation_result)
        elif virtue == Virtue.HUMILITY:
            return self._assess_humility(action, dilemma, evaluation_result)
        elif virtue == Virtue.PRUDENCE:
            return self._assess_prudence(action, dilemma, evaluation_result)
        elif virtue == Virtue.INTEGRITY:
            return self._assess_integrity(action, dilemma, evaluation_result)
        elif virtue == Virtue.RESPONSIBILITY:
            return self._assess_responsibility(action, dilemma, evaluation_result)
        else:
            return VirtueAssessment(
                virtue=virtue,
                embodiment_score=0.5,
                vice_score=0.0,
                balance_score=0.5,
                justification="Virtue assessment not implemented"
            )
    
    def _assess_wisdom(self, action: str, dilemma, evaluation_result) -> VirtueAssessment:
        """Assess wisdom in the proposed action."""
        
        embodiment_score = 0.5
        vice_score = 0.0
        justification = "Moderate wisdom demonstrated"
        
        if action == "escalate_for_human_review":
            embodiment_score = 0.9
            justification = "Seeking human wisdom shows prudent judgment"
        elif action == "approve_with_warnings":
            embodiment_score = 0.8
            justification = "Balanced approach shows practical wisdom"
        elif action == "approve_content_as_is":
            if evaluation_result and evaluation_result.overall_score < 0.6:
                embodiment_score = 0.2
                vice_score = 0.7
                justification = "Approval despite concerns shows poor judgment"
            else:
                embodiment_score = 0.7
                justification = "Approval of suitable content shows good judgment"
        elif action == "reject_content":
            if evaluation_result and evaluation_result.overall_score < 0.3:
                embodiment_score = 0.9
                justification = "Rejection of harmful content shows wisdom"
            else:
                embodiment_score = 0.4
                justification = "Rejection may be overly cautious"
        
        balance_score = self._calculate_balance_score(embodiment_score, 0.7, 0.3)
        
        return VirtueAssessment(
            virtue=Virtue.WISDOM,
            embodiment_score=embodiment_score,
            vice_score=vice_score,
            balance_score=balance_score,
            justification=justification
        )
    
    def _assess_courage(self, action: str, dilemma, evaluation_result) -> VirtueAssessment:
        """Assess moral courage in the proposed action."""
        
        embodiment_score = 0.5
        vice_score = 0.0
        justification = "Moderate courage shown"
        
        # High severity issues require courage to address
        max_severity = 0.0
        if evaluation_result:
            max_severity = max([issue.severity for issue in evaluation_result.issues], default=0.0)
        
        if action == "reject_content" and max_severity > 0.7:
            embodiment_score = 0.9
            justification = "Courageous stand against harmful content"
        elif action == "approve_content_as_is" and max_severity > 0.5:
            embodiment_score = 0.2
            vice_score = 0.6
            justification = "Lacks courage to address serious concerns"
        elif action == "request_modifications":
            embodiment_score = 0.7
            justification = "Shows courage to request improvements"
        elif action == "escalate_for_human_review":
            embodiment_score = 0.6
            justification = "Shows appropriate caution, not cowardice"
        
        balance_score = self._calculate_balance_score(embodiment_score, 0.8, 0.4)
        
        return VirtueAssessment(
            virtue=Virtue.COURAGE,
            embodiment_score=embodiment_score,
            vice_score=vice_score,
            balance_score=balance_score,
            justification=justification
        )
    
    def _assess_temperance(self, action: str, dilemma, evaluation_result) -> VirtueAssessment:
        """Assess temperance (moderation) in the proposed action."""
        
        embodiment_score = 0.7  # Default to moderate
        vice_score = 0.0
        justification = "Moderate approach taken"
        
        if action == "approve_with_warnings":
            embodiment_score = 0.9
            justification = "Perfect balance between access and safety"
        elif action == "request_modifications":
            embodiment_score = 0.8
            justification = "Balanced approach seeking improvement"
        elif action == "approve_content_as_is":
            embodiment_score = 0.6
            justification = "Somewhat permissive approach"
        elif action == "reject_content":
            if evaluation_result and evaluation_result.overall_score > 0.7:
                embodiment_score = 0.3
                vice_score = 0.4
                justification = "Overly restrictive for acceptable content"
            else:
                embodiment_score = 0.8
                justification = "Appropriate restriction for problematic content"
        
        balance_score = embodiment_score  # Temperance IS about balance
        
        return VirtueAssessment(
            virtue=Virtue.TEMPERANCE,
            embodiment_score=embodiment_score,
            vice_score=vice_score,
            balance_score=balance_score,
            justification=justification
        )
    
    def _assess_justice(self, action: str, dilemma, evaluation_result) -> VirtueAssessment:
        """Assess justice in the proposed action."""
        
        embodiment_score = 0.7
        vice_score = 0.0
        justification = "Generally just approach"
        
        # Check for bias and fairness issues
        bias_severity = 0.0
        if evaluation_result:
            for issue in evaluation_result.issues:
                if issue.category.value in ["bias_detection", "hate_speech"]:
                    bias_severity = max(bias_severity, issue.severity)
        
        if action == "approve_content_as_is" and bias_severity > 0.5:
            embodiment_score = 0.2
            vice_score = 0.7
            justification = "Approval of biased content perpetuates injustice"
        elif action in ["request_modifications", "reject_content"] and bias_severity > 0.5:
            embodiment_score = 0.9
            justification = "Action addresses potential injustices"
        elif action == "approve_with_warnings":
            embodiment_score = 0.8
            justification = "Warnings provide fair notice to users"
        
        balance_score = self._calculate_balance_score(embodiment_score, 0.8, 0.4)
        
        return VirtueAssessment(
            virtue=Virtue.JUSTICE,
            embodiment_score=embodiment_score,
            vice_score=vice_score,
            balance_score=balance_score,
            justification=justification
        )
    
    def _assess_compassion(self, action: str, dilemma, evaluation_result) -> VirtueAssessment:
        """Assess compassion in the proposed action."""
        
        embodiment_score = 0.6
        vice_score = 0.0
        justification = "Shows consideration for others"
        
        # Child safety issues require strong compassion
        child_safety_severity = 0.0
        if evaluation_result:
            for issue in evaluation_result.issues:
                if issue.category.value == "child_safety":
                    child_safety_severity = max(child_safety_severity, issue.severity)
        
        if child_safety_severity > 0.5:
            if action == "reject_content":
                embodiment_score = 1.0
                justification = "Shows deep compassion for child welfare"
            elif action == "approve_content_as_is":
                embodiment_score = 0.1
                vice_score = 0.8
                justification = "Lacks compassion for vulnerable children"
        
        # General harm considerations
        if evaluation_result:
            total_severity = sum(issue.severity for issue in evaluation_result.issues)
            if total_severity > 2.0:  # Multiple serious issues
                if action in ["reject_content", "request_modifications"]:
                    embodiment_score = 0.9
                    justification = "Shows compassion for potential victims"
                elif action == "approve_content_as_is":
                    embodiment_score = 0.3
                    justification = "Limited compassion for those who might be harmed"
        
        balance_score = self._calculate_balance_score(embodiment_score, 0.9, 0.2)
        
        return VirtueAssessment(
            virtue=Virtue.COMPASSION,
            embodiment_score=embodiment_score,
            vice_score=vice_score,
            balance_score=balance_score,
            justification=justification
        )
    
    def _assess_honesty(self, action: str, dilemma, evaluation_result) -> VirtueAssessment:
        """Assess honesty in the proposed action."""
        
        embodiment_score = 0.8
        vice_score = 0.0
        justification = "Honest approach to content evaluation"
        
        # Check for misinformation issues
        misinfo_severity = 0.0
        if evaluation_result:
            for issue in evaluation_result.issues:
                if issue.category.value == "misinformation":
                    misinfo_severity = max(misinfo_severity, issue.severity)
        
        if action == "approve_content_as_is" and misinfo_severity > 0.3:
            embodiment_score = 0.2
            vice_score = 0.6
            justification = "Approval of misinformation undermines honesty"
        elif action == "approve_with_warnings" and misinfo_severity > 0.3:
            embodiment_score = 0.7
            justification = "Warnings provide honest disclosure of concerns"
        elif action in ["request_modifications", "reject_content"] and misinfo_severity > 0.5:
            embodiment_score = 0.9
            justification = "Action upholds commitment to truth"
        
        balance_score = self._calculate_balance_score(embodiment_score, 0.9, 0.5)
        
        return VirtueAssessment(
            virtue=Virtue.HONESTY,
            embodiment_score=embodiment_score,
            vice_score=vice_score,
            balance_score=balance_score,
            justification=justification
        )
    
    def _assess_humility(self, action: str, dilemma, evaluation_result) -> VirtueAssessment:
        """Assess humility in the proposed action."""
        
        embodiment_score = 0.7
        vice_score = 0.0
        justification = "Shows appropriate humility"
        
        if action == "escalate_for_human_review":
            embodiment_score = 0.9
            justification = "Humility in recognizing need for human judgment"
        elif action == "approve_with_warnings":
            embodiment_score = 0.8
            justification = "Humble acknowledgment of limitations"
        elif action == "approve_content_as_is":
            if evaluation_result and len(evaluation_result.issues) > 3:
                embodiment_score = 0.3
                vice_score = 0.5
                justification = "Overconfident despite multiple concerns"
            else:
                embodiment_score = 0.7
                justification = "Reasonable confidence in assessment"
        
        balance_score = self._calculate_balance_score(embodiment_score, 0.8, 0.3)
        
        return VirtueAssessment(
            virtue=Virtue.HUMILITY,
            embodiment_score=embodiment_score,
            vice_score=vice_score,
            balance_score=balance_score,
            justification=justification
        )
    
    def _assess_prudence(self, action: str, dilemma, evaluation_result) -> VirtueAssessment:
        """Assess prudence (practical wisdom) in the proposed action."""
        
        embodiment_score = 0.7
        vice_score = 0.0
        justification = "Shows reasonable prudence"
        
        urgency = dilemma.urgency
        
        if urgency > 0.8:  # High urgency situations
            if action == "escalate_for_human_review":
                embodiment_score = 0.4
                justification = "Escalation delays urgent decision"
            elif action == "reject_content":
                embodiment_score = 0.9
                justification = "Prudent to err on side of caution in urgent case"
        
        # Consider risk-benefit ratio
        if evaluation_result:
            if evaluation_result.overall_score < 0.4:
                if action == "approve_content_as_is":
                    embodiment_score = 0.2
                    vice_score = 0.6
                    justification = "Imprudent to approve high-risk content"
                elif action == "reject_content":
                    embodiment_score = 0.9
                    justification = "Prudent to reject risky content"
        
        balance_score = self._calculate_balance_score(embodiment_score, 0.8, 0.4)
        
        return VirtueAssessment(
            virtue=Virtue.PRUDENCE,
            embodiment_score=embodiment_score,
            vice_score=vice_score,
            balance_score=balance_score,
            justification=justification
        )
    
    def _assess_integrity(self, action: str, dilemma, evaluation_result) -> VirtueAssessment:
        """Assess integrity in the proposed action."""
        
        embodiment_score = 0.8
        vice_score = 0.0
        justification = "Action consistent with ethical principles"
        
        # Check consistency with platform policies (promise-keeping aspect)
        if evaluation_result and evaluation_result.compliance_status.value in ["violation", "critical"]:
            if action == "approve_content_as_is":
                embodiment_score = 0.2
                vice_score = 0.7
                justification = "Approval violates stated ethical commitments"
            elif action in ["request_modifications", "reject_content"]:
                embodiment_score = 0.9
                justification = "Action maintains integrity of ethical standards"
        
        balance_score = self._calculate_balance_score(embodiment_score, 0.9, 0.5)
        
        return VirtueAssessment(
            virtue=Virtue.INTEGRITY,
            embodiment_score=embodiment_score,
            vice_score=vice_score,
            balance_score=balance_score,
            justification=justification
        )
    
    def _assess_responsibility(self, action: str, dilemma, evaluation_result) -> VirtueAssessment:
        """Assess responsibility in the proposed action."""
        
        embodiment_score = 0.7
        vice_score = 0.0
        justification = "Shows appropriate responsibility"
        
        if action == "escalate_for_human_review":
            embodiment_score = 0.8
            justification = "Responsible escalation of complex issues"
        elif action == "approve_content_as_is":
            if evaluation_result and len(evaluation_result.issues) > 2:
                embodiment_score = 0.4
                justification = "May not be taking sufficient responsibility for consequences"
            else:
                embodiment_score = 0.8
                justification = "Taking responsibility for safe content approval"
        elif action in ["request_modifications", "reject_content"]:
            embodiment_score = 0.9
            justification = "Taking responsibility to prevent potential harm"
        
        balance_score = self._calculate_balance_score(embodiment_score, 0.9, 0.3)
        
        return VirtueAssessment(
            virtue=Virtue.RESPONSIBILITY,
            embodiment_score=embodiment_score,
            vice_score=vice_score,
            balance_score=balance_score,
            justification=justification
        )
    
    def _calculate_balance_score(self, score: float, ideal_high: float, ideal_low: float) -> float:
        """Calculate how well a score represents the golden mean (balanced virtue)."""
        
        ideal_mean = (ideal_high + ideal_low) / 2
        
        # Distance from ideal mean
        distance = abs(score - ideal_mean)
        max_distance = max(abs(ideal_high - ideal_mean), abs(ideal_low - ideal_mean))
        
        # Convert distance to balance score (closer to mean = higher balance)
        if max_distance > 0:
            balance_score = 1.0 - (distance / max_distance)
        else:
            balance_score = 1.0
        
        return max(0.0, balance_score)
    
    def _calculate_virtue_score(self, virtue_assessments: List[VirtueAssessment]) -> float:
        """Calculate overall virtue score from individual assessments."""
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for assessment in virtue_assessments:
            virtue_config = self.virtue_definitions[assessment.virtue]
            weight = virtue_config["weight"]
            
            # Combine embodiment and balance, penalize vice
            virtue_score = (assessment.embodiment_score + assessment.balance_score) / 2
            virtue_score = virtue_score * (1.0 - assessment.vice_score * 0.5)  # Vice penalty
            
            total_weighted_score += virtue_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return np.clip(total_weighted_score / total_weight, 0.0, 1.0)
        else:
            return 0.5
    
    def _generate_virtue_reasoning(
        self, 
        action_assessments: Dict[str, Any], 
        best_action: str, 
        dilemma
    ) -> List[str]:
        """Generate reasoning for virtue ethics analysis."""
        
        reasoning = []
        
        reasoning.append(
            f"VIRTUE ETHICS: Evaluated character virtues across {len(dilemma.potential_actions)} actions "
            f"seeking the golden mean and human flourishing"
        )
        
        # Report virtue embodiment for best action
        best_assessment = action_assessments[best_action]
        reasoning.append(f"Most virtuous action: '{best_action}' with score {best_assessment['score']:.2f}")
        
        # Highlight key virtues and vices
        for virtue_assessment in best_assessment["virtue_assessments"]:
            if virtue_assessment.embodiment_score > 0.8:
                reasoning.append(
                    f"VIRTUE: {virtue_assessment.virtue.value} strongly embodied - {virtue_assessment.justification}"
                )
            elif virtue_assessment.vice_score > 0.5:
                reasoning.append(
                    f"VICE: {virtue_assessment.virtue.value} shows concerning vice - {virtue_assessment.justification}"
                )
            elif virtue_assessment.balance_score < 0.4:
                reasoning.append(
                    f"IMBALANCE: {virtue_assessment.virtue.value} lacks golden mean - {virtue_assessment.justification}"
                )
        
        # Overall character assessment
        avg_embodiment = np.mean([va.embodiment_score for va in best_assessment["virtue_assessments"]])
        avg_balance = np.mean([va.balance_score for va in best_assessment["virtue_assessments"]])
        
        if avg_embodiment > 0.8 and avg_balance > 0.7:
            reasoning.append("Action demonstrates excellent moral character")
        elif avg_embodiment < 0.4 or avg_balance < 0.4:
            reasoning.append("Action shows concerning character deficiencies")
        else:
            reasoning.append("Action shows moderate virtue development")
        
        return reasoning
    
    async def shutdown(self) -> None:
        """Shutdown virtue ethics framework."""
        logger.info("Virtue Ethics Framework shutdown complete")