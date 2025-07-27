"""
Deontological Ethics Framework for Project Athena

Implementation of deontological ethical framework focusing on
duty-based ethics, rules, and categorical imperatives.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class EthicalDuty(Enum):
    """Core ethical duties in deontological framework."""
    RESPECT_PERSONS = "respect_persons"
    TRUTHFULNESS = "truthfulness"
    PROMISE_KEEPING = "promise_keeping"
    NON_MALEFICENCE = "non_maleficence"
    JUSTICE = "justice"
    BENEFICENCE = "beneficence"
    AUTONOMY = "autonomy"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"

@dataclass
class DutyEvaluation:
    """Evaluation of adherence to a specific duty."""
    duty: EthicalDuty
    adherence_score: float  # 0.0 to 1.0
    violation_severity: float  # 0.0 to 1.0
    is_absolute_violation: bool
    justification: str

class DeontologicalFramework:
    """
    Deontological ethics framework implementation.
    
    Evaluates actions based on adherence to ethical duties and rules,
    regardless of consequences (duty-based ethics).
    """
    
    def __init__(self, config):
        """Initialize deontological framework."""
        self.config = config
        
        # Define categorical imperatives and duties
        self.ethical_duties = {
            EthicalDuty.RESPECT_PERSONS: {
                "description": "Treat humanity, whether in your own person or that of another, always as an end and never merely as a means",
                "weight": 1.0,
                "absolute": True
            },
            EthicalDuty.TRUTHFULNESS: {
                "description": "Always tell the truth and avoid deception",
                "weight": 0.9,
                "absolute": False
            },
            EthicalDuty.NON_MALEFICENCE: {
                "description": "Do no harm to others",
                "weight": 1.0,
                "absolute": True
            },
            EthicalDuty.AUTONOMY: {
                "description": "Respect the self-determination of rational beings",
                "weight": 0.9,
                "absolute": False
            },
            EthicalDuty.JUSTICE: {
                "description": "Give each person their due and treat equals equally",
                "weight": 0.8,
                "absolute": False
            },
            EthicalDuty.BENEFICENCE: {
                "description": "Act to benefit others and promote well-being",
                "weight": 0.7,
                "absolute": False
            },
            EthicalDuty.PRIVACY: {
                "description": "Respect the privacy and confidentiality of others",
                "weight": 0.8,
                "absolute": False
            },
            EthicalDuty.FAIRNESS: {
                "description": "Treat all persons fairly without discrimination",
                "weight": 0.9,
                "absolute": False
            },
            EthicalDuty.PROMISE_KEEPING: {
                "description": "Honor commitments and agreements made to others",
                "weight": 0.7,
                "absolute": False
            }
        }
        
        logger.info("Deontological Framework initialized")
    
    async def analyze_dilemma(self, dilemma) -> Dict[str, Any]:
        """
        Analyze ethical dilemma using deontological principles.
        
        Args:
            dilemma: EthicalDilemma instance
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Evaluate adherence to duties for each action
            action_evaluations = {}
            
            for action in dilemma.potential_actions:
                duty_evaluations = await self._evaluate_action_duties(action, dilemma)
                overall_score = self._calculate_deontological_score(duty_evaluations)
                action_evaluations[action] = {
                    "score": overall_score,
                    "duty_evaluations": duty_evaluations
                }
            
            # Find action with highest deontological score
            best_action = max(action_evaluations, key=lambda x: action_evaluations[x]["score"])
            best_score = action_evaluations[best_action]["score"]
            
            # Generate reasoning
            reasoning = self._generate_deontological_reasoning(
                action_evaluations, best_action, dilemma
            )
            
            return {
                "score": best_score,
                "reasoning": reasoning,
                "best_action": best_action,
                "action_evaluations": action_evaluations
            }
            
        except Exception as e:
            logger.error(f"Deontological analysis failed: {e}")
            return {
                "score": 0.5,
                "reasoning": [f"Analysis failed: {e}"],
                "best_action": "escalate_for_human_review",
                "action_evaluations": {}
            }
    
    async def _evaluate_action_duties(self, action: str, dilemma) -> List[DutyEvaluation]:
        """Evaluate how well an action adheres to ethical duties."""
        
        duty_evaluations = []
        evaluation_result = dilemma.context.get("evaluation_result")
        
        for duty in EthicalDuty:
            evaluation = self._evaluate_specific_duty(duty, action, dilemma, evaluation_result)
            duty_evaluations.append(evaluation)
        
        return duty_evaluations
    
    def _evaluate_specific_duty(
        self, 
        duty: EthicalDuty, 
        action: str, 
        dilemma, 
        evaluation_result
    ) -> DutyEvaluation:
        """Evaluate adherence to a specific ethical duty."""
        
        if duty == EthicalDuty.RESPECT_PERSONS:
            return self._evaluate_respect_persons(action, dilemma, evaluation_result)
        elif duty == EthicalDuty.TRUTHFULNESS:
            return self._evaluate_truthfulness(action, dilemma, evaluation_result)
        elif duty == EthicalDuty.NON_MALEFICENCE:
            return self._evaluate_non_maleficence(action, dilemma, evaluation_result)
        elif duty == EthicalDuty.AUTONOMY:
            return self._evaluate_autonomy(action, dilemma, evaluation_result)
        elif duty == EthicalDuty.JUSTICE:
            return self._evaluate_justice(action, dilemma, evaluation_result)
        elif duty == EthicalDuty.BENEFICENCE:
            return self._evaluate_beneficence(action, dilemma, evaluation_result)
        elif duty == EthicalDuty.PRIVACY:
            return self._evaluate_privacy(action, dilemma, evaluation_result)
        elif duty == EthicalDuty.FAIRNESS:
            return self._evaluate_fairness(action, dilemma, evaluation_result)
        elif duty == EthicalDuty.PROMISE_KEEPING:
            return self._evaluate_promise_keeping(action, dilemma, evaluation_result)
        else:
            return DutyEvaluation(
                duty=duty,
                adherence_score=0.5,
                violation_severity=0.0,
                is_absolute_violation=False,
                justification="Duty not implemented"
            )
    
    def _evaluate_respect_persons(self, action: str, dilemma, evaluation_result) -> DutyEvaluation:
        """Evaluate adherence to respect for persons duty."""
        
        adherence_score = 1.0
        violation_severity = 0.0
        is_absolute_violation = False
        justification = "Action respects human dignity"
        
        if evaluation_result:
            # Check for violations that treat people as mere means
            for issue in evaluation_result.issues:
                if issue.category.value in ["hate_speech", "sexual_content", "violence"]:
                    violation_severity = max(violation_severity, issue.severity)
                    if issue.severity > 0.8:
                        is_absolute_violation = True
        
        if action == "approve_content_as_is" and violation_severity > 0.5:
            adherence_score = 1.0 - violation_severity
            justification = f"Approval may violate respect for persons (severity: {violation_severity:.2f})"
        elif action == "reject_content" and violation_severity > 0.7:
            adherence_score = 1.0  # Rejection upholds respect for persons
            justification = "Rejection protects human dignity"
        
        return DutyEvaluation(
            duty=EthicalDuty.RESPECT_PERSONS,
            adherence_score=adherence_score,
            violation_severity=violation_severity,
            is_absolute_violation=is_absolute_violation,
            justification=justification
        )
    
    def _evaluate_truthfulness(self, action: str, dilemma, evaluation_result) -> DutyEvaluation:
        """Evaluate adherence to truthfulness duty."""
        
        adherence_score = 1.0
        violation_severity = 0.0
        justification = "Action maintains truthfulness"
        
        if evaluation_result:
            # Check for misinformation
            for issue in evaluation_result.issues:
                if issue.category.value == "misinformation":
                    violation_severity = max(violation_severity, issue.severity)
        
        if action == "approve_content_as_is" and violation_severity > 0.3:
            adherence_score = 1.0 - violation_severity
            justification = f"Approval may spread misinformation (severity: {violation_severity:.2f})"
        elif action in ["request_modifications", "reject_content"] and violation_severity > 0.3:
            adherence_score = 1.0
            justification = "Action prevents spread of misinformation"
        
        return DutyEvaluation(
            duty=EthicalDuty.TRUTHFULNESS,
            adherence_score=adherence_score,
            violation_severity=violation_severity,
            is_absolute_violation=False,
            justification=justification
        )
    
    def _evaluate_non_maleficence(self, action: str, dilemma, evaluation_result) -> DutyEvaluation:
        """Evaluate adherence to non-maleficence (do no harm) duty."""
        
        adherence_score = 1.0
        violation_severity = 0.0
        is_absolute_violation = False
        justification = "Action avoids causing harm"
        
        if evaluation_result:
            # Calculate total harm potential
            harmful_categories = ["harmful_content", "violence", "hate_speech", "child_safety"]
            for issue in evaluation_result.issues:
                if issue.category.value in harmful_categories:
                    violation_severity = max(violation_severity, issue.severity)
                    if issue.severity > 0.9:
                        is_absolute_violation = True
        
        if action == "approve_content_as_is" and violation_severity > 0.0:
            adherence_score = 1.0 - violation_severity
            justification = f"Approval may cause harm (severity: {violation_severity:.2f})"
        elif action in ["reject_content", "request_modifications"] and violation_severity > 0.5:
            adherence_score = 1.0
            justification = "Action prevents potential harm"
        
        return DutyEvaluation(
            duty=EthicalDuty.NON_MALEFICENCE,
            adherence_score=adherence_score,
            violation_severity=violation_severity,
            is_absolute_violation=is_absolute_violation,
            justification=justification
        )
    
    def _evaluate_autonomy(self, action: str, dilemma, evaluation_result) -> DutyEvaluation:
        """Evaluate adherence to autonomy duty."""
        
        adherence_score = 0.8  # Default moderate adherence
        violation_severity = 0.0
        justification = "Action respects autonomy"
        
        if evaluation_result:
            # Check for privacy violations that undermine autonomy
            for issue in evaluation_result.issues:
                if issue.category.value == "privacy_violation":
                    violation_severity = max(violation_severity, issue.severity)
        
        if action == "approve_content_as_is":
            adherence_score = 0.9  # Allows user choice
            if violation_severity > 0.3:
                adherence_score = 0.7
                justification = "Approval respects choice but may violate privacy"
        elif action == "approve_with_warnings":
            adherence_score = 1.0  # Informed consent maximizes autonomy
            justification = "Warnings enable informed autonomous choice"
        elif action == "reject_content":
            adherence_score = 0.3  # Limits choice
            justification = "Rejection limits user autonomy"
        
        return DutyEvaluation(
            duty=EthicalDuty.AUTONOMY,
            adherence_score=adherence_score,
            violation_severity=violation_severity,
            is_absolute_violation=False,
            justification=justification
        )
    
    def _evaluate_justice(self, action: str, dilemma, evaluation_result) -> DutyEvaluation:
        """Evaluate adherence to justice duty."""
        
        adherence_score = 0.8
        violation_severity = 0.0
        justification = "Action upholds justice"
        
        if evaluation_result:
            # Check for bias and discrimination
            for issue in evaluation_result.issues:
                if issue.category.value == "bias_detection":
                    violation_severity = max(violation_severity, issue.severity)
        
        if action == "approve_content_as_is" and violation_severity > 0.3:
            adherence_score = 1.0 - violation_severity
            justification = f"Approval may perpetuate injustice (bias: {violation_severity:.2f})"
        elif action in ["request_modifications", "reject_content"] and violation_severity > 0.5:
            adherence_score = 1.0
            justification = "Action addresses potential injustices"
        
        return DutyEvaluation(
            duty=EthicalDuty.JUSTICE,
            adherence_score=adherence_score,
            violation_severity=violation_severity,
            is_absolute_violation=False,
            justification=justification
        )
    
    def _evaluate_beneficence(self, action: str, dilemma, evaluation_result) -> DutyEvaluation:
        """Evaluate adherence to beneficence duty."""
        
        adherence_score = 0.6  # Default moderate
        justification = "Action has neutral benefit"
        
        if action == "approve_content_as_is":
            if evaluation_result and evaluation_result.overall_score > 0.7:
                adherence_score = 0.8
                justification = "Approval provides benefit to users"
            else:
                adherence_score = 0.4
                justification = "Approval provides limited benefit"
        elif action == "approve_with_warnings":
            adherence_score = 0.7
            justification = "Warnings provide safety benefit"
        elif action == "request_modifications":
            adherence_score = 0.8
            justification = "Modifications improve content for all"
        elif action == "reject_content":
            adherence_score = 0.9
            justification = "Rejection protects potential victims"
        
        return DutyEvaluation(
            duty=EthicalDuty.BENEFICENCE,
            adherence_score=adherence_score,
            violation_severity=0.0,
            is_absolute_violation=False,
            justification=justification
        )
    
    def _evaluate_privacy(self, action: str, dilemma, evaluation_result) -> DutyEvaluation:
        """Evaluate adherence to privacy duty."""
        
        adherence_score = 1.0
        violation_severity = 0.0
        justification = "Action respects privacy"
        
        if evaluation_result:
            for issue in evaluation_result.issues:
                if issue.category.value == "privacy_violation":
                    violation_severity = max(violation_severity, issue.severity)
        
        if action == "approve_content_as_is" and violation_severity > 0.2:
            adherence_score = 1.0 - violation_severity
            justification = f"Approval may violate privacy (severity: {violation_severity:.2f})"
        elif action in ["request_modifications", "reject_content"] and violation_severity > 0.3:
            adherence_score = 1.0
            justification = "Action protects privacy"
        
        return DutyEvaluation(
            duty=EthicalDuty.PRIVACY,
            adherence_score=adherence_score,
            violation_severity=violation_severity,
            is_absolute_violation=False,
            justification=justification
        )
    
    def _evaluate_fairness(self, action: str, dilemma, evaluation_result) -> DutyEvaluation:
        """Evaluate adherence to fairness duty."""
        
        adherence_score = 0.8
        violation_severity = 0.0
        justification = "Action treats all fairly"
        
        if evaluation_result:
            for issue in evaluation_result.issues:
                if issue.category.value in ["bias_detection", "hate_speech"]:
                    violation_severity = max(violation_severity, issue.severity)
        
        if action == "approve_content_as_is" and violation_severity > 0.3:
            adherence_score = 1.0 - violation_severity
            justification = f"Approval may be unfair to affected groups (severity: {violation_severity:.2f})"
        elif action in ["request_modifications", "reject_content"] and violation_severity > 0.5:
            adherence_score = 1.0
            justification = "Action promotes fairness"
        
        return DutyEvaluation(
            duty=EthicalDuty.FAIRNESS,
            adherence_score=adherence_score,
            violation_severity=violation_severity,
            is_absolute_violation=False,
            justification=justification
        )
    
    def _evaluate_promise_keeping(self, action: str, dilemma, evaluation_result) -> DutyEvaluation:
        """Evaluate adherence to promise-keeping duty."""
        
        # This duty is more relevant to explicit agreements
        # For content evaluation, we consider platform policies as implicit promises
        adherence_score = 0.8
        justification = "Action honors platform policies"
        
        if evaluation_result and evaluation_result.compliance_status.value in ["violation", "critical"]:
            if action == "approve_content_as_is":
                adherence_score = 0.2
                justification = "Approval violates platform policy commitments"
            elif action in ["request_modifications", "reject_content"]:
                adherence_score = 1.0
                justification = "Action upholds platform policy commitments"
        
        return DutyEvaluation(
            duty=EthicalDuty.PROMISE_KEEPING,
            adherence_score=adherence_score,
            violation_severity=0.0,
            is_absolute_violation=False,
            justification=justification
        )
    
    def _calculate_deontological_score(self, duty_evaluations: List[DutyEvaluation]) -> float:
        """Calculate overall deontological score from duty evaluations."""
        
        # Check for absolute violations first
        for evaluation in duty_evaluations:
            if evaluation.is_absolute_violation:
                return 0.0  # Absolute duties cannot be violated
        
        # Calculate weighted average of adherence scores
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for evaluation in duty_evaluations:
            duty_config = self.ethical_duties[evaluation.duty]
            weight = duty_config["weight"]
            
            total_weighted_score += evaluation.adherence_score * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_weighted_score / total_weight
        else:
            return 0.5
    
    def _generate_deontological_reasoning(
        self, 
        action_evaluations: Dict[str, Any], 
        best_action: str, 
        dilemma
    ) -> List[str]:
        """Generate reasoning for deontological analysis."""
        
        reasoning = []
        
        reasoning.append(
            f"DEONTOLOGICAL: Evaluated duties across {len(dilemma.potential_actions)} actions "
            f"based on categorical imperatives and ethical rules"
        )
        
        # Check for absolute violations
        absolute_violations = []
        for action, evaluation in action_evaluations.items():
            for duty_eval in evaluation["duty_evaluations"]:
                if duty_eval.is_absolute_violation:
                    absolute_violations.append(f"{action}: {duty_eval.duty.value}")
        
        if absolute_violations:
            reasoning.append(f"ABSOLUTE VIOLATIONS FOUND: {absolute_violations}")
        
        # Report duty adherence for best action
        best_evaluation = action_evaluations[best_action]
        reasoning.append(f"Best action '{best_action}' with score {best_evaluation['score']:.2f}")
        
        # Highlight key duty considerations
        for duty_eval in best_evaluation["duty_evaluations"]:
            if duty_eval.adherence_score < 0.5 or duty_eval.is_absolute_violation:
                reasoning.append(f"CONCERN: {duty_eval.duty.value} - {duty_eval.justification}")
            elif duty_eval.adherence_score > 0.9:
                reasoning.append(f"STRENGTH: {duty_eval.duty.value} - {duty_eval.justification}")
        
        return reasoning
    
    async def shutdown(self) -> None:
        """Shutdown deontological framework."""
        logger.info("Deontological Framework shutdown complete")