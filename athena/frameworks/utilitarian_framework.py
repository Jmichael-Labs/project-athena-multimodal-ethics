"""
Utilitarian Ethics Framework for Project Athena

Implementation of utilitarian ethical framework focusing on
maximizing overall well-being and minimizing harm.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class UtilityCalculation:
    """Result of utility calculation for stakeholders."""
    stakeholder: str
    positive_utility: float
    negative_utility: float
    net_utility: float
    confidence: float

class UtilitarianFramework:
    """
    Utilitarian ethics framework implementation.
    
    Evaluates actions based on their consequences and overall
    utility maximization for all affected parties.
    """
    
    def __init__(self, config):
        """Initialize utilitarian framework."""
        self.config = config
        
        # Utility weights for different types of outcomes
        self.utility_weights = {
            "safety": 1.0,
            "autonomy": 0.8,
            "pleasure": 0.6,
            "knowledge": 0.7,
            "fairness": 0.9,
            "privacy": 0.8
        }
        
        logger.info("Utilitarian Framework initialized")
    
    async def analyze_dilemma(self, dilemma) -> Dict[str, Any]:
        """
        Analyze ethical dilemma using utilitarian principles.
        
        Args:
            dilemma: EthicalDilemma instance
        
        Returns:
            Dictionary with analysis results
        """
        try:
            # Calculate utility for each potential action
            action_utilities = {}
            
            for action in dilemma.potential_actions:
                utility_score = await self._calculate_action_utility(action, dilemma)
                action_utilities[action] = utility_score
            
            # Find action that maximizes utility
            best_action = max(action_utilities, key=action_utilities.get)
            best_utility = action_utilities[best_action]
            
            # Generate reasoning
            reasoning = self._generate_utilitarian_reasoning(
                action_utilities, best_action, dilemma
            )
            
            # Normalize score to 0-1 range
            normalized_score = self._normalize_utility_score(best_utility)
            
            return {
                "score": normalized_score,
                "reasoning": reasoning,
                "best_action": best_action,
                "action_utilities": action_utilities
            }
            
        except Exception as e:
            logger.error(f"Utilitarian analysis failed: {e}")
            return {
                "score": 0.5,
                "reasoning": [f"Analysis failed: {e}"],
                "best_action": "escalate_for_human_review",
                "action_utilities": {}
            }
    
    async def _calculate_action_utility(self, action: str, dilemma) -> float:
        """Calculate total utility for a specific action."""
        
        stakeholder_utilities = []
        
        for stakeholder in dilemma.stakeholders:
            utility = self._calculate_stakeholder_utility(action, stakeholder, dilemma)
            stakeholder_utilities.append(utility)
        
        # Total utility is sum of all stakeholder utilities
        return sum(u.net_utility for u in stakeholder_utilities)
    
    def _calculate_stakeholder_utility(
        self, 
        action: str, 
        stakeholder: str, 
        dilemma
    ) -> UtilityCalculation:
        """Calculate utility for a specific stakeholder."""
        
        positive_utility = 0.0
        negative_utility = 0.0
        
        # Get evaluation result from context
        evaluation_result = dilemma.context.get("evaluation_result")
        
        if action == "approve_content_as_is":
            positive_utility += self._calculate_approval_benefits(stakeholder, evaluation_result)
            negative_utility += self._calculate_approval_harms(stakeholder, evaluation_result)
        
        elif action == "approve_with_warnings":
            positive_utility += self._calculate_warning_benefits(stakeholder, evaluation_result)
            negative_utility += self._calculate_warning_harms(stakeholder, evaluation_result)
        
        elif action == "request_modifications":
            positive_utility += self._calculate_modification_benefits(stakeholder, evaluation_result)
            negative_utility += self._calculate_modification_harms(stakeholder, evaluation_result)
        
        elif action == "reject_content":
            positive_utility += self._calculate_rejection_benefits(stakeholder, evaluation_result)
            negative_utility += self._calculate_rejection_harms(stakeholder, evaluation_result)
        
        elif action == "escalate_for_human_review":
            positive_utility += self._calculate_escalation_benefits(stakeholder, evaluation_result)
            negative_utility += self._calculate_escalation_harms(stakeholder, evaluation_result)
        
        net_utility = positive_utility - negative_utility
        confidence = 0.8  # Default confidence
        
        return UtilityCalculation(
            stakeholder=stakeholder,
            positive_utility=positive_utility,
            negative_utility=negative_utility,
            net_utility=net_utility,
            confidence=confidence
        )
    
    def _calculate_approval_benefits(self, stakeholder: str, evaluation_result) -> float:
        """Calculate benefits of approving content as-is."""
        benefits = 0.0
        
        if stakeholder == "content_users":
            # Users get access to content
            benefits += 0.5 * self.utility_weights["pleasure"]
            # If content is high quality (high score), more benefit
            if evaluation_result and evaluation_result.overall_score > 0.8:
                benefits += 0.3
        
        elif stakeholder == "content_creators":
            # Creators get their content distributed
            benefits += 0.8 * self.utility_weights["autonomy"]
            # Economic benefit
            benefits += 0.6
        
        elif stakeholder == "platform_operators":
            # Operational efficiency
            benefits += 0.4
            # User engagement
            benefits += 0.3
        
        return benefits
    
    def _calculate_approval_harms(self, stakeholder: str, evaluation_result) -> float:
        """Calculate harms of approving content as-is."""
        harms = 0.0
        
        if not evaluation_result:
            return 0.1  # Unknown risk
        
        # Base harm on evaluation issues
        total_severity = sum(issue.severity for issue in evaluation_result.issues)
        
        if stakeholder == "content_users":
            # Direct exposure to harmful content
            harms += total_severity * 0.8 * self.utility_weights["safety"]
        
        elif stakeholder == "society":
            # Broader social impact
            harms += total_severity * 0.6
        
        elif stakeholder == "platform_operators":
            # Reputational and legal risks
            harms += total_severity * 0.4
        
        return harms
    
    def _calculate_warning_benefits(self, stakeholder: str, evaluation_result) -> float:
        """Calculate benefits of approving with warnings."""
        benefits = 0.0
        
        if stakeholder == "content_users":
            # Informed consent
            benefits += 0.6 * self.utility_weights["autonomy"]
            # Still get access to content
            benefits += 0.4
        
        elif stakeholder == "content_creators":
            # Content still distributed
            benefits += 0.6
        
        elif stakeholder == "platform_operators":
            # Reduced liability
            benefits += 0.5
        
        return benefits
    
    def _calculate_warning_harms(self, stakeholder: str, evaluation_result) -> float:
        """Calculate harms of approving with warnings."""
        harms = 0.0
        
        if not evaluation_result:
            return 0.05
        
        # Reduced harm due to warnings
        warning_reduction_factor = 0.5
        total_severity = sum(issue.severity for issue in evaluation_result.issues)
        
        if stakeholder == "content_users":
            harms += total_severity * 0.4 * warning_reduction_factor
        
        elif stakeholder == "content_creators":
            # Potential reduced engagement
            harms += 0.2
        
        return harms
    
    def _calculate_modification_benefits(self, stakeholder: str, evaluation_result) -> float:
        """Calculate benefits of requesting modifications."""
        benefits = 0.0
        
        if stakeholder == "content_users":
            # Safer, improved content
            benefits += 0.7 * self.utility_weights["safety"]
        
        elif stakeholder == "content_creators":
            # Opportunity to improve content
            benefits += 0.4
        
        elif stakeholder == "society":
            # Better content standards
            benefits += 0.6
        
        return benefits
    
    def _calculate_modification_harms(self, stakeholder: str, evaluation_result) -> float:
        """Calculate harms of requesting modifications."""
        harms = 0.0
        
        if stakeholder == "content_creators":
            # Time and effort to modify
            harms += 0.5
            # Potential loss of original vision
            harms += 0.3
        
        elif stakeholder == "content_users":
            # Delayed access
            harms += 0.2
        
        return harms
    
    def _calculate_rejection_benefits(self, stakeholder: str, evaluation_result) -> float:
        """Calculate benefits of rejecting content."""
        benefits = 0.0
        
        if not evaluation_result:
            return 0.3  # Safety benefit of unknown
        
        # Benefits proportional to harm prevented
        total_severity = sum(issue.severity for issue in evaluation_result.issues)
        
        if stakeholder == "content_users":
            # Protection from harmful content
            benefits += total_severity * 0.9 * self.utility_weights["safety"]
        
        elif stakeholder == "society":
            # Upholding ethical standards
            benefits += total_severity * 0.8
        
        elif stakeholder == "platform_operators":
            # Avoiding legal and reputational risks
            benefits += total_severity * 0.6
        
        return benefits
    
    def _calculate_rejection_harms(self, stakeholder: str, evaluation_result) -> float:
        """Calculate harms of rejecting content."""
        harms = 0.0
        
        if stakeholder == "content_creators":
            # Loss of work and investment
            harms += 0.8
        
        elif stakeholder == "content_users":
            # Loss of potentially valuable content
            harms += 0.3
        
        elif stakeholder == "platform_operators":
            # Potential over-censorship concerns
            harms += 0.2
        
        return harms
    
    def _calculate_escalation_benefits(self, stakeholder: str, evaluation_result) -> float:
        """Calculate benefits of human review escalation."""
        benefits = 0.0
        
        # Human review provides better judgment
        if stakeholder in ["content_users", "society"]:
            benefits += 0.6 * self.utility_weights["fairness"]
        
        elif stakeholder == "content_creators":
            # Fair human consideration
            benefits += 0.5
        
        return benefits
    
    def _calculate_escalation_harms(self, stakeholder: str, evaluation_result) -> float:
        """Calculate harms of human review escalation."""
        harms = 0.0
        
        # Delays and costs
        if stakeholder == "platform_operators":
            harms += 0.4
        
        if stakeholder in ["content_users", "content_creators"]:
            # Delayed resolution
            harms += 0.3
        
        return harms
    
    def _generate_utilitarian_reasoning(
        self, 
        action_utilities: Dict[str, float], 
        best_action: str, 
        dilemma
    ) -> List[str]:
        """Generate reasoning for utilitarian analysis."""
        
        reasoning = []
        
        reasoning.append(
            f"UTILITARIAN: Analyzed {len(dilemma.potential_actions)} potential actions "
            f"across {len(dilemma.stakeholders)} stakeholder groups"
        )
        
        # Report utility scores
        for action, utility in sorted(action_utilities.items(), key=lambda x: x[1], reverse=True):
            reasoning.append(f"Action '{action}': Total utility = {utility:.2f}")
        
        reasoning.append(
            f"Maximum utility achieved by '{best_action}' with score {action_utilities[best_action]:.2f}"
        )
        
        # Explain key trade-offs
        if action_utilities[best_action] < 0:
            reasoning.append("Best option still results in net negative utility")
        
        return reasoning
    
    def _normalize_utility_score(self, utility_score: float) -> float:
        """Normalize utility score to 0-1 range."""
        
        # Utility scores can be negative, so we need to map to 0-1
        # Assume utility range is roughly -5 to +5
        min_utility = -5.0
        max_utility = 5.0
        
        normalized = (utility_score - min_utility) / (max_utility - min_utility)
        return np.clip(normalized, 0.0, 1.0)
    
    async def shutdown(self) -> None:
        """Shutdown utilitarian framework."""
        logger.info("Utilitarian Framework shutdown complete")