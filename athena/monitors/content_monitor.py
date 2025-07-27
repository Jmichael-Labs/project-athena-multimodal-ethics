"""
Content Monitor for Project Athena

Real-time content monitoring system for tracking ethical evaluations,
detecting patterns, and providing alerts for concerning content trends.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import time
from collections import deque, defaultdict
import numpy as np

# Monitoring and metrics imports
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    import redis
    from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Text
    from sqlalchemy.orm import sessionmaker
except ImportError as e:
    logging.warning(f"Some monitoring dependencies not available: {e}")

from ..core.evaluator import EvaluationResult, EthicsIssue, ComplianceStatus
from ..core.ethics_engine import MultimodalContent

logger = logging.getLogger(__name__)

@dataclass
class ContentAlert:
    """Alert for concerning content patterns."""
    alert_id: str
    alert_type: str
    severity: float
    message: str
    content_ids: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MonitoringMetrics:
    """Real-time monitoring metrics."""
    total_evaluations: int = 0
    evaluations_per_second: float = 0.0
    avg_processing_time: float = 0.0
    compliance_rate: float = 0.0
    critical_violations: int = 0
    warnings: int = 0
    modality_distribution: Dict[str, int] = field(default_factory=dict)
    issue_distribution: Dict[str, int] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ContentPattern:
    """Detected pattern in content evaluations."""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    severity: float
    first_seen: datetime
    last_seen: datetime
    examples: List[str] = field(default_factory=list)

class ContentMonitor:
    """
    Real-time content monitoring system.
    
    Monitors ethical evaluations in real-time, detects patterns,
    generates alerts, and provides comprehensive metrics.
    """
    
    def __init__(self, config):
        """
        Initialize content monitor.
        
        Args:
            config: EthicsConfig instance
        """
        self.config = config
        self.monitoring_active = False
        
        # Initialize storage and metrics
        self._initialize_storage()
        self._initialize_metrics()
        self._initialize_pattern_detection()
        
        # Real-time data structures
        self.recent_evaluations = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        self.detected_patterns = {}
        
        # Monitoring thresholds
        self.alert_thresholds = {
            "critical_violation_rate": 0.05,  # 5% critical violations
            "avg_processing_time": 10.0,      # 10 seconds
            "evaluation_rate_drop": 0.5,      # 50% drop in evaluations
            "pattern_frequency": 10            # Pattern seen 10+ times
        }
        
        # Performance tracking
        self.performance_window = timedelta(minutes=5)
        self.evaluation_timestamps = deque(maxlen=1000)
        
        logger.info("Content Monitor initialized")
    
    def _initialize_storage(self) -> None:
        """Initialize storage backends for monitoring data."""
        try:
            # Redis for real-time data
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True
            ) if 'redis' in globals() else None
            
            # SQLAlchemy for persistent data
            if 'create_engine' in globals():
                self.engine = create_engine('sqlite:///data/athena_monitoring.db')
                self.metadata = MetaData()
                
                # Define monitoring tables
                self.evaluations_table = Table(
                    'evaluations', self.metadata,
                    Column('id', Integer, primary_key=True),
                    Column('timestamp', DateTime),
                    Column('overall_score', Float),
                    Column('compliance_status', String),
                    Column('modalities', String),
                    Column('issues_count', Integer),
                    Column('processing_time', Float),
                    Column('metadata', Text)
                )
                
                self.alerts_table = Table(
                    'alerts', self.metadata,
                    Column('id', Integer, primary_key=True),
                    Column('alert_id', String),
                    Column('alert_type', String),
                    Column('severity', Float),
                    Column('message', Text),
                    Column('timestamp', DateTime),
                    Column('acknowledged', Integer)
                )
                
                self.metadata.create_all(self.engine)
                self.Session = sessionmaker(bind=self.engine)
            else:
                self.engine = None
                self.Session = None
            
        except Exception as e:
            logger.warning(f"Storage initialization failed: {e}")
            self.redis_client = None
            self.engine = None
    
    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        if 'Counter' not in globals():
            logger.warning("Prometheus metrics not available")
            return
        
        try:
            # Prometheus metrics
            self.metrics = {
                'evaluations_total': Counter('athena_evaluations_total', 'Total evaluations processed'),
                'evaluations_by_status': Counter('athena_evaluations_by_status', 'Evaluations by compliance status', ['status']),
                'evaluations_by_modality': Counter('athena_evaluations_by_modality', 'Evaluations by modality', ['modality']),
                'processing_time': Histogram('athena_processing_time_seconds', 'Processing time for evaluations'),
                'issues_total': Counter('athena_issues_total', 'Total ethical issues detected', ['category']),
                'alerts_total': Counter('athena_alerts_total', 'Total alerts generated', ['type']),
                'compliance_rate': Gauge('athena_compliance_rate', 'Current compliance rate'),
                'active_patterns': Gauge('athena_active_patterns', 'Number of active content patterns')
            }
            
            # Start Prometheus HTTP server
            if self.config.monitoring_enabled:
                start_http_server(self.config.metrics_port)
                logger.info(f"Prometheus metrics server started on port {self.config.metrics_port}")
            
        except Exception as e:
            logger.warning(f"Metrics initialization failed: {e}")
            self.metrics = {}
    
    def _initialize_pattern_detection(self) -> None:
        """Initialize pattern detection algorithms."""
        self.pattern_detectors = {
            'recurring_violations': self._detect_recurring_violations,
            'escalating_severity': self._detect_escalating_severity,
            'modality_anomalies': self._detect_modality_anomalies,
            'temporal_clusters': self._detect_temporal_clusters
        }
        
        # Pattern detection windows
        self.pattern_windows = {
            'short': timedelta(minutes=15),
            'medium': timedelta(hours=1),
            'long': timedelta(hours=24)
        }
    
    async def log_evaluation(
        self, 
        evaluation_result: EvaluationResult, 
        content: Any, 
        processing_time: float = 0.0
    ) -> None:
        """
        Log an evaluation result for monitoring.
        
        Args:
            evaluation_result: The evaluation result to log
            content: The content that was evaluated
            processing_time: Time taken to process the evaluation
        """
        if not self.monitoring_active:
            return
        
        try:
            timestamp = datetime.now()
            
            # Update real-time data
            self.recent_evaluations.append({
                'timestamp': timestamp,
                'result': evaluation_result,
                'content_type': type(content).__name__ if content else 'unknown',
                'processing_time': processing_time
            })
            
            self.evaluation_timestamps.append(timestamp)
            
            # Update Prometheus metrics
            if self.metrics:
                self.metrics['evaluations_total'].inc()
                self.metrics['evaluations_by_status'].labels(
                    status=evaluation_result.compliance_status.value
                ).inc()
                
                for modality in evaluation_result.modality_scores.keys():
                    self.metrics['evaluations_by_modality'].labels(modality=modality).inc()
                
                self.metrics['processing_time'].observe(processing_time)
                
                for issue in evaluation_result.issues:
                    self.metrics['issues_total'].labels(category=issue.category.value).inc()
            
            # Store in persistent storage
            await self._store_evaluation(evaluation_result, content, processing_time, timestamp)
            
            # Update Redis cache
            if self.redis_client:
                await self._update_redis_metrics(evaluation_result, timestamp)
            
            # Run pattern detection
            await self._run_pattern_detection()
            
            # Check for alerts
            await self._check_alert_conditions(evaluation_result, timestamp)
            
        except Exception as e:
            logger.error(f"Error logging evaluation: {e}")
    
    async def _store_evaluation(
        self, 
        evaluation_result: EvaluationResult, 
        content: Any, 
        processing_time: float, 
        timestamp: datetime
    ) -> None:
        """Store evaluation in persistent database."""
        if not self.engine or not self.Session:
            return
        
        try:
            session = self.Session()
            
            # Prepare data for storage
            modalities_str = ','.join(evaluation_result.modality_scores.keys())
            metadata_str = json.dumps(evaluation_result.metadata, default=str)
            
            # Insert evaluation record
            session.execute(
                self.evaluations_table.insert().values(
                    timestamp=timestamp,
                    overall_score=evaluation_result.overall_score,
                    compliance_status=evaluation_result.compliance_status.value,
                    modalities=modalities_str,
                    issues_count=len(evaluation_result.issues),
                    processing_time=processing_time,
                    metadata=metadata_str
                )
            )
            
            session.commit()
            session.close()
            
        except Exception as e:
            logger.warning(f"Database storage failed: {e}")
    
    async def _update_redis_metrics(
        self, 
        evaluation_result: EvaluationResult, 
        timestamp: datetime
    ) -> None:
        """Update real-time metrics in Redis."""
        if not self.redis_client:
            return
        
        try:
            pipe = self.redis_client.pipeline()
            
            # Update counters
            pipe.incr('athena:evaluations:total')
            pipe.incr(f'athena:evaluations:status:{evaluation_result.compliance_status.value}')
            
            for modality in evaluation_result.modality_scores.keys():
                pipe.incr(f'athena:evaluations:modality:{modality}')
            
            for issue in evaluation_result.issues:
                pipe.incr(f'athena:issues:category:{issue.category.value}')
            
            # Update recent metrics (with expiration)
            pipe.zadd('athena:recent:scores', {str(timestamp): evaluation_result.overall_score})
            pipe.expire('athena:recent:scores', 3600)  # 1 hour
            
            pipe.execute()
            
        except Exception as e:
            logger.warning(f"Redis update failed: {e}")
    
    async def _run_pattern_detection(self) -> None:
        """Run pattern detection algorithms on recent data."""
        if len(self.recent_evaluations) < 10:
            return  # Need minimum data for pattern detection
        
        try:
            for pattern_name, detector in self.pattern_detectors.items():
                patterns = await detector()
                
                for pattern in patterns:
                    pattern_id = f"{pattern_name}_{pattern.pattern_id}"
                    
                    if pattern_id in self.detected_patterns:
                        # Update existing pattern
                        existing = self.detected_patterns[pattern_id]
                        existing.frequency += 1
                        existing.last_seen = datetime.now()
                        existing.severity = max(existing.severity, pattern.severity)
                    else:
                        # New pattern detected
                        self.detected_patterns[pattern_id] = pattern
                        
                        # Generate alert if pattern is significant
                        if pattern.frequency >= self.alert_thresholds['pattern_frequency']:
                            await self._generate_pattern_alert(pattern)
            
            # Update metrics
            if self.metrics:
                self.metrics['active_patterns'].set(len(self.detected_patterns))
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
    
    async def _detect_recurring_violations(self) -> List[ContentPattern]:
        """Detect recurring violation patterns."""
        patterns = []
        
        # Group evaluations by issue categories
        issue_groups = defaultdict(list)
        
        for eval_data in list(self.recent_evaluations)[-100:]:  # Last 100 evaluations
            result = eval_data['result']
            for issue in result.issues:
                if issue.severity > 0.7:  # High severity issues only
                    issue_groups[issue.category.value].append(eval_data)
        
        # Detect patterns
        for category, evaluations in issue_groups.items():
            if len(evaluations) >= 5:  # At least 5 occurrences
                pattern = ContentPattern(
                    pattern_id=f"recurring_{category}",
                    pattern_type="recurring_violations",
                    description=f"Recurring {category} violations detected",
                    frequency=len(evaluations),
                    severity=np.mean([
                        max(issue.severity for issue in eval_data['result'].issues 
                            if issue.category.value == category)
                        for eval_data in evaluations
                    ]),
                    first_seen=min(eval_data['timestamp'] for eval_data in evaluations),
                    last_seen=max(eval_data['timestamp'] for eval_data in evaluations)
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_escalating_severity(self) -> List[ContentPattern]:
        """Detect escalating severity patterns."""
        patterns = []
        
        if len(self.recent_evaluations) < 20:
            return patterns
        
        # Get recent scores
        recent_scores = [eval_data['result'].overall_score 
                        for eval_data in list(self.recent_evaluations)[-20:]]
        
        # Calculate trend
        x = np.arange(len(recent_scores))
        slope = np.polyfit(x, recent_scores, 1)[0]
        
        # Negative slope indicates declining scores (increasing problems)
        if slope < -0.02:  # Significant negative trend
            pattern = ContentPattern(
                pattern_id="escalating_severity",
                pattern_type="escalating_severity",
                description=f"Escalating severity trend detected (slope: {slope:.3f})",
                frequency=len(recent_scores),
                severity=abs(slope) * 10,  # Convert slope to severity
                first_seen=list(self.recent_evaluations)[-20]['timestamp'],
                last_seen=list(self.recent_evaluations)[-1]['timestamp']
            )
            patterns.append(pattern)
        
        return patterns
    
    async def _detect_modality_anomalies(self) -> List[ContentPattern]:
        """Detect anomalies in modality usage patterns."""
        patterns = []
        
        # Count modality usage
        modality_counts = defaultdict(int)
        for eval_data in list(self.recent_evaluations)[-50:]:
            for modality in eval_data['result'].modality_scores.keys():
                modality_counts[modality] += 1
        
        if not modality_counts:
            return patterns
        
        # Calculate expected distribution (assume roughly equal)
        expected_count = sum(modality_counts.values()) / len(modality_counts)
        
        # Detect anomalies
        for modality, count in modality_counts.items():
            deviation = abs(count - expected_count) / expected_count
            
            if deviation > 0.5:  # 50% deviation from expected
                pattern = ContentPattern(
                    pattern_id=f"modality_anomaly_{modality}",
                    pattern_type="modality_anomalies",
                    description=f"Unusual {modality} usage pattern (deviation: {deviation:.1%})",
                    frequency=count,
                    severity=min(deviation, 1.0),
                    first_seen=datetime.now() - timedelta(minutes=30),
                    last_seen=datetime.now()
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _detect_temporal_clusters(self) -> List[ContentPattern]:
        """Detect temporal clustering of issues."""
        patterns = []
        
        # Group evaluations by time windows
        now = datetime.now()
        time_buckets = defaultdict(list)
        
        for eval_data in self.recent_evaluations:
            # 5-minute buckets
            bucket = eval_data['timestamp'].replace(second=0, microsecond=0)
            bucket = bucket.replace(minute=bucket.minute // 5 * 5)
            time_buckets[bucket].append(eval_data)
        
        # Detect clusters
        for bucket, evaluations in time_buckets.items():
            if len(evaluations) >= 10:  # High volume in short time
                issue_count = sum(len(eval_data['result'].issues) for eval_data in evaluations)
                
                if issue_count >= 15:  # Many issues in cluster
                    pattern = ContentPattern(
                        pattern_id=f"temporal_cluster_{bucket.strftime('%Y%m%d_%H%M')}",
                        pattern_type="temporal_clusters",
                        description=f"High issue density at {bucket.strftime('%H:%M')} ({issue_count} issues in {len(evaluations)} evaluations)",
                        frequency=len(evaluations),
                        severity=min(issue_count / len(evaluations) / 3, 1.0),
                        first_seen=bucket,
                        last_seen=bucket + timedelta(minutes=5)
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _check_alert_conditions(
        self, 
        evaluation_result: EvaluationResult, 
        timestamp: datetime
    ) -> None:
        """Check if alert conditions are met."""
        
        # Critical violation rate alert
        recent_critical = sum(
            1 for eval_data in list(self.recent_evaluations)[-50:]
            if eval_data['result'].compliance_status == ComplianceStatus.CRITICAL
        )
        
        if len(self.recent_evaluations) >= 20:
            critical_rate = recent_critical / min(len(self.recent_evaluations), 50)
            
            if critical_rate > self.alert_thresholds['critical_violation_rate']:
                await self._generate_alert(
                    alert_type="high_critical_rate",
                    severity=min(critical_rate * 2, 1.0),
                    message=f"High critical violation rate: {critical_rate:.1%} (threshold: {self.alert_thresholds['critical_violation_rate']:.1%})"
                )
        
        # Processing time alert
        if len(self.recent_evaluations) >= 10:
            recent_times = [eval_data['processing_time'] for eval_data in list(self.recent_evaluations)[-10:]]
            avg_time = np.mean(recent_times)
            
            if avg_time > self.alert_thresholds['avg_processing_time']:
                await self._generate_alert(
                    alert_type="slow_processing",
                    severity=min(avg_time / self.alert_thresholds['avg_processing_time'] - 1, 1.0),
                    message=f"Slow processing detected: {avg_time:.1f}s average (threshold: {self.alert_thresholds['avg_processing_time']}s)"
                )
        
        # Evaluation rate drop alert
        if len(self.evaluation_timestamps) >= 50:
            recent_period = datetime.now() - timedelta(minutes=5)
            recent_count = sum(1 for ts in self.evaluation_timestamps if ts > recent_period)
            
            older_period = recent_period - timedelta(minutes=5)
            older_count = sum(1 for ts in self.evaluation_timestamps 
                            if older_period < ts <= recent_period)
            
            if older_count > 0:
                rate_change = (recent_count - older_count) / older_count
                
                if rate_change < -self.alert_thresholds['evaluation_rate_drop']:
                    await self._generate_alert(
                        alert_type="evaluation_rate_drop",
                        severity=min(abs(rate_change), 1.0),
                        message=f"Evaluation rate dropped by {abs(rate_change):.1%}"
                    )
    
    async def _generate_alert(
        self, 
        alert_type: str, 
        severity: float, 
        message: str, 
        content_ids: List[str] = None
    ) -> None:
        """Generate a monitoring alert."""
        
        alert = ContentAlert(
            alert_id=f"{alert_type}_{int(time.time())}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            content_ids=content_ids or []
        )
        
        self.alerts.append(alert)
        
        # Update metrics
        if self.metrics:
            self.metrics['alerts_total'].labels(type=alert_type).inc()
        
        # Store in database
        if self.engine and self.Session:
            try:
                session = self.Session()
                session.execute(
                    self.alerts_table.insert().values(
                        alert_id=alert.alert_id,
                        alert_type=alert.alert_type,
                        severity=alert.severity,
                        message=alert.message,
                        timestamp=alert.timestamp,
                        acknowledged=0
                    )
                )
                session.commit()
                session.close()
            except Exception as e:
                logger.warning(f"Alert storage failed: {e}")
        
        logger.warning(f"ALERT [{alert_type}]: {message} (severity: {severity:.2f})")
    
    async def _generate_pattern_alert(self, pattern: ContentPattern) -> None:
        """Generate alert for detected pattern."""
        
        await self._generate_alert(
            alert_type=f"pattern_{pattern.pattern_type}",
            severity=pattern.severity,
            message=f"Pattern detected: {pattern.description}",
            content_ids=pattern.examples
        )
    
    def get_current_metrics(self) -> MonitoringMetrics:
        """Get current monitoring metrics."""
        
        metrics = MonitoringMetrics()
        
        if not self.recent_evaluations:
            return metrics
        
        # Basic counts
        metrics.total_evaluations = len(self.recent_evaluations)
        
        # Evaluations per second (last 5 minutes)
        recent_period = datetime.now() - timedelta(minutes=5)
        recent_count = sum(1 for ts in self.evaluation_timestamps if ts > recent_period)
        metrics.evaluations_per_second = recent_count / 300.0  # 5 minutes = 300 seconds
        
        # Average processing time
        processing_times = [eval_data['processing_time'] for eval_data in self.recent_evaluations]
        metrics.avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        # Compliance rate
        compliant_count = sum(
            1 for eval_data in self.recent_evaluations
            if eval_data['result'].compliance_status == ComplianceStatus.COMPLIANT
        )
        metrics.compliance_rate = compliant_count / len(self.recent_evaluations)
        
        # Violation counts
        metrics.critical_violations = sum(
            1 for eval_data in self.recent_evaluations
            if eval_data['result'].compliance_status == ComplianceStatus.CRITICAL
        )
        
        metrics.warnings = sum(
            1 for eval_data in self.recent_evaluations
            if eval_data['result'].compliance_status == ComplianceStatus.WARNING
        )
        
        # Modality distribution
        for eval_data in self.recent_evaluations:
            for modality in eval_data['result'].modality_scores.keys():
                metrics.modality_distribution[modality] = metrics.modality_distribution.get(modality, 0) + 1
        
        # Issue distribution
        for eval_data in self.recent_evaluations:
            for issue in eval_data['result'].issues:
                category = issue.category.value
                metrics.issue_distribution[category] = metrics.issue_distribution.get(category, 0) + 1
        
        return metrics
    
    def get_recent_alerts(self, limit: int = 10) -> List[ContentAlert]:
        """Get recent alerts."""
        return list(self.alerts)[-limit:]
    
    def get_detected_patterns(self) -> List[ContentPattern]:
        """Get currently detected patterns."""
        return list(self.detected_patterns.values())
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                
                # Update database
                if self.engine and self.Session:
                    try:
                        session = self.Session()
                        session.execute(
                            self.alerts_table.update().where(
                                self.alerts_table.c.alert_id == alert_id
                            ).values(acknowledged=1)
                        )
                        session.commit()
                        session.close()
                    except Exception as e:
                        logger.warning(f"Alert acknowledgment update failed: {e}")
                
                return True
        
        return False
    
    async def start_monitoring(self) -> None:
        """Start the monitoring system."""
        self.monitoring_active = True
        logger.info("Content monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop the monitoring system."""
        self.monitoring_active = False
        logger.info("Content monitoring stopped")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the monitoring system."""
        logger.info("Shutting down Content Monitor...")
        
        await self.stop_monitoring()
        
        # Close database connections
        if self.engine:
            self.engine.dispose()
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Content Monitor shutdown complete")