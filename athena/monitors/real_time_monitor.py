"""
Real-time Monitor for Project Athena

High-performance real-time monitoring system for ethical evaluations
with advanced streaming analytics and alerting capabilities.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import json
from collections import deque, defaultdict
import numpy as np
from threading import Lock
import uuid

# Streaming and queue imports
try:
    import asyncio
    from asyncio import Queue
    import aioredis
    from kafka import KafkaProducer, KafkaConsumer
    import aiokafka
except ImportError as e:
    logging.warning(f"Some streaming dependencies not available: {e}")

from ..core.evaluator import EvaluationResult, EthicsIssue, ComplianceStatus
from .content_monitor import ContentMonitor, ContentAlert

logger = logging.getLogger(__name__)

@dataclass
class StreamingEvent:
    """Event for streaming monitoring."""
    event_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "athena"
    severity: float = 0.0

@dataclass
class MonitoringRule:
    """Rule for real-time monitoring."""
    rule_id: str
    name: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    action: Callable[[Dict[str, Any]], Any]
    enabled: bool = True
    threshold: float = 0.8
    cooldown_seconds: int = 60
    last_triggered: Optional[datetime] = None

@dataclass
class StreamMetrics:
    """Real-time streaming metrics."""
    events_per_second: float = 0.0
    total_events: int = 0
    error_rate: float = 0.0
    latency_p95: float = 0.0
    queue_size: int = 0
    active_rules: int = 0
    triggered_rules: int = 0

class EventProcessor:
    """High-performance event processor for streaming analytics."""
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 1.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.event_queue = Queue(maxsize=10000)
        self.processing_active = False
        self.batch_buffer = []
        self.last_flush = time.time()
        self.processors = []
        
    async def add_event(self, event: StreamingEvent) -> None:
        """Add event to processing queue."""
        try:
            await self.event_queue.put(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")
    
    def add_processor(self, processor: Callable[[List[StreamingEvent]], None]) -> None:
        """Add event processor function."""
        self.processors.append(processor)
    
    async def start_processing(self) -> None:
        """Start event processing loop."""
        self.processing_active = True
        
        while self.processing_active:
            try:
                # Get events from queue
                while len(self.batch_buffer) < self.batch_size:
                    try:
                        event = await asyncio.wait_for(
                            self.event_queue.get(), 
                            timeout=0.1
                        )
                        self.batch_buffer.append(event)
                    except asyncio.TimeoutError:
                        break
                
                # Flush if batch is full or timeout reached
                current_time = time.time()
                should_flush = (
                    len(self.batch_buffer) >= self.batch_size or
                    (self.batch_buffer and current_time - self.last_flush >= self.flush_interval)
                )
                
                if should_flush and self.batch_buffer:
                    await self._process_batch()
                
                if not self.batch_buffer:
                    await asyncio.sleep(0.01)  # Small delay if no events
                    
            except Exception as e:
                logger.error(f"Event processing error: {e}")
                await asyncio.sleep(0.1)
    
    async def _process_batch(self) -> None:
        """Process a batch of events."""
        batch = self.batch_buffer.copy()
        self.batch_buffer.clear()
        self.last_flush = time.time()
        
        # Process with all registered processors
        for processor in self.processors:
            try:
                await processor(batch)
            except Exception as e:
                logger.error(f"Event processor failed: {e}")
    
    async def stop_processing(self) -> None:
        """Stop event processing."""
        self.processing_active = False

class RealTimeMonitor:
    """
    High-performance real-time monitoring system.
    
    Provides streaming analytics, real-time alerting, and
    advanced pattern detection for ethical evaluations.
    """
    
    def __init__(self, config, content_monitor: Optional[ContentMonitor] = None):
        """
        Initialize real-time monitor.
        
        Args:
            config: EthicsConfig instance
            content_monitor: ContentMonitor instance for integration
        """
        self.config = config
        self.content_monitor = content_monitor
        
        # Real-time processing
        self.event_processor = EventProcessor()
        self.monitoring_rules = {}
        self.active_alerts = {}
        
        # Streaming infrastructure
        self._initialize_streaming()
        
        # Metrics and performance
        self.metrics = StreamMetrics()
        self.performance_tracker = deque(maxlen=1000)
        self.metrics_lock = Lock()
        
        # Event handlers
        self.event_handlers = defaultdict(list)
        
        # State tracking
        self.monitoring_active = False
        self.processing_tasks = []
        
        logger.info("Real-time Monitor initialized")
    
    def _initialize_streaming(self) -> None:
        """Initialize streaming infrastructure."""
        try:
            # Initialize Redis for pub/sub
            self.redis_config = {
                'host': 'localhost',
                'port': 6379,
                'decode_responses': True
            }
            
            # Initialize Kafka for high-throughput streaming
            self.kafka_config = {
                'bootstrap_servers': ['localhost:9092'],
                'topic': 'athena-ethics-events'
            }
            
            # Event stream settings
            self.stream_settings = {
                'max_batch_size': 1000,
                'max_latency_ms': 100,
                'compression': 'gzip',
                'retention_hours': 24
            }
            
        except Exception as e:
            logger.warning(f"Streaming infrastructure initialization failed: {e}")
    
    async def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        if self.monitoring_active:
            logger.warning("Real-time monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start event processor
        processing_task = asyncio.create_task(self.event_processor.start_processing())
        self.processing_tasks.append(processing_task)
        
        # Start streaming tasks
        metrics_task = asyncio.create_task(self._update_metrics_loop())
        self.processing_tasks.append(metrics_task)
        
        # Initialize default processors
        self.event_processor.add_processor(self._process_evaluation_events)
        self.event_processor.add_processor(self._process_alert_events)
        self.event_processor.add_processor(self._update_stream_metrics)
        
        # Initialize default rules
        self._initialize_default_rules()
        
        logger.info("Real-time monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        # Stop event processor
        await self.event_processor.stop_processing()
        
        # Cancel all processing tasks
        for task in self.processing_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.processing_tasks.clear()
        
        logger.info("Real-time monitoring stopped")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default monitoring rules."""
        
        # High severity issue rule
        self.add_rule(MonitoringRule(
            rule_id="high_severity_issues",
            name="High Severity Issues",
            description="Trigger when high severity issues are detected",
            condition=lambda data: (
                data.get("result", {}).get("issues", []) and
                any(issue.get("severity", 0) > 0.8 for issue in data["result"]["issues"])
            ),
            action=self._handle_high_severity_alert,
            threshold=0.8,
            cooldown_seconds=30
        ))
        
        # Critical compliance violations rule
        self.add_rule(MonitoringRule(
            rule_id="critical_violations",
            name="Critical Compliance Violations",
            description="Trigger on critical compliance violations",
            condition=lambda data: (
                data.get("result", {}).get("compliance_status") == "critical"
            ),
            action=self._handle_critical_violation_alert,
            threshold=1.0,
            cooldown_seconds=10
        ))
        
        # Processing time anomaly rule
        self.add_rule(MonitoringRule(
            rule_id="processing_time_anomaly",
            name="Processing Time Anomaly",
            description="Trigger when processing time is unusually high",
            condition=lambda data: (
                data.get("processing_time", 0) > 10.0  # 10 seconds
            ),
            action=self._handle_performance_alert,
            threshold=0.9,
            cooldown_seconds=60
        ))
        
        # Rapid evaluation rate rule
        self.add_rule(MonitoringRule(
            rule_id="evaluation_rate_spike",
            name="Evaluation Rate Spike",
            description="Trigger when evaluation rate spikes unusually",
            condition=self._check_evaluation_rate_spike,
            action=self._handle_rate_spike_alert,
            threshold=2.0,  # 2x normal rate
            cooldown_seconds=120
        ))
    
    def add_rule(self, rule: MonitoringRule) -> None:
        """Add a monitoring rule."""
        self.monitoring_rules[rule.rule_id] = rule
        logger.info(f"Added monitoring rule: {rule.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a monitoring rule."""
        if rule_id in self.monitoring_rules:
            del self.monitoring_rules[rule_id]
            logger.info(f"Removed monitoring rule: {rule_id}")
            return True
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable a monitoring rule."""
        if rule_id in self.monitoring_rules:
            self.monitoring_rules[rule_id].enabled = True
            return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable a monitoring rule."""
        if rule_id in self.monitoring_rules:
            self.monitoring_rules[rule_id].enabled = False
            return True
        return False
    
    async def process_evaluation(
        self, 
        evaluation_result: EvaluationResult,
        content: Any,
        processing_time: float = 0.0
    ) -> None:
        """Process an evaluation result in real-time."""
        if not self.monitoring_active:
            return
        
        try:
            # Create streaming event
            event = StreamingEvent(
                event_id=str(uuid.uuid4()),
                event_type="evaluation",
                timestamp=datetime.now(),
                data={
                    "result": evaluation_result.to_dict(),
                    "content_type": type(content).__name__ if content else "unknown",
                    "processing_time": processing_time,
                    "modalities": list(evaluation_result.modality_scores.keys())
                }
            )
            
            # Add to processing queue
            await self.event_processor.add_event(event)
            
            # Update performance tracking
            self.performance_tracker.append({
                "timestamp": event.timestamp,
                "processing_time": processing_time,
                "overall_score": evaluation_result.overall_score,
                "issue_count": len(evaluation_result.issues)
            })
            
        except Exception as e:
            logger.error(f"Error processing evaluation: {e}")
    
    async def _process_evaluation_events(self, events: List[StreamingEvent]) -> None:
        """Process evaluation events in batch."""
        evaluation_events = [e for e in events if e.event_type == "evaluation"]
        
        if not evaluation_events:
            return
        
        # Apply monitoring rules
        for event in evaluation_events:
            await self._apply_monitoring_rules(event)
        
        # Update aggregated metrics
        with self.metrics_lock:
            self.metrics.total_events += len(evaluation_events)
    
    async def _process_alert_events(self, events: List[StreamingEvent]) -> None:
        """Process alert events in batch."""
        alert_events = [e for e in events if e.event_type == "alert"]
        
        for event in alert_events:
            # Forward to content monitor if available
            if self.content_monitor:
                try:
                    alert_data = event.data
                    # Process alert through content monitor
                    pass
                except Exception as e:
                    logger.error(f"Alert forwarding failed: {e}")
    
    async def _update_stream_metrics(self, events: List[StreamingEvent]) -> None:
        """Update streaming metrics from events."""
        if not events:
            return
        
        current_time = time.time()
        
        with self.metrics_lock:
            # Calculate events per second
            time_window = 60  # 1 minute window
            recent_events = [
                e for e in events 
                if (current_time - e.timestamp.timestamp()) < time_window
            ]
            
            self.metrics.events_per_second = len(recent_events) / min(time_window, current_time)
            self.metrics.queue_size = self.event_processor.event_queue.qsize()
            self.metrics.active_rules = sum(1 for rule in self.monitoring_rules.values() if rule.enabled)
    
    async def _apply_monitoring_rules(self, event: StreamingEvent) -> None:
        """Apply monitoring rules to an event."""
        for rule_id, rule in self.monitoring_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check cooldown
                if rule.last_triggered:
                    cooldown_elapsed = (datetime.now() - rule.last_triggered).total_seconds()
                    if cooldown_elapsed < rule.cooldown_seconds:
                        continue
                
                # Apply rule condition
                if rule.condition(event.data):
                    # Execute rule action
                    await rule.action(event.data)
                    
                    # Update rule state
                    rule.last_triggered = datetime.now()
                    
                    with self.metrics_lock:
                        self.metrics.triggered_rules += 1
                    
                    logger.info(f"Monitoring rule triggered: {rule.name}")
                
            except Exception as e:
                logger.error(f"Rule execution failed [{rule_id}]: {e}")
    
    async def _handle_high_severity_alert(self, data: Dict[str, Any]) -> None:
        """Handle high severity issue alert."""
        issues = data.get("result", {}).get("issues", [])
        high_severity_issues = [
            issue for issue in issues 
            if issue.get("severity", 0) > 0.8
        ]
        
        alert_event = StreamingEvent(
            event_id=str(uuid.uuid4()),
            event_type="alert",
            timestamp=datetime.now(),
            data={
                "alert_type": "high_severity_issues",
                "severity": max(issue.get("severity", 0) for issue in high_severity_issues),
                "message": f"Detected {len(high_severity_issues)} high-severity issues",
                "issues": high_severity_issues
            },
            severity=max(issue.get("severity", 0) for issue in high_severity_issues)
        )
        
        await self.event_processor.add_event(alert_event)
    
    async def _handle_critical_violation_alert(self, data: Dict[str, Any]) -> None:
        """Handle critical compliance violation alert."""
        alert_event = StreamingEvent(
            event_id=str(uuid.uuid4()),
            event_type="alert",
            timestamp=datetime.now(),
            data={
                "alert_type": "critical_violation",
                "severity": 1.0,
                "message": "Critical compliance violation detected",
                "compliance_status": data.get("result", {}).get("compliance_status"),
                "overall_score": data.get("result", {}).get("overall_score")
            },
            severity=1.0
        )
        
        await self.event_processor.add_event(alert_event)
    
    async def _handle_performance_alert(self, data: Dict[str, Any]) -> None:
        """Handle performance anomaly alert."""
        processing_time = data.get("processing_time", 0)
        
        alert_event = StreamingEvent(
            event_id=str(uuid.uuid4()),
            event_type="alert",
            timestamp=datetime.now(),
            data={
                "alert_type": "performance_anomaly",
                "severity": min(processing_time / 10.0, 1.0),
                "message": f"High processing time detected: {processing_time:.2f}s",
                "processing_time": processing_time
            },
            severity=min(processing_time / 10.0, 1.0)
        )
        
        await self.event_processor.add_event(alert_event)
    
    async def _handle_rate_spike_alert(self, data: Dict[str, Any]) -> None:
        """Handle evaluation rate spike alert."""
        alert_event = StreamingEvent(
            event_id=str(uuid.uuid4()),
            event_type="alert",
            timestamp=datetime.now(),
            data={
                "alert_type": "evaluation_rate_spike",
                "severity": 0.7,
                "message": "Unusual spike in evaluation rate detected",
                "current_rate": self.metrics.events_per_second
            },
            severity=0.7
        )
        
        await self.event_processor.add_event(alert_event)
    
    def _check_evaluation_rate_spike(self, data: Dict[str, Any]) -> bool:
        """Check if there's an evaluation rate spike."""
        # Calculate recent rate vs historical average
        if len(self.performance_tracker) < 100:
            return False
        
        recent_period = datetime.now() - timedelta(minutes=5)
        recent_count = sum(
            1 for entry in self.performance_tracker
            if entry["timestamp"] > recent_period
        )
        
        historical_period = recent_period - timedelta(minutes=30)
        historical_count = sum(
            1 for entry in self.performance_tracker
            if historical_period < entry["timestamp"] <= recent_period
        )
        
        if historical_count == 0:
            return False
        
        recent_rate = recent_count / 5  # per minute
        historical_rate = historical_count / 30  # per minute
        
        return recent_rate > historical_rate * 2.0  # 2x spike
    
    async def _update_metrics_loop(self) -> None:
        """Continuously update metrics."""
        while self.monitoring_active:
            try:
                await self._calculate_advanced_metrics()
                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Metrics update failed: {e}")
                await asyncio.sleep(5)
    
    async def _calculate_advanced_metrics(self) -> None:
        """Calculate advanced performance metrics."""
        if not self.performance_tracker:
            return
        
        with self.metrics_lock:
            # Calculate latency percentiles
            processing_times = [entry["processing_time"] for entry in self.performance_tracker]
            if processing_times:
                self.metrics.latency_p95 = np.percentile(processing_times, 95)
            
            # Calculate error rate (placeholder)
            self.metrics.error_rate = 0.0  # Would calculate from actual errors
    
    def get_stream_metrics(self) -> StreamMetrics:
        """Get current streaming metrics."""
        with self.metrics_lock:
            return StreamMetrics(
                events_per_second=self.metrics.events_per_second,
                total_events=self.metrics.total_events,
                error_rate=self.metrics.error_rate,
                latency_p95=self.metrics.latency_p95,
                queue_size=self.metrics.queue_size,
                active_rules=self.metrics.active_rules,
                triggered_rules=self.metrics.triggered_rules
            )
    
    def get_active_rules(self) -> List[Dict[str, Any]]:
        """Get list of active monitoring rules."""
        return [
            {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "description": rule.description,
                "enabled": rule.enabled,
                "threshold": rule.threshold,
                "last_triggered": rule.last_triggered.isoformat() if rule.last_triggered else None
            }
            for rule in self.monitoring_rules.values()
        ]
    
    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add custom event handler."""
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable) -> bool:
        """Remove event handler."""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            return True
        return False
    
    async def publish_custom_event(
        self, 
        event_type: str, 
        data: Dict[str, Any],
        severity: float = 0.0
    ) -> None:
        """Publish custom event to the stream."""
        event = StreamingEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            severity=severity
        )
        
        await self.event_processor.add_event(event)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown real-time monitor."""
        logger.info("Shutting down Real-time Monitor...")
        
        await self.stop_monitoring()
        
        logger.info("Real-time Monitor shutdown complete")