"""
Logging and Metrics Utilities for Project Athena

Advanced logging, audit trails, metrics collection, and monitoring
for comprehensive observability of the ethics evaluation system.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import json
import time
import os
import sys
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import threading

# Structured logging imports
try:
    import structlog
    from pythonjsonlogger import jsonlogger
    import colorlog
except ImportError as e:
    logging.warning(f"Some advanced logging dependencies not available: {e}")

# Metrics and monitoring imports
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry
    import psutil
except ImportError as e:
    logging.warning(f"Some metrics dependencies not available: {e}")

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Enhanced log levels for ethics evaluation."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"
    SECURITY = "SECURITY"

@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: datetime
    level: str
    message: str
    module: str
    function: str
    line_number: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    content_id: Optional[str] = None
    evaluation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram, summary

@dataclass
class AuditRecord:
    """Comprehensive audit record."""
    audit_id: str
    timestamp: datetime
    event_type: str
    actor: str
    resource: str
    action: str
    result: str
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    context: Dict[str, Any] = field(default_factory=dict)
    compliance_impact: Optional[str] = None

class LogManager:
    """
    Advanced logging management system.
    
    Provides structured logging, multiple handlers, filtering,
    and integration with external logging systems.
    """
    
    def __init__(self, config):
        """Initialize log manager."""
        self.config = config
        
        # Logging configuration
        self.log_config = {
            "level": getattr(config, "logging_level", "INFO"),
            "format": "structured",  # structured, json, traditional
            "enable_console": True,
            "enable_file": True,
            "enable_audit": True,
            "log_directory": "logs",
            "max_file_size": 100 * 1024 * 1024,  # 100MB
            "backup_count": 5,
            "buffer_size": 1000
        }
        
        # Initialize loggers
        self._setup_loggers()
        
        # Log buffer for real-time monitoring
        self.log_buffer = deque(maxlen=self.log_config["buffer_size"])
        self.log_filters = []
        self.log_handlers = []
        
        # Performance tracking
        self.log_stats = {
            "total_logs": 0,
            "logs_by_level": defaultdict(int),
            "logs_by_module": defaultdict(int),
            "errors": defaultdict(int)
        }
        
        logger.info("Log Manager initialized")
    
    def _setup_loggers(self) -> None:
        """Setup logging infrastructure."""
        
        # Create logs directory
        log_dir = Path(self.log_config["log_directory"])
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_config["level"]))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors
        if self.log_config["enable_console"]:
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.log_config["enable_file"]:
            file_handler = self._create_file_handler()
            root_logger.addHandler(file_handler)
        
        # Structured/JSON handler
        if self.log_config["format"] == "structured":
            structured_handler = self._create_structured_handler()
            root_logger.addHandler(structured_handler)
        
        # Custom handler for log buffering
        buffer_handler = self._create_buffer_handler()
        root_logger.addHandler(buffer_handler)
    
    def _create_console_handler(self) -> logging.Handler:
        """Create colorized console handler."""
        handler = logging.StreamHandler(sys.stdout)
        
        if 'colorlog' in globals():
            formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                    'AUDIT': 'blue',
                    'SECURITY': 'magenta'
                }
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_file_handler(self) -> logging.Handler:
        """Create rotating file handler."""
        from logging.handlers import RotatingFileHandler
        
        log_file = Path(self.log_config["log_directory"]) / "athena.log"
        
        handler = RotatingFileHandler(
            log_file,
            maxBytes=self.log_config["max_file_size"],
            backupCount=self.log_config["backup_count"]
        )
        
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        return handler
    
    def _create_structured_handler(self) -> logging.Handler:
        """Create structured JSON handler."""
        from logging.handlers import RotatingFileHandler
        
        log_file = Path(self.log_config["log_directory"]) / "athena_structured.jsonl"
        
        handler = RotatingFileHandler(
            log_file,
            maxBytes=self.log_config["max_file_size"],
            backupCount=self.log_config["backup_count"]
        )
        
        if 'jsonlogger' in globals():
            formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s',
                timestamp=True
            )
        else:
            formatter = logging.Formatter('%(message)s')
        
        handler.setFormatter(formatter)
        return handler
    
    def _create_buffer_handler(self) -> logging.Handler:
        """Create handler that buffers logs for real-time monitoring."""
        
        class BufferHandler(logging.Handler):
            def __init__(self, log_manager):
                super().__init__()
                self.log_manager = log_manager
            
            def emit(self, record):
                try:
                    log_entry = LogEntry(
                        timestamp=datetime.fromtimestamp(record.created),
                        level=record.levelname,
                        message=record.getMessage(),
                        module=record.module if hasattr(record, 'module') else record.name,
                        function=record.funcName,
                        line_number=record.lineno,
                        metadata=getattr(record, 'metadata', {}),
                        tags=getattr(record, 'tags', [])
                    )
                    
                    self.log_manager.log_buffer.append(log_entry)
                    
                    # Update statistics
                    self.log_manager.log_stats["total_logs"] += 1
                    self.log_manager.log_stats["logs_by_level"][record.levelname] += 1
                    self.log_manager.log_stats["logs_by_module"][record.name] += 1
                    
                except Exception:
                    pass  # Don't let logging errors break the application
        
        return BufferHandler(self)
    
    def add_filter(self, filter_func: Callable[[LogEntry], bool]) -> None:
        """Add custom log filter."""
        self.log_filters.append(filter_func)
    
    def get_logs(
        self,
        level: Optional[str] = None,
        module: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """Get filtered logs from buffer."""
        
        logs = list(self.log_buffer)
        
        # Apply filters
        if level:
            logs = [log for log in logs if log.level == level]
        
        if module:
            logs = [log for log in logs if module in log.module]
        
        if start_time:
            logs = [log for log in logs if log.timestamp >= start_time]
        
        if end_time:
            logs = [log for log in logs if log.timestamp <= end_time]
        
        # Apply custom filters
        for filter_func in self.log_filters:
            logs = [log for log in logs if filter_func(log)]
        
        # Sort by timestamp (newest first) and limit
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return logs[:limit]
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "total_logs": self.log_stats["total_logs"],
            "buffer_size": len(self.log_buffer),
            "logs_by_level": dict(self.log_stats["logs_by_level"]),
            "logs_by_module": dict(self.log_stats["logs_by_module"]),
            "active_filters": len(self.log_filters),
            "log_directory": str(Path(self.log_config["log_directory"]).absolute())
        }

class AuditLogger:
    """
    Specialized audit logging for compliance and security.
    
    Provides immutable audit trails, compliance reporting,
    and forensic analysis capabilities.
    """
    
    def __init__(self, config):
        """Initialize audit logger."""
        self.config = config
        
        # Audit configuration
        self.audit_config = {
            "enabled": True,
            "audit_directory": "audit_logs",
            "retention_days": 2555,  # 7 years for compliance
            "encryption_enabled": True,
            "digital_signatures": True,
            "batch_size": 100,
            "flush_interval": 60  # seconds
        }
        
        # Audit storage
        self.audit_records = deque(maxlen=10000)
        self.audit_batch = []
        self.last_flush = time.time()
        
        # Setup audit infrastructure
        self._setup_audit_storage()
        
        # Compliance tracking
        self.compliance_events = defaultdict(list)
        
        logger.info("Audit Logger initialized")
    
    def _setup_audit_storage(self) -> None:
        """Setup audit log storage infrastructure."""
        
        # Create audit directory
        audit_dir = Path(self.audit_config["audit_directory"])
        audit_dir.mkdir(exist_ok=True)
        
        # Setup audit file handler
        audit_file = audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        self.audit_handler = logging.FileHandler(audit_file)
        
        formatter = logging.Formatter('%(message)s')
        self.audit_handler.setFormatter(formatter)
        
        # Create audit logger
        self.audit_logger = logging.getLogger('athena.audit')
        self.audit_logger.setLevel(logging.INFO)
        self.audit_logger.addHandler(self.audit_handler)
        self.audit_logger.propagate = False
    
    async def log_audit_event(
        self,
        event_type: str,
        actor: str,
        resource: str,
        action: str,
        result: str,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        compliance_impact: Optional[str] = None
    ) -> str:
        """Log audit event with full context."""
        
        import secrets
        
        audit_id = secrets.token_urlsafe(16)
        
        record = AuditRecord(
            audit_id=audit_id,
            timestamp=datetime.now(),
            event_type=event_type,
            actor=actor,
            resource=resource,
            action=action,
            result=result,
            before_state=before_state,
            after_state=after_state,
            context=context or {},
            compliance_impact=compliance_impact
        )
        
        # Add to buffer
        self.audit_records.append(record)
        self.audit_batch.append(record)
        
        # Track compliance events
        if compliance_impact:
            self.compliance_events[compliance_impact].append(record)
        
        # Flush if batch is full or time threshold reached
        current_time = time.time()
        if (len(self.audit_batch) >= self.audit_config["batch_size"] or
            current_time - self.last_flush >= self.audit_config["flush_interval"]):
            await self._flush_audit_batch()
        
        logger.debug(f"Audit event logged: {event_type} by {actor}")
        return audit_id
    
    async def _flush_audit_batch(self) -> None:
        """Flush audit batch to persistent storage."""
        
        if not self.audit_batch:
            return
        
        try:
            for record in self.audit_batch:
                # Convert to JSON
                audit_json = json.dumps(asdict(record), default=str, ensure_ascii=False)
                
                # Log to file
                self.audit_logger.info(audit_json)
            
            self.audit_batch.clear()
            self.last_flush = time.time()
            
        except Exception as e:
            logger.error(f"Audit batch flush failed: {e}")
    
    def get_audit_trail(
        self,
        resource: Optional[str] = None,
        actor: Optional[str] = None,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditRecord]:
        """Get filtered audit trail."""
        
        records = list(self.audit_records)
        
        # Apply filters
        if resource:
            records = [r for r in records if r.resource == resource]
        
        if actor:
            records = [r for r in records if r.actor == actor]
        
        if event_type:
            records = [r for r in records if r.event_type == event_type]
        
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]
        
        # Sort by timestamp (newest first) and limit
        records.sort(key=lambda x: x.timestamp, reverse=True)
        
        return records[:limit]
    
    def get_compliance_report(
        self,
        compliance_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate compliance report."""
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "period_start": start_date.isoformat() if start_date else None,
            "period_end": end_date.isoformat() if end_date else None,
            "compliance_events": {},
            "summary": {}
        }
        
        # Filter compliance events
        for impact_type, events in self.compliance_events.items():
            if compliance_type and impact_type != compliance_type:
                continue
            
            filtered_events = events
            if start_date:
                filtered_events = [e for e in filtered_events if e.timestamp >= start_date]
            if end_date:
                filtered_events = [e for e in filtered_events if e.timestamp <= end_date]
            
            report["compliance_events"][impact_type] = len(filtered_events)
        
        # Generate summary
        total_events = sum(report["compliance_events"].values())
        report["summary"] = {
            "total_compliance_events": total_events,
            "compliance_types": len(report["compliance_events"]),
            "audit_coverage": len(self.audit_records) > 0
        }
        
        return report

class MetricsCollector:
    """
    Comprehensive metrics collection and monitoring.
    
    Collects performance metrics, business metrics, and
    operational metrics for the ethics evaluation system.
    """
    
    def __init__(self, config):
        """Initialize metrics collector."""
        self.config = config
        
        # Metrics storage
        self.metrics_buffer = deque(maxlen=10000)
        self.aggregated_metrics = defaultdict(list)
        
        # Prometheus metrics
        self.prometheus_enabled = 'Counter' in globals()
        if self.prometheus_enabled:
            self._setup_prometheus_metrics()
        
        # System metrics
        self.system_metrics_enabled = 'psutil' in globals()
        
        # Collection settings
        self.collection_interval = 30  # seconds
        self.collection_active = False
        
        logger.info("Metrics Collector initialized")
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        
        # Create custom registry
        self.registry = CollectorRegistry()
        
        # Ethics evaluation metrics
        self.metrics = {
            'evaluations_total': Counter(
                'athena_evaluations_total',
                'Total number of ethics evaluations',
                ['modality', 'status'],
                registry=self.registry
            ),
            'evaluation_duration': Histogram(
                'athena_evaluation_duration_seconds',
                'Time spent on ethics evaluations',
                ['modality'],
                registry=self.registry
            ),
            'compliance_score': Histogram(
                'athena_compliance_score',
                'Ethics compliance scores',
                ['modality'],
                buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                registry=self.registry
            ),
            'issues_detected': Counter(
                'athena_issues_detected_total',
                'Total ethical issues detected',
                ['category', 'severity'],
                registry=self.registry
            ),
            'system_memory_usage': Gauge(
                'athena_system_memory_usage_bytes',
                'System memory usage',
                registry=self.registry
            ),
            'system_cpu_usage': Gauge(
                'athena_system_cpu_usage_percent',
                'System CPU usage percentage',
                registry=self.registry
            ),
            'model_cache_size': Gauge(
                'athena_model_cache_size_bytes',
                'Model cache size in bytes',
                registry=self.registry
            )
        }
    
    async def start_collection(self) -> None:
        """Start automatic metrics collection."""
        
        if self.collection_active:
            return
        
        self.collection_active = True
        
        # Start collection loop
        asyncio.create_task(self._collection_loop())
        
        logger.info("Metrics collection started")
    
    async def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        self.collection_active = False
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        
        while self.collection_active:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        
        if not self.system_metrics_enabled:
            return
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_metric("system.cpu.usage_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.record_metric("system.memory.total_bytes", memory.total)
            await self.record_metric("system.memory.available_bytes", memory.available)
            await self.record_metric("system.memory.used_bytes", memory.used)
            await self.record_metric("system.memory.usage_percent", memory.percent)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            await self.record_metric("system.disk.total_bytes", disk.total)
            await self.record_metric("system.disk.used_bytes", disk.used)
            await self.record_metric("system.disk.usage_percent", (disk.used / disk.total) * 100)
            
            # Update Prometheus metrics
            if self.prometheus_enabled:
                self.metrics['system_memory_usage'].set(memory.used)
                self.metrics['system_cpu_usage'].set(cpu_percent)
            
        except Exception as e:
            logger.warning(f"System metrics collection failed: {e}")
    
    async def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: str = "gauge"
    ) -> None:
        """Record a metric point."""
        
        metric_point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            metric_type=metric_type
        )
        
        # Add to buffer
        self.metrics_buffer.append(metric_point)
        
        # Add to aggregated metrics
        self.aggregated_metrics[name].append(metric_point)
        
        # Limit aggregated metrics size
        if len(self.aggregated_metrics[name]) > 1000:
            self.aggregated_metrics[name] = self.aggregated_metrics[name][-500:]
    
    async def record_evaluation_metrics(
        self,
        modality: str,
        duration: float,
        compliance_score: float,
        status: str,
        issues: List[Any]
    ) -> None:
        """Record ethics evaluation metrics."""
        
        # Record basic metrics
        await self.record_metric(f"evaluation.{modality}.duration", duration)
        await self.record_metric(f"evaluation.{modality}.compliance_score", compliance_score)
        
        # Count by status
        await self.record_metric(f"evaluation.{modality}.count", 1, {"status": status}, "counter")
        
        # Issue metrics
        for issue in issues:
            category = getattr(issue, 'category', 'unknown')
            severity = getattr(issue, 'severity', 0)
            
            severity_level = "low" if severity < 0.3 else "medium" if severity < 0.7 else "high"
            
            await self.record_metric(
                "evaluation.issues.count",
                1,
                {"category": str(category), "severity": severity_level},
                "counter"
            )
        
        # Update Prometheus metrics
        if self.prometheus_enabled:
            self.metrics['evaluations_total'].labels(modality=modality, status=status).inc()
            self.metrics['evaluation_duration'].labels(modality=modality).observe(duration)
            self.metrics['compliance_score'].labels(modality=modality).observe(compliance_score)
            
            for issue in issues:
                category = str(getattr(issue, 'category', 'unknown'))
                severity = getattr(issue, 'severity', 0)
                severity_level = "low" if severity < 0.3 else "medium" if severity < 0.7 else "high"
                
                self.metrics['issues_detected'].labels(
                    category=category,
                    severity=severity_level
                ).inc()
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MetricPoint]:
        """Get filtered metrics."""
        
        if metric_name:
            metrics = self.aggregated_metrics.get(metric_name, [])
        else:
            metrics = list(self.metrics_buffer)
        
        # Apply time filters
        if start_time:
            metrics = [m for m in metrics if m.timestamp >= start_time]
        
        if end_time:
            metrics = [m for m in metrics if m.timestamp <= end_time]
        
        # Sort by timestamp (newest first) and limit
        metrics.sort(key=lambda x: x.timestamp, reverse=True)
        
        return metrics[:limit]
    
    def get_metric_summary(
        self,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        
        metrics = self.get_metrics(metric_name, start_time, end_time, limit=10000)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[0] if values else 0
        }
    
    def get_metrics_stats(self) -> Dict[str, Any]:
        """Get metrics collection statistics."""
        
        return {
            "total_metrics": len(self.metrics_buffer),
            "unique_metric_names": len(self.aggregated_metrics),
            "prometheus_enabled": self.prometheus_enabled,
            "system_metrics_enabled": self.system_metrics_enabled,
            "collection_active": self.collection_active,
            "collection_interval": self.collection_interval
        }