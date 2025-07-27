"""
Ethics Dashboard for Project Athena

Real-time web dashboard for monitoring ethical evaluation performance,
visualizing metrics, and managing alerts with Meta AI integration.

Author: Michael Jaramillo (jmichaeloficial@gmail.com)
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Web framework and visualization imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    import pandas as pd
except ImportError as e:
    logging.warning(f"Some dashboard dependencies not available: {e}")

from ..core.evaluator import EvaluationResult, ComplianceStatus
from .content_monitor import ContentMonitor, MonitoringMetrics, ContentAlert

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for dashboard settings."""
    host: str = "0.0.0.0"
    port: int = 8080
    auto_refresh_interval: int = 5  # seconds
    max_chart_points: int = 100
    enable_websockets: bool = True

class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def broadcast(self, data: dict):
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

class EthicsDashboard:
    """
    Real-time ethics dashboard for Project Athena.
    
    Provides web-based visualization of ethical evaluation metrics,
    alerts, patterns, and system performance.
    """
    
    def __init__(self, config, content_monitor: Optional[ContentMonitor] = None):
        """
        Initialize ethics dashboard.
        
        Args:
            config: EthicsConfig instance
            content_monitor: ContentMonitor instance for data source
        """
        self.config = config
        self.content_monitor = content_monitor
        self.dashboard_config = DashboardConfig(port=config.api_port)
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Athena Ethics Dashboard",
            description="Real-time monitoring dashboard for Project Athena",
            version="1.0.0"
        )
        
        # WebSocket manager
        self.ws_manager = WebSocketManager()
        
        # Dashboard state
        self.dashboard_active = False
        self.update_task = None
        
        # Initialize components
        self._setup_routes()
        self._setup_templates()
        
        logger.info("Ethics Dashboard initialized")
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes for dashboard."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page."""
            return self.templates.TemplateResponse(
                "dashboard.html", 
                {"request": request, "title": "Athena Ethics Dashboard"}
            )
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get current monitoring metrics."""
            if not self.content_monitor:
                return {"error": "Content monitor not available"}
            
            metrics = self.content_monitor.get_current_metrics()
            return asdict(metrics)
        
        @self.app.get("/api/alerts")
        async def get_alerts(limit: int = 20):
            """Get recent alerts."""
            if not self.content_monitor:
                return {"alerts": []}
            
            alerts = self.content_monitor.get_recent_alerts(limit)
            return {"alerts": [asdict(alert) for alert in alerts]}
        
        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str):
            """Acknowledge an alert."""
            if not self.content_monitor:
                return {"success": False, "error": "Content monitor not available"}
            
            success = self.content_monitor.acknowledge_alert(alert_id)
            return {"success": success}
        
        @self.app.get("/api/patterns")
        async def get_patterns():
            """Get detected content patterns."""
            if not self.content_monitor:
                return {"patterns": []}
            
            patterns = self.content_monitor.get_detected_patterns()
            return {"patterns": [asdict(pattern) for pattern in patterns]}
        
        @self.app.get("/api/charts/compliance-trend")
        async def get_compliance_trend():
            """Get compliance trend chart data."""
            return await self._generate_compliance_trend_chart()
        
        @self.app.get("/api/charts/modality-distribution")
        async def get_modality_distribution():
            """Get modality distribution chart data."""
            return await self._generate_modality_distribution_chart()
        
        @self.app.get("/api/charts/issue-categories")
        async def get_issue_categories():
            """Get issue categories chart data."""
            return await self._generate_issue_categories_chart()
        
        @self.app.get("/api/charts/performance-metrics")
        async def get_performance_metrics():
            """Get performance metrics chart data."""
            return await self._generate_performance_metrics_chart()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await self.ws_manager.connect(websocket)
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.ws_manager.disconnect(websocket)
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "dashboard_active": self.dashboard_active,
                "content_monitor": self.content_monitor is not None,
                "timestamp": datetime.now().isoformat()
            }
    
    def _setup_templates(self) -> None:
        """Setup Jinja2 templates."""
        try:
            # Create templates directory if it doesn't exist
            templates_dir = Path("templates")
            templates_dir.mkdir(exist_ok=True)
            
            # Create static directory for CSS/JS
            static_dir = Path("static")
            static_dir.mkdir(exist_ok=True)
            
            self.templates = Jinja2Templates(directory=str(templates_dir))
            
            # Mount static files
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            
            # Create dashboard template if it doesn't exist
            self._create_dashboard_template()
            self._create_dashboard_styles()
            
        except Exception as e:
            logger.warning(f"Template setup failed: {e}")
            self.templates = None
    
    def _create_dashboard_template(self) -> None:
        """Create the main dashboard HTML template."""
        template_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="/static/dashboard.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://unpkg.com/htmx.org@1.8.4"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <header class="dashboard-header">
            <div class="header-content">
                <h1><i class="fas fa-shield-alt"></i> Athena Ethics Dashboard</h1>
                <div class="status-indicators">
                    <div id="connection-status" class="status-indicator">
                        <i class="fas fa-circle"></i> Connected
                    </div>
                    <div id="last-update" class="status-indicator">
                        Last Update: <span id="update-time">--</span>
                    </div>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="dashboard-main">
            <!-- Metrics Overview -->
            <section class="metrics-overview">
                <div class="metric-card">
                    <div class="metric-icon">
                        <i class="fas fa-chart-line"></i>
                    </div>
                    <div class="metric-content">
                        <h3>Total Evaluations</h3>
                        <span id="total-evaluations" class="metric-value">0</span>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-icon">
                        <i class="fas fa-check-circle"></i>
                    </div>
                    <div class="metric-content">
                        <h3>Compliance Rate</h3>
                        <span id="compliance-rate" class="metric-value">0%</span>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-icon">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                    <div class="metric-content">
                        <h3>Critical Violations</h3>
                        <span id="critical-violations" class="metric-value">0</span>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-icon">
                        <i class="fas fa-clock"></i>
                    </div>
                    <div class="metric-content">
                        <h3>Avg Processing Time</h3>
                        <span id="processing-time" class="metric-value">0s</span>
                    </div>
                </div>
            </section>

            <!-- Charts Section -->
            <section class="charts-section">
                <div class="chart-container">
                    <h3>Compliance Trend</h3>
                    <div id="compliance-trend-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Modality Distribution</h3>
                    <div id="modality-distribution-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Issue Categories</h3>
                    <div id="issue-categories-chart"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Performance Metrics</h3>
                    <div id="performance-metrics-chart"></div>
                </div>
            </section>

            <!-- Alerts Section -->
            <section class="alerts-section">
                <h3><i class="fas fa-bell"></i> Recent Alerts</h3>
                <div id="alerts-container" class="alerts-container">
                    <!-- Alerts will be populated here -->
                </div>
            </section>

            <!-- Patterns Section -->
            <section class="patterns-section">
                <h3><i class="fas fa-search"></i> Detected Patterns</h3>
                <div id="patterns-container" class="patterns-container">
                    <!-- Patterns will be populated here -->
                </div>
            </section>
        </main>
    </div>

    <script src="/static/dashboard.js"></script>
</body>
</html>
        '''
        
        template_file = Path("templates/dashboard.html")
        with open(template_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
    
    def _create_dashboard_styles(self) -> None:
        """Create dashboard CSS styles."""
        css_content = '''
/* Dashboard Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f5f5;
    color: #333;
}

.dashboard-container {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* Header */
.dashboard-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1200px;
    margin: 0 auto;
}

.header-content h1 {
    font-size: 1.5rem;
    font-weight: 600;
}

.status-indicators {
    display: flex;
    gap: 2rem;
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
}

.status-indicator .fa-circle {
    color: #4ade80;
    font-size: 0.7rem;
}

/* Main Content */
.dashboard-main {
    flex: 1;
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
    width: 100%;
}

/* Metrics Overview */
.metrics-overview {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-bottom: 2rem;
}

.metric-card {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: transform 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
}

.metric-icon {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 1.5rem;
}

.metric-content h3 {
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #333;
}

/* Charts Section */
.charts-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
    gap: 2rem;
    margin-bottom: 2rem;
}

.chart-container {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.chart-container h3 {
    margin-bottom: 1rem;
    color: #333;
    font-size: 1.1rem;
}

/* Alerts Section */
.alerts-section, .patterns-section {
    background: white;
    border-radius: 10px;
    padding: 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

.alerts-section h3, .patterns-section h3 {
    margin-bottom: 1rem;
    color: #333;
    font-size: 1.1rem;
}

.alert-item, .pattern-item {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    background: #fafafa;
}

.alert-item.critical {
    border-color: #ef4444;
    background: #fef2f2;
}

.alert-item.warning {
    border-color: #f59e0b;
    background: #fffbeb;
}

.alert-header, .pattern-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.alert-type, .pattern-type {
    font-weight: 600;
    color: #374151;
}

.alert-time, .pattern-time {
    font-size: 0.8rem;
    color: #6b7280;
}

.alert-message, .pattern-description {
    color: #4b5563;
    margin-bottom: 0.5rem;
}

.acknowledge-btn {
    background: #10b981;
    color: white;
    border: none;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
}

.acknowledge-btn:hover {
    background: #059669;
}

/* Responsive Design */
@media (max-width: 768px) {
    .dashboard-header {
        padding: 1rem;
    }
    
    .header-content {
        flex-direction: column;
        gap: 1rem;
    }
    
    .status-indicators {
        gap: 1rem;
    }
    
    .dashboard-main {
        padding: 1rem;
    }
    
    .charts-section {
        grid-template-columns: 1fr;
    }
    
    .chart-container {
        min-width: 0;
    }
}
        '''
        
        css_file = Path("static/dashboard.css")
        with open(css_file, 'w', encoding='utf-8') as f:
            f.write(css_content)
        
        # Create JavaScript file
        js_content = '''
// Dashboard JavaScript
class EthicsDashboard {
    constructor() {
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.init();
    }

    init() {
        this.connectWebSocket();
        this.startDataRefresh();
        this.setupEventListeners();
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.updateConnectionStatus(true);
            this.reconnectAttempts = 0;
        };
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            this.attemptReconnect();
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            setTimeout(() => {
                console.log(`Reconnection attempt ${this.reconnectAttempts}`);
                this.connectWebSocket();
            }, 2000 * this.reconnectAttempts);
        }
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        const icon = statusElement.querySelector('i');
        
        if (connected) {
            statusElement.innerHTML = '<i class="fas fa-circle"></i> Connected';
            icon.style.color = '#4ade80';
        } else {
            statusElement.innerHTML = '<i class="fas fa-circle"></i> Disconnected';
            icon.style.color = '#ef4444';
        }
    }

    handleWebSocketMessage(data) {
        // Handle real-time updates
        if (data.type === 'metrics_update') {
            this.updateMetrics(data.metrics);
        } else if (data.type === 'new_alert') {
            this.addAlert(data.alert);
        }
    }

    startDataRefresh() {
        this.refreshData();
        setInterval(() => {
            this.refreshData();
        }, 5000); // Refresh every 5 seconds
    }

    async refreshData() {
        try {
            await Promise.all([
                this.updateMetrics(),
                this.updateCharts(),
                this.updateAlerts(),
                this.updatePatterns()
            ]);
            
            document.getElementById('update-time').textContent = new Date().toLocaleTimeString();
        } catch (error) {
            console.error('Data refresh failed:', error);
        }
    }

    async updateMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const metrics = await response.json();
            
            document.getElementById('total-evaluations').textContent = metrics.total_evaluations || 0;
            document.getElementById('compliance-rate').textContent = 
                Math.round((metrics.compliance_rate || 0) * 100) + '%';
            document.getElementById('critical-violations').textContent = metrics.critical_violations || 0;
            document.getElementById('processing-time').textContent = 
                (metrics.avg_processing_time || 0).toFixed(1) + 's';
        } catch (error) {
            console.error('Metrics update failed:', error);
        }
    }

    async updateCharts() {
        try {
            // Update all charts
            await Promise.all([
                this.updateComplianceTrendChart(),
                this.updateModalityDistributionChart(),
                this.updateIssueCategoriesChart(),
                this.updatePerformanceMetricsChart()
            ]);
        } catch (error) {
            console.error('Charts update failed:', error);
        }
    }

    async updateComplianceTrendChart() {
        try {
            const response = await fetch('/api/charts/compliance-trend');
            const chartData = await response.json();
            
            Plotly.newPlot('compliance-trend-chart', chartData.data, chartData.layout, {
                responsive: true,
                displayModeBar: false
            });
        } catch (error) {
            console.error('Compliance trend chart update failed:', error);
        }
    }

    async updateModalityDistributionChart() {
        try {
            const response = await fetch('/api/charts/modality-distribution');
            const chartData = await response.json();
            
            Plotly.newPlot('modality-distribution-chart', chartData.data, chartData.layout, {
                responsive: true,
                displayModeBar: false
            });
        } catch (error) {
            console.error('Modality distribution chart update failed:', error);
        }
    }

    async updateIssueCategoriesChart() {
        try {
            const response = await fetch('/api/charts/issue-categories');
            const chartData = await response.json();
            
            Plotly.newPlot('issue-categories-chart', chartData.data, chartData.layout, {
                responsive: true,
                displayModeBar: false
            });
        } catch (error) {
            console.error('Issue categories chart update failed:', error);
        }
    }

    async updatePerformanceMetricsChart() {
        try {
            const response = await fetch('/api/charts/performance-metrics');
            const chartData = await response.json();
            
            Plotly.newPlot('performance-metrics-chart', chartData.data, chartData.layout, {
                responsive: true,
                displayModeBar: false
            });
        } catch (error) {
            console.error('Performance metrics chart update failed:', error);
        }
    }

    async updateAlerts() {
        try {
            const response = await fetch('/api/alerts');
            const data = await response.json();
            
            const container = document.getElementById('alerts-container');
            container.innerHTML = '';
            
            if (data.alerts && data.alerts.length > 0) {
                data.alerts.forEach(alert => {
                    this.addAlertToContainer(alert, container);
                });
            } else {
                container.innerHTML = '<p class="no-data">No recent alerts</p>';
            }
        } catch (error) {
            console.error('Alerts update failed:', error);
        }
    }

    addAlertToContainer(alert, container) {
        const alertElement = document.createElement('div');
        alertElement.className = `alert-item ${alert.severity > 0.7 ? 'critical' : 'warning'}`;
        
        alertElement.innerHTML = `
            <div class="alert-header">
                <span class="alert-type">${alert.alert_type}</span>
                <span class="alert-time">${new Date(alert.timestamp).toLocaleString()}</span>
            </div>
            <div class="alert-message">${alert.message}</div>
            ${!alert.acknowledged ? 
                `<button class="acknowledge-btn" onclick="dashboard.acknowledgeAlert('${alert.alert_id}')">
                    Acknowledge
                </button>` : 
                '<span class="acknowledged">✓ Acknowledged</span>'
            }
        `;
        
        container.appendChild(alertElement);
    }

    async updatePatterns() {
        try {
            const response = await fetch('/api/patterns');
            const data = await response.json();
            
            const container = document.getElementById('patterns-container');
            container.innerHTML = '';
            
            if (data.patterns && data.patterns.length > 0) {
                data.patterns.forEach(pattern => {
                    this.addPatternToContainer(pattern, container);
                });
            } else {
                container.innerHTML = '<p class="no-data">No patterns detected</p>';
            }
        } catch (error) {
            console.error('Patterns update failed:', error);
        }
    }

    addPatternToContainer(pattern, container) {
        const patternElement = document.createElement('div');
        patternElement.className = 'pattern-item';
        
        patternElement.innerHTML = `
            <div class="pattern-header">
                <span class="pattern-type">${pattern.pattern_type}</span>
                <span class="pattern-time">Frequency: ${pattern.frequency}</span>
            </div>
            <div class="pattern-description">${pattern.description}</div>
            <div class="pattern-details">
                Severity: ${(pattern.severity * 100).toFixed(1)}% | 
                First seen: ${new Date(pattern.first_seen).toLocaleString()}
            </div>
        `;
        
        container.appendChild(patternElement);
    }

    async acknowledgeAlert(alertId) {
        try {
            const response = await fetch(`/api/alerts/${alertId}/acknowledge`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.updateAlerts(); // Refresh alerts
            }
        } catch (error) {
            console.error('Alert acknowledgment failed:', error);
        }
    }

    setupEventListeners() {
        // Add any additional event listeners here
        window.addEventListener('resize', () => {
            // Resize charts on window resize
            setTimeout(() => {
                Plotly.Plots.resize('compliance-trend-chart');
                Plotly.Plots.resize('modality-distribution-chart');
                Plotly.Plots.resize('issue-categories-chart');
                Plotly.Plots.resize('performance-metrics-chart');
            }, 100);
        });
    }
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new EthicsDashboard();
});
        '''
        
        js_file = Path("static/dashboard.js")
        with open(js_file, 'w', encoding='utf-8') as f:
            f.write(js_content)
    
    async def _generate_compliance_trend_chart(self) -> Dict[str, Any]:
        """Generate compliance trend chart data."""
        if not self.content_monitor:
            return {"data": [], "layout": {}}
        
        # Get recent evaluations data (last 100)
        recent_evaluations = list(self.content_monitor.recent_evaluations)[-100:]
        
        if not recent_evaluations:
            return {"data": [], "layout": {"title": "No data available"}}
        
        # Extract timestamps and compliance scores
        timestamps = [eval_data['timestamp'] for eval_data in recent_evaluations]
        scores = [eval_data['result'].overall_score for eval_data in recent_evaluations]
        
        # Create trend line
        trace = go.Scatter(
            x=timestamps,
            y=scores,
            mode='lines+markers',
            name='Compliance Score',
            line=dict(color='#667eea', width=2),
            marker=dict(size=4)
        )
        
        layout = go.Layout(
            title="Compliance Score Trend",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Compliance Score", range=[0, 1]),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return {
            "data": [trace],
            "layout": layout
        }
    
    async def _generate_modality_distribution_chart(self) -> Dict[str, Any]:
        """Generate modality distribution chart data."""
        if not self.content_monitor:
            return {"data": [], "layout": {}}
        
        metrics = self.content_monitor.get_current_metrics()
        
        if not metrics.modality_distribution:
            return {"data": [], "layout": {"title": "No data available"}}
        
        # Create pie chart
        trace = go.Pie(
            labels=list(metrics.modality_distribution.keys()),
            values=list(metrics.modality_distribution.values()),
            hole=0.3,
            marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
        )
        
        layout = go.Layout(
            title="Content by Modality",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return {
            "data": [trace],
            "layout": layout
        }
    
    async def _generate_issue_categories_chart(self) -> Dict[str, Any]:
        """Generate issue categories chart data."""
        if not self.content_monitor:
            return {"data": [], "layout": {}}
        
        metrics = self.content_monitor.get_current_metrics()
        
        if not metrics.issue_distribution:
            return {"data": [], "layout": {"title": "No issues detected"}}
        
        # Create bar chart
        trace = go.Bar(
            x=list(metrics.issue_distribution.keys()),
            y=list(metrics.issue_distribution.values()),
            marker=dict(color='#f5576c')
        )
        
        layout = go.Layout(
            title="Issues by Category",
            xaxis=dict(title="Issue Category"),
            yaxis=dict(title="Count"),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return {
            "data": [trace],
            "layout": layout
        }
    
    async def _generate_performance_metrics_chart(self) -> Dict[str, Any]:
        """Generate performance metrics chart data."""
        if not self.content_monitor:
            return {"data": [], "layout": {}}
        
        # Get recent processing times
        recent_evaluations = list(self.content_monitor.recent_evaluations)[-50:]
        
        if not recent_evaluations:
            return {"data": [], "layout": {"title": "No performance data"}}
        
        timestamps = [eval_data['timestamp'] for eval_data in recent_evaluations]
        processing_times = [eval_data['processing_time'] for eval_data in recent_evaluations]
        
        # Create performance line chart
        trace = go.Scatter(
            x=timestamps,
            y=processing_times,
            mode='lines+markers',
            name='Processing Time',
            line=dict(color='#10b981', width=2),
            marker=dict(size=4)
        )
        
        layout = go.Layout(
            title="Processing Time Trend",
            xaxis=dict(title="Time"),
            yaxis=dict(title="Processing Time (seconds)"),
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return {
            "data": [trace],
            "layout": layout
        }
    
    async def start_dashboard(self) -> None:
        """Start the dashboard server."""
        self.dashboard_active = True
        
        # Start real-time update task
        self.update_task = asyncio.create_task(self._update_websocket_clients())
        
        logger.info(f"Starting dashboard server on {self.dashboard_config.host}:{self.dashboard_config.port}")
        
        # Start FastAPI server
        config = uvicorn.Config(
            self.app,
            host=self.dashboard_config.host,
            port=self.dashboard_config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    async def _update_websocket_clients(self) -> None:
        """Send real-time updates to WebSocket clients."""
        while self.dashboard_active:
            try:
                if self.content_monitor:
                    metrics = self.content_monitor.get_current_metrics()
                    
                    await self.ws_manager.broadcast({
                        "type": "metrics_update",
                        "metrics": asdict(metrics),
                        "timestamp": datetime.now().isoformat()
                    })
                
                await asyncio.sleep(self.dashboard_config.auto_refresh_interval)
                
            except Exception as e:
                logger.error(f"WebSocket update failed: {e}")
                await asyncio.sleep(5)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the dashboard."""
        logger.info("Shutting down Ethics Dashboard...")
        
        self.dashboard_active = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Ethics Dashboard shutdown complete")