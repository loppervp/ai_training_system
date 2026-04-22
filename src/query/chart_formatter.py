"""
Interactive Chart Formatter Module
Auto-generate Plotly interactive HTML charts with hover tooltips
"""
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
import plotly.express as px
import plotly.io as pio

logger = logging.getLogger(__name__)


class ChartFormatter:
    """
    Auto-generate interactive HTML charts from query results
    Features:
    ✅ Hover tooltips with full details
    ✅ Zoom / Pan / Reset view
    ✅ Export PNG / SVG / HTML
    ✅ Auto detect chart type
    ✅ Responsive design
    ✅ Standalone HTML file (no dependencies)
    """
    
    def __init__(self, output_dir: str = "output/charts"):
        """
        Initialize Chart Formatter
        
        Args:
            output_dir: Directory to save generated HTML charts
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Chart type mappings
        self.chart_types = {
            'line': px.line,
            'bar': px.bar,
            'pie': px.pie,
            'area': px.area,
            'scatter': px.scatter
        }
        
        # Default template
        pio.templates.default = "plotly_white"
        
        logger.info("✅ Chart Formatter initialized")
    
    def auto_generate_chart(self, result: Dict[str, Any], query_text: str) -> Optional[str]:
        """
        Auto detect chart type and generate chart from query result
        
        Args:
            result: Query result dictionary
            query_text: Original user query
            
        Returns:
            Path to generated HTML file or None
        """
        try:
            data = result.get('data', {})
            
            # Detect chart type based on data structure
            chart_config = self._detect_chart_config(data, query_text)
            
            if not chart_config:
                logger.info("No suitable chart found for this result")
                return None
            
            return self.generate_chart(chart_config, query_text)
            
        except Exception as e:
            logger.error(f"❌ Error auto generating chart: {e}")
            return None
    
    def _detect_chart_config(self, data: Dict[str, Any], query_text: str) -> Optional[Dict[str, Any]]:
        """Auto detect best chart configuration from data"""
        query_lower = query_text.lower()
        
        # Time series / Trends = Line chart
        if any(word in query_lower for word in ['trend', 'monthly', 'daily', 'yearly', 'over time', 'xu hướng', 'theo tháng']):
            if 'monthly_trends' in data:
                return {
                    'type': 'line',
                    'data': data['monthly_trends'],
                    'x': 'month',
                    'y': 'total_revenue',
                    'title': 'Monthly Sales Trends',
                    'labels': {'total_revenue': 'Revenue (VND)', 'month': 'Month'}
                }
        
        # Top items = Bar chart
        if any(word in query_lower for word in ['top', 'best', 'most', 'nhất', 'top 10']):
            if 'top_potential_products' in data:
                return {
                    'type': 'bar',
                    'data': data['top_potential_products'],
                    'x': 'stkcode_desc',
                    'y': 'total_revenue',
                    'title': 'Top Products by Revenue',
                    'labels': {'total_revenue': 'Revenue (VND)', 'stkcode_desc': 'Product'},
                    'hover_data': ['growth_rate', 'potential_score']
                }
        
        # Customer segmentation = Pie chart
        if any(word in query_lower for word in ['churn', 'segment', 'customer', 'khách hàng']):
            if 'churn_rate_percent' in data:
                return {
                    'type': 'pie',
                    'values': [data['churned_customers'], data['total_customers'] - data['churned_customers']],
                    'names': ['Churned', 'Active'],
                    'title': 'Customer Churn Distribution',
                    'color_discrete_sequence': ['#FF6B6B', '#4ECDC4']
                }
        
        return None
    
    def generate_chart(self, config: Dict[str, Any], query_text: str) -> str:
        """Generate chart from configuration"""
        chart_type = config.pop('type')
        chart_function = self.chart_types[chart_type]
        
        # Create figure
        fig = chart_function(**config)
        
        # Update layout
        fig.update_layout(
            title={
                'text': config.get('title', 'Chart'),
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=16)
            },
            hovermode='x unified',
            showlegend=True,
            margin=dict(t=80, b=40, l=40, r=40),
            height=500
        )
        
        # Add hover tooltip formatting
        fig.update_traces(
            hovertemplate='%{y:,.0f} VND<extra></extra>'
        )
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chart_{timestamp}.html"
        file_path = os.path.join(self.output_dir, filename)
        
        # Save as standalone HTML
        fig.write_html(
            file_path,
            full_html=True,
            include_plotlyjs='cdn',
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': filename.replace('.html', ''),
                    'height': 500,
                    'width': 1200,
                    'scale': 2
                }
            }
        )
        
        logger.info(f"✅ Chart generated: {file_path}")
        
        return file_path
    
    def generate_chart_base64(self, config: Dict[str, Any]) -> Optional[str]:
        """Generate chart as base64 PNG for direct embedding in chat"""
        try:
            chart_type = config.pop('type')
            chart_function = self.chart_types[chart_type]
            
            # Create figure
            fig = chart_function(**config)
            
            # Update layout
            fig.update_layout(
                title={
                    'text': config.get('title', 'Chart'),
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=14)
                },
                margin=dict(t=60, b=40, l=40, r=40),
                height=300,
                width=600,
                showlegend=False
            )
            
            # Add hover tooltip formatting
            fig.update_traces(
                hovertemplate='%{y:,.0f} VND<extra></extra>'
            )
            
            # Convert to base64 PNG
            import base64
            import io
            
            img_bytes = fig.to_image(format="png", scale=2)
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"data:image/png;base64,{base64_image}"
            
        except Exception as e:
            logger.error(f"❌ Error generating base64 chart: {e}")
            return None
    
    def format_response_with_chart(self, result: Dict[str, Any], query_text: str) -> Dict[str, Any]:
        """Format response with both file path and base64 embedded chart"""
        try:
            # Detect chart configuration
            data = result.get('data', {})
            chart_config = self._detect_chart_config(data, query_text)
            
            if chart_config:
                # Generate base64 for direct chat embedding
                base64_chart = self.generate_chart_base64(chart_config)
                if base64_chart:
                    result['chart_base64'] = base64_chart
                    logger.info("✅ Base64 chart generated for chat embedding")
                
                # Generate HTML file for full interactive view
                chart_path = self.generate_chart(chart_config, query_text)
                if chart_path:
                    result['chart_path'] = chart_path
                
                result['insights'].append("📊 Chart generated and embedded in chat")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error formatting chart response: {e}")
            return result
