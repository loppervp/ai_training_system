"""
Intent Parser Module
Extracts intent, entities, and references from user queries
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Represents parsed intent"""
    query_type: str  # customer_analysis, product_analysis, etc.
    sub_type: str  # top_customers, repeat_customers, etc.
    confidence: float  # 0.0 to 1.0
    entities: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)  # Pronouns/references to resolve
    filters: Dict[str, Any] = field(default_factory=dict)


class EntityExtractor:
    """Extracts entities from queries"""
    
    # Month mapping
    MONTH_MAP = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12
    }
    
    @staticmethod
    def extract_dates(query: str) -> Dict[str, Any]:
        """Extract date entities from query"""
        entities = {}
        
        # Match MM/YYYY, MM.YYYY, MM-YYYY
        date_match = re.search(r'(\d{1,2})[./-](\d{4})', query)
        if date_match:
            month = int(date_match.group(1))
            year = int(date_match.group(2))
            entities['month'] = month
            entities['year'] = year
            entities['period'] = f"{month}/{year}"
            return entities
        
        # Match text month (January 2026, Jan 2026)
        month_match = re.search(
            r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(\d{4})',
            query,
            re.IGNORECASE
        )
        if month_match:
            month_text = month_match.group(1).lower()
            month = EntityExtractor.MONTH_MAP.get(month_text, 1)
            year = int(month_match.group(2))
            entities['month'] = month
            entities['year'] = year
            entities['period'] = f"{month}/{year}"
            return entities
        
        # Match year only
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            year = int(year_match.group(1))
            entities['year'] = year
            entities['period'] = f"Year {year}"
            return entities
        
        return entities
    
    @staticmethod
    def extract_numbers(query: str) -> Dict[str, Any]:
        """Extract numeric entities (amounts, counts)"""
        entities = {}
        
        # Extract top N (top 10, top 5, etc.)
        top_match = re.search(r'top\s+(\d+)', query, re.IGNORECASE)
        if top_match:
            entities['top_n'] = int(top_match.group(1))
        
        # Extract amounts (with currency symbols)
        amount_matches = re.findall(r'[\$€£]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', query)
        if amount_matches:
            entities['amounts'] = amount_matches
        
        return entities
    
    @staticmethod
    def extract_keywords(query: str) -> Dict[str, Any]:
        """Extract important keywords"""
        entities = {}
        
        # Time-related keywords
        time_keywords = ['last', 'previous', 'next', 'this', 'current', 'month', 'year', 'week', 'day']
        found_time = [kw for kw in time_keywords if kw in query.lower()]
        if found_time:
            entities['time_keywords'] = found_time
        
        # Comparison keywords
        comparison_keywords = ['vs', 'versus', 'compared', 'compare', 'growth', 'increase', 'decrease']
        found_comparison = [kw for kw in comparison_keywords if kw in query.lower()]
        if found_comparison:
            entities['comparison_keywords'] = found_comparison
        
        return entities
    
    @staticmethod
    def extract_all(query: str) -> Dict[str, Any]:
        """Extract all entities from query"""
        entities = {}
        entities.update(EntityExtractor.extract_dates(query))
        entities.update(EntityExtractor.extract_numbers(query))
        entities.update(EntityExtractor.extract_keywords(query))
        return entities


class ReferenceDetector:
    """Detects pronouns and references that need resolution"""
    
    PRONOUNS = ['they', 'them', 'those', 'that', 'it', 'these', 'this', 'he', 'she', 'his', 'her']
    
    @staticmethod
    def detect_references(query: str) -> List[str]:
        """Detect pronouns/references in query"""
        references = []
        query_lower = query.lower()
        
        for pronoun in ReferenceDetector.PRONOUNS:
            if pronoun in query_lower:
                references.append(pronoun)
        
        return references
    
    @staticmethod
    def has_reference(query: str) -> bool:
        """Check if query contains references"""
        return len(ReferenceDetector.detect_references(query)) > 0


class IntentParser:
    """Parses user queries into structured intents"""
    
    def __init__(self):
        """Initialize intent parser"""
        self.entity_extractor = EntityExtractor()
        self.reference_detector = ReferenceDetector()
        logger.info("IntentParser initialized")
    
    def parse(self, query: str) -> Intent:
        """Parse query into intent"""
        query_lower = query.lower()
        
        # Extract entities
        entities = self.entity_extractor.extract_all(query)
        
        # Detect references
        references = self.reference_detector.detect_references(query)
        
        # Classify query type
        query_type, sub_type, confidence = self._classify_query(query_lower)
        
        # Extract filters
        filters = self._extract_filters(query_lower, entities)
        
        intent = Intent(
            query_type=query_type,
            sub_type=sub_type,
            confidence=confidence,
            entities=entities,
            references=references,
            filters=filters
        )
        
        logger.debug(f"Parsed intent: {intent.query_type}/{intent.sub_type} (confidence: {confidence})")
        return intent
    
    def _classify_query(self, query: str) -> Tuple[str, str, float]:
        """Classify query type and sub-type"""
        
        # Customer queries
        customer_keywords = ['customer', 'purchase', 'repurchase', 'churn', 'retention', 'active customers', 'average order', 'repeat customer', 'order value']
        if any(kw in query for kw in customer_keywords):
            if 'top' in query or 'best' in query or 'most' in query:
                return 'customer_analysis', 'top_customers', 0.9
            elif 'repeat' in query or 'return' in query:
                return 'customer_analysis', 'repeat_customers', 0.9
            elif 'segment' in query:
                return 'customer_analysis', 'customer_segments', 0.9
            elif 'churn' in query or 'at risk' in query:
                return 'customer_analysis', 'churn_risk', 0.9
            else:
                return 'customer_analysis', 'overview', 0.8
        
        # Product queries
        product_keywords = ['product', 'item', 'goods', 'bestseller', 'potential products', 'top products', 'best selling', 'category', 'brand']
        if any(kw in query for kw in product_keywords):
            if 'top' in query or 'best' in query or 'bestseller' in query:
                return 'product_analysis', 'top_products', 0.9
            elif 'category' in query:
                return 'product_analysis', 'by_category', 0.9
            elif 'brand' in query:
                return 'product_analysis', 'by_brand', 0.9
            else:
                return 'product_analysis', 'overview', 0.8
        
        # Trend queries
        trend_keywords = ['trend', 'over time', 'monthly', 'yearly', 'growth', 'sales trends', 'monthly sales', 'daily sales']
        if any(kw in query for kw in trend_keywords):
            if 'monthly' in query:
                return 'sales_trend', 'monthly', 0.9
            elif 'quarterly' in query or 'quarter' in query:
                return 'sales_trend', 'quarterly', 0.9
            elif 'daily' in query or 'day' in query:
                return 'sales_trend', 'daily', 0.9
            else:
                return 'sales_trend', 'overview', 0.8
        
        # Forecast queries
        forecast_keywords = ['forecast', 'predict', 'future', 'projection', 'plan', 'next 30 days', 'sales forecast']
        if any(kw in query for kw in forecast_keywords):
            return 'sales_forecast', 'revenue_forecast', 0.9
        
        # Revenue/Date queries
        revenue_keywords = ['revenue', 'sales', 'income', 'earnings']
        if any(kw in query for kw in revenue_keywords):
            if re.search(r'(\d{1,2})[./-](\d{4})|(\d{4})', query):
                return 'revenue_report', 'by_date', 0.9
            else:
                return 'revenue_report', 'overview', 0.8
        
        # Default
        return 'general', 'unknown', 0.5
    
    def _extract_filters(self, query: str, entities: Dict) -> Dict[str, Any]:
        """Extract filters from query"""
        filters = {}
        
        # Time filters
        if 'month' in entities:
            filters['month'] = entities['month']
        if 'year' in entities:
            filters['year'] = entities['year']
        
        # Top N filter
        if 'top_n' in entities:
            filters['limit'] = entities['top_n']
        
        # Time range filters
        if 'last' in query:
            filters['time_range'] = 'last'
        elif 'previous' in query:
            filters['time_range'] = 'previous'
        elif 'next' in query:
            filters['time_range'] = 'next'
        
        return filters


class QueryIntentAnalyzer:
    """High-level analyzer combining parsing and context"""
    
    def __init__(self):
        """Initialize analyzer"""
        self.parser = IntentParser()
        logger.info("QueryIntentAnalyzer initialized")
    
    def analyze(self, query: str, context: Optional[Dict] = None) -> Intent:
        """Analyze query with optional context"""
        intent = self.parser.parse(query)
        
        # If query has references and context is provided, try to resolve
        if intent.references and context:
            intent = self._resolve_references(intent, context)
        
        return intent
    
    def _resolve_references(self, intent: Intent, context: Dict) -> Intent:
        """Resolve pronouns using context"""
        # This is a placeholder - actual resolution would use context
        # For example: "they" might refer to "top 10 customers" from previous query
        
        if intent.references:
            logger.debug(f"Detected references to resolve: {intent.references}")
            # In a real implementation, we would:
            # 1. Look at previous query results
            # 2. Identify what the pronoun refers to
            # 3. Add that to filters
        
        return intent
