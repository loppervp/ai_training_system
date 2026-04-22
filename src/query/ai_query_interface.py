"""
AI Query Interface Module
Refactored to follow QA test flow: Intent Detection -> Query Execution -> Context Storage -> Follow-up Resolution
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import json
import re

from .context_manager import (
    ConversationContext, QueryResult, ContextManager, get_context_manager
)
from .intent_parser import QueryIntentAnalyzer, EntityExtractor
from .reference_resolver import ContextAwareQueryResolver
from src.extractors.sales_extractor import SalesExtractor

logger = logging.getLogger(__name__)

# Global session context for storing query results
SESSION_CONTEXT = {}


class AIQueryInterface:
    """
    AI Query Interface with context-aware flow
    
    ARCHITECTURE:
    1. Detect Intent (new_query, follow_up, drill_down)
    2. Handle New Query (generate SQL, execute, store context)
    3. Handle Follow-up (reuse context, no LLM fake data)
    4. Handle Drill-down (find specific transaction by ID)
    5. LLM Usage (only for formatting/explaining, never generating fake data)
    """
    
    def __init__(self, data_path: str = "data/processed", companyfn: Optional[str] = None):
        """
        Initialize AIQueryInterface
        
        Args:
            data_path: Path to processed data
            companyfn: Company code for data isolation (unique per company)
        """
        self.data_path = data_path
        self.companyfn = companyfn or "p11011004464072155"
        self.data_cache = {}
        self.query_history = []
        
        # Initialize context management
        self.context_mgr = get_context_manager()
        self.intent_analyzer = QueryIntentAnalyzer()
        self.reference_resolver = ContextAwareQueryResolver()
        
        logger.info(f"AIQueryInterface initialized for company: {self.companyfn}")
    
    def process_query(self, query: str, conversation_id: Optional[str] = None, context: Optional[Dict] = None) -> Dict:
        """
        Main entry point: Process query with full context awareness
        
        FLOW:
        1. Get or create conversation context
        2. Detect intent (new_query, follow_up, drill_down)
        3. Route to appropriate handler
        4. Store result in context
        5. Return formatted response
        
        Args:
            query: User question
            conversation_id: Conversation ID for context tracking
            context: Additional context (legacy support)
            
        Returns:
            Dict with results
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # Step 1: Get or create conversation context
            conv_id = conversation_id or "default"
            conv_context = self.context_mgr.get_or_create_conversation(conv_id, self.companyfn)
            
            # Step 2: Detect intent
            intent = self.intent_analyzer.analyze(query, conv_context.to_dict())
            logger.info(f"Detected intent: {intent.query_type}/{intent.sub_type}")
            
            # Step 3: Determine query type
            query_intent = self._detect_query_intent(query, conv_context)
            logger.info(f"Query intent: {query_intent}")
            
            # Step 4: Route to appropriate handler
            if query_intent == "drill_down":
                result = self._handle_drill_down(query, conv_context)
            elif query_intent == "follow_up":
                result = self._handle_follow_up(query, conv_context)
            else:  # new_query
                result = self._handle_new_query(query, conv_context, intent)
            
            # Step 5: Store in conversation context
            if result and 'error' not in result:
                query_result = QueryResult(
                    query_id=f"q_{len(conv_context.query_results) + 1}",
                    query_text=query,
                    query_type=intent.query_type,
                    timestamp=datetime.now().isoformat(),
                    data=result.get('data', {}),
                    summary=result.get('summary', ''),
                    insights=result.get('insights', []),
                    entities=intent.entities
                )
                conv_context.add_query_result(query_result)
            
            # Add to message history
            conv_context.add_message("user", query)
            conv_context.add_message("assistant", result.get('summary', ''))
            
            # Save history
            self.query_history.append({
                'query': query,
                'intent': query_intent,
                'query_type': intent.query_type,
                'timestamp': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                'error': str(e),
                'query': query
            }
    
    def _detect_query_intent(self, query: str, context: ConversationContext) -> str:
        """
        STEP 1: Detect Intent
        
        Classify user input into:
        - "new_query": First query or completely new topic
        - "follow_up": References previous data ("those transactions", "that list", "13 transactions", "54 transactions")
        - "drill_down": Contains transaction ID (pattern: DNxxx)
        
        Args:
            query: User query
            context: Conversation context
            
        Returns:
            Intent type: "new_query", "follow_up", or "drill_down"
        """
        query_lower = query.lower()
        
        # Check for drill-down (transaction ID pattern: DNxxx)
        if re.search(r'\bDN\d+\b', query, re.IGNORECASE):
            logger.info("✅ Detected drill-down intent (transaction ID found)")
            return "drill_down"
        
        # Check for follow-up references
        follow_up_keywords = [
            'those transactions', 'that list', 'those', 'that',
            'them', 'they', 'these', 'this',
            'đó', 'cái đó', 'những cái đó',  # Vietnamese
            'transactions', 'orders', 'items',
            'gồm những cái nào', 'gồm những gì'  # Vietnamese: "what are they"
        ]
        
        # Check if query contains numbers that might refer to previous count
        # This matches patterns like "54 transactions", "13 giao dịch", etc.
        number_match = re.search(r'(\d+)\s+(transactions|orders|items|giao dịch|đơn)', query_lower)
        
        if context.query_results and (
            any(kw in query_lower for kw in follow_up_keywords) or number_match
        ):
            logger.info("✅ Detected follow-up intent (references to previous data)")
            return "follow_up"
        
        logger.info("✅ Detected new_query intent")
        return "new_query"
    
    def _handle_new_query(self, query: str, context: ConversationContext, intent: Any) -> Dict:
        """
        STEP 2: Handle New Query
        
        Flow:
        1. Generate SQL based on intent
        2. Execute query against database
        3. Get real data
        4. Store context for follow-ups
        5. Return formatted response
        
        Args:
            query: User query
            context: Conversation context
            intent: Parsed intent
            
        Returns:
            Dict with results
        """
        try:
            logger.info("🔍 Handling NEW QUERY")
            
            # Determine query type and generate SQL
            query_lower = query.lower()
            
            # ✅ Best selling products
            if 'best selling' in query_lower or 'bestseller' in query_lower or 'top product' in query_lower:
                return self._handle_best_selling_products(query_lower, context)
            
            # 📉 Customer churn rate
            if 'churn' in query_lower or 'churn rate' in query_lower:
                return self._handle_churn_rate(query_lower, context)
            
            # 📈 Monthly sales trends
            if 'monthly sales' in query_lower or 'sales trend' in query_lower:
                return self._handle_monthly_sales_trends(query_lower, context)
            
            # 🔮 Sales forecast next 30 days
            if 'forecast' in query_lower or 'predict' in query_lower or 'next 30 days' in query_lower:
                return self._handle_sales_forecast(query_lower, context)
            
            # 👥 Active customers per month
            if 'active customers' in query_lower:
                return self._handle_active_customers(query_lower, context)
            
            # 💰 Average order value
            if 'average order' in query_lower or 'aov' in query_lower or 'order value' in query_lower:
                return self._handle_average_order_value(query_lower, context)
            
            # 🔄 Repeat customer rate
            if 'repeat customer' in query_lower or 'returning customer' in query_lower:
                return self._handle_repeat_customer_rate(query_lower, context)
            
            # Check for revenue/date queries
            if self._is_revenue_query(query_lower):
                return self._handle_date_revenue_query(query_lower, context)
            
            # Check for customer queries
            if self._is_customer_query(query_lower):
                return self._handle_customer_query(query_lower, context)
            
            # Check for product queries
            if self._is_product_query(query_lower):
                return self._handle_product_query(query_lower, context)
            
            # Check for sales trend queries
            if self._is_trend_query(query_lower):
                return self._handle_sales_trend_query(query_lower, context)
            
            # Check for top sales orders
            if 'top' in query_lower and ('sales' in query_lower or 'orders' in query_lower):
                return self._handle_top_sales_query(query_lower, context)
            
            # Default: general query
            return self._handle_general_query(query_lower, context)
            
        except Exception as e:
            logger.error(f"Error handling new query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_follow_up(self, query: str, context: ConversationContext) -> Dict:
        """
        STEP 3: Handle Follow-up
        
        CRITICAL FIX:
        - Check SESSION_CONTEXT for stored transaction data
        - Return ALL transactions (not just top 5)
        - Support patterns like "those transactions", "13 transactions", "that list"
        
        If user says:
        - "those transactions"
        - "that list"
        - "13 transactions"
        - "đó"
        
        Then:
        1. Get context from SESSION_CONTEXT (stored by previous query)
        2. Return stored data (NO LLM fake data)
        3. Format response with ALL transactions
        
        Args:
            query: User query
            context: Conversation context
            
        Returns:
            Dict with results
        """
        try:
            logger.info("🔄 Handling FOLLOW-UP")
            
            # ============================================================================
            # STEP 1: Check SESSION_CONTEXT for stored transaction data
            # ============================================================================
            session_data = SESSION_CONTEXT.get(self.companyfn)
            
            print(f"SESSION_CONTEXT[{self.companyfn}]: {session_data}")
            
            if not session_data or not session_data.get('data'):
                logger.warning("❌ No transaction data in SESSION_CONTEXT")
                return {
                    'error': 'No transaction data available',
                    'summary': 'I need to know what you are referring to. Please ask a new question.',
                    'data': {}
                }
            
            logger.info(f"✅ Found {session_data['count']} transactions in context")
            
            # ============================================================================
            # STEP 2: Extract transaction data from context
            # ============================================================================
            all_transactions = session_data.get('data', [])
            transaction_count = session_data.get('count', len(all_transactions))
            period = session_data.get('period', 'previous query')
            total_revenue = session_data.get('total_revenue', 0)
            
            # ============================================================================
            # STEP 3: Format response with ALL transactions
            # ============================================================================
            query_lower = query.lower()
            
            # Build transaction list
            transaction_lines = []
            for i, trans in enumerate(all_transactions, 1):
                dnum = trans.get('dnum_auto', trans.get('order_id', 'N/A'))
                amount = trans.get('amount', 0)
                customer = trans.get('customer_name', 'N/A')
                transaction_lines.append(f"{i}. {dnum}: ${amount:,.2f} ({customer})")
            
            transaction_text = "\n".join(transaction_lines)
            
            # Build summary
            summary = f"Here are the {transaction_count} transactions from {period}:\n\n{transaction_text}"
            
            return {
                'query_type': 'transaction_list',
                'summary': summary,
                'data': {
                    'period': period,
                    'total_revenue': total_revenue,
                    'transaction_count': transaction_count,
                    'transactions': all_transactions
                },
                'insights': [
                    f"Total: {transaction_count} transactions",
                    f"Total Revenue: ${total_revenue:,.2f}",
                    f"Average per transaction: ${total_revenue/transaction_count:,.2f}" if transaction_count > 0 else ""
                ],
                'is_follow_up': True
            }
            
        except Exception as e:
            logger.error(f"Error handling follow-up: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _handle_drill_down(self, query: str, context: ConversationContext) -> Dict:
        """
        STEP 4: Handle Drill-down
        
        If user input contains transaction ID (pattern: DNxxx):
        
        SELECT * FROM scm_sal_main WHERE dnum_auto = 'DN001'
        
        Args:
            query: User query
            context: Conversation context
            
        Returns:
            Dict with results
        """
        try:
            logger.info("🔎 Handling DRILL-DOWN")
            
            # Extract transaction ID
            match = re.search(r'\b(DN\d+)\b', query, re.IGNORECASE)
            if not match:
                return {'error': 'No transaction ID found'}
            
            transaction_id = match.group(1).upper()
            logger.info(f"✅ Found transaction ID: {transaction_id}")
            
            # Query database for transaction details
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                # Query scm_sal_main for transaction
                query_sql = f"""
                    SELECT * FROM scm_sal_main 
                    WHERE dnum_auto = :dnum_auto 
                    AND companyfn = :companyfn
                """
                
                df = extractor.db_extractor.extract_data(
                    query_sql,
                    {'dnum_auto': transaction_id, 'companyfn': self.companyfn}
                )
                
                if df.empty:
                    return {
                        'error': f'Transaction {transaction_id} not found',
                        'summary': f'No data found for transaction {transaction_id}'
                    }
                
                # Get transaction details
                trans = df.iloc[0]
                
                return {
                    'query_type': 'transaction_detail',
                    'summary': f'Details for transaction {transaction_id}',
                    'data': {
                        'transaction_id': transaction_id,
                        'customer': trans.get('party_desc', 'N/A'),
                        'amount': float(trans.get('amount_local', 0)),
                        'date': str(trans.get('date_trans', 'N/A')),
                        'status': 'Completed',
                        'raw_data': df.to_dict('records')[0]
                    },
                    'insights': [
                        f"Transaction {transaction_id} from {trans.get('party_desc', 'Unknown')}",
                        f"Amount: ${float(trans.get('amount_local', 0)):,.2f}"
                    ]
                }
            
        except Exception as e:
            logger.error(f"Error handling drill-down: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_previous_data(self, query: str, last_result: QueryResult, context: ConversationContext) -> Dict:
        """
        Analyze previous data based on follow-up question
        
        Args:
            query: Follow-up query
            last_result: Previous query result
            context: Conversation context
            
        Returns:
            Dict with analysis
        """
        try:
            data = last_result.data
            
            # Check what user is asking about
            if 'highest' in query or 'most' in query or 'top' in query:
                # Find highest value
                if 'transactions' in data or 'orders' in data:
                    items = data.get('transactions') or data.get('orders', [])
                    if items:
                        top_item = items[0]
                        return {
                            'query_type': last_result.query_type,
                            'summary': f"The highest value item is {top_item.get('order_id', 'N/A')}",
                            'data': {'top_item': top_item},
                            'insights': [f"Amount: ${top_item.get('amount', 0):,.2f}"]
                        }
            
            # Default: return data
            return {
                'query_type': last_result.query_type,
                'summary': f"Analysis of {last_result.query_type}",
                'data': data,
                'insights': last_result.insights
            }
            
        except Exception as e:
            logger.error(f"Error analyzing previous data: {str(e)}")
            return {'error': str(e)}
    
    # ============================================================================
    # QUERY TYPE DETECTION HELPERS
    # ============================================================================
    
    def _is_revenue_query(self, query: str) -> bool:
        """Check if query is about revenue/date"""
        revenue_keywords = ['revenue', 'sales', 'income', 'earnings', 'report']
        date_pattern = r'(\d{1,2})[./-](\d{4})|(\d{4})'
        return (any(kw in query for kw in revenue_keywords) or 
                re.search(date_pattern, query) is not None)
    
    def _is_customer_query(self, query: str) -> bool:
        """Check if query is about customers"""
        keywords = ['customer', 'purchase', 'repurchase', 'churn', 'retention', 'active customers']
        return any(kw in query for kw in keywords)
    
    def _is_product_query(self, query: str) -> bool:
        """Check if query is about products"""
        keywords = ['product', 'item', 'goods', 'bestseller', 'category', 'brand']
        return any(kw in query for kw in keywords)
    
    def _is_trend_query(self, query: str) -> bool:
        """Check if query is about trends"""
        keywords = ['trend', 'over time', 'monthly', 'yearly', 'growth', 'daily']
        return any(kw in query for kw in keywords)
    
    # ============================================================================
    # QUERY HANDLERS
    # ============================================================================
    
    def _handle_top_sales_query(self, query: str, context: ConversationContext) -> Dict:
        """Handle top sales orders query"""
        try:
            limit = 10
            match = re.search(r'top\s+(\d+)', query, re.IGNORECASE)
            if match:
                limit = int(match.group(1))
            
            logger.info(f"Fetching top {limit} sales orders")
            
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                df = extractor.extract_sales_main(companyfn=self.companyfn)
                
                if df.empty:
                    return {
                        'query_type': 'sales_analysis',
                        'summary': 'No sales data found',
                        'data': {},
                        'insights': []
                    }
                
                # Get top orders by amount
                top_orders = df.nlargest(limit, 'amount_local')[
                    ['uniquenum_pri', 'amount_local', 'party_desc', 'date_trans']
                ].to_dict('records')
                
                # Rename columns for consistency
                for order in top_orders:
                    order['order_id'] = order.pop('uniquenum_pri')
                    order['amount'] = order.pop('amount_local')
                    order['customer_name'] = order.pop('party_desc')
                    order['date'] = order.pop('date_trans')
                
                # Store in session context for follow-ups
                SESSION_CONTEXT[self.companyfn] = {
                    'data': top_orders,
                    'ids': [o['order_id'] for o in top_orders],
                    'count': len(top_orders),
                    'type': 'sales_orders'
                }
                
                return {
                    'query_type': 'sales_analysis',
                    'summary': f'Top {limit} sales orders',
                    'data': {'orders': top_orders},
                    'insights': [
                        f"Found {len(top_orders)} orders",
                        f"Highest order: ${top_orders[0]['amount']:,.2f}" if top_orders else "No data"
                    ]
                }
            
        except Exception as e:
            logger.error(f"Error handling top sales query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_date_revenue_query(self, query: str, context: ConversationContext) -> Dict:
        """
        Handle revenue by date query
        
        CRITICAL FIX:
        - Generate TWO queries: SUMMARY + DETAIL
        - SUMMARY: SUM(amount_local), COUNT(DISTINCT dnum_auto)
        - DETAIL: GROUP BY dnum_auto (NOT month)
        - Store ALL transaction details in context
        """
        try:
            # Extract month and year
            month_map = {
                'jan': 1, 'january': 1, 'feb': 2, 'february': 2,
                'mar': 3, 'march': 3, 'apr': 4, 'april': 4,
                'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
                'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
                'oct': 10, 'october': 10, 'nov': 11, 'november': 11,
                'dec': 12, 'december': 12
            }
            
            # Try numeric format first
            match = re.search(r'(\d{1,2})[./-](\d{4})', query)
            if match:
                month = int(match.group(1))
                year = int(match.group(2))
            else:
                # Try text month
                month_match = re.search(
                    r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(\d{4})',
                    query, re.IGNORECASE
                )
                if month_match:
                    month = month_map[month_match.group(1).lower()]
                    year = int(month_match.group(2))
                else:
                    return {'error': 'No valid date format found'}
            
            logger.info(f"🔍 Fetching revenue for {month}/{year}")
            
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                # Get date range
                date_from = f"{year}-{month:02d}-01"
                if month == 12:
                    date_to = f"{year}-12-31"
                else:
                    date_to = f"{year}-{month+1:02d}-01"
                
                # ============================================================================
                # QUERY 1: SUMMARY (SUM + COUNT DISTINCT dnum_auto)
                # ============================================================================
                logger.info(f"📊 Executing SUMMARY query for {month}/{year}")
                
                summary_query = f"""
                    SELECT 
                        SUM(amount_local) AS total_revenue,
                        COUNT(DISTINCT dnum_auto) AS total_transactions
                    FROM scm_sal_main
                    WHERE companyfn = :companyfn
                        AND date_trans >= :date_from
                        AND date_trans < :date_to
                        AND tag_void_yn = 'n'
                """
                
                summary_params = {
                    'companyfn': self.companyfn,
                    'date_from': date_from,
                    'date_to': date_to
                }
                
                summary_df = extractor.db_extractor.extract_data(summary_query, summary_params)
                print(f"SQL SUMMARY: {summary_query}")
                print(f"SUMMARY RESULT: {summary_df.to_dict('records') if not summary_df.empty else 'EMPTY'}")
                
                if summary_df.empty:
                    return {
                        'query_type': 'revenue_report',
                        'summary': f'No data for {month}/{year}',
                        'data': {'period': f"{month}/{year}", 'total_revenue': 0, 'transaction_count': 0},
                        'insights': []
                    }
                
                total_revenue = float(summary_df.iloc[0]['total_revenue'] or 0)
                transaction_count = int(summary_df.iloc[0]['total_transactions'] or 0)
                
                # ============================================================================
                # QUERY 2: DETAIL (GROUP BY dnum_auto - NOT month)
                # ============================================================================
                logger.info(f"📋 Executing DETAIL query for {month}/{year}")
                
                detail_query = f"""
                    SELECT 
                        dnum_auto,
                        party_desc,
                        SUM(amount_local) AS amt_local,
                        MAX(date_trans) AS date_trans
                    FROM scm_sal_main
                    WHERE companyfn = :companyfn
                        AND date_trans >= :date_from
                        AND date_trans < :date_to
                        AND tag_void_yn = 'n'
                    GROUP BY dnum_auto, party_desc
                    ORDER BY amt_local DESC
                """
                
                detail_params = {
                    'companyfn': self.companyfn,
                    'date_from': date_from,
                    'date_to': date_to
                }
                
                detail_df = extractor.db_extractor.extract_data(detail_query, detail_params)
                print(f"SQL DETAIL: {detail_query}")
                print(f"DETAIL RESULT: {detail_df.to_dict('records') if not detail_df.empty else 'EMPTY'}")
                
                # Convert detail data to transaction list
                all_transactions = []
                if not detail_df.empty:
                    for _, row in detail_df.iterrows():
                        all_transactions.append({
                            'dnum_auto': row['dnum_auto'],
                            'customer_name': row['party_desc'],
                            'amount': float(row['amt_local']),
                            'date': str(row['date_trans'])
                        })
                
                # ============================================================================
                # STORE IN CONTEXT FOR FOLLOW-UPS
                # ============================================================================
                SESSION_CONTEXT[self.companyfn] = {
                    'data': all_transactions,
                    'ids': [t['dnum_auto'] for t in all_transactions],
                    'count': transaction_count,
                    'type': 'transactions',
                    'period': f"{month}/{year}",
                    'total_revenue': total_revenue
                }
                
                print(f"CONTEXT: {SESSION_CONTEXT.get(self.companyfn)}")
                
                # Return top 5 for display, but store all for follow-ups
                top_transactions = all_transactions[:5]
                
                return {
                    'query_type': 'revenue_report',
                    'summary': f'Revenue for {month}/{year}: ${total_revenue:,.2f}',
                    'data': {
                        'period': f"{month}/{year}",
                        'total_revenue': total_revenue,
                        'transaction_count': transaction_count,
                        'transactions': top_transactions,
                        'all_transactions': all_transactions  # Store all for follow-ups
                    },
                    'insights': [
                        f"{transaction_count} transactions in {month}/{year}",
                        f"Average per transaction: ${total_revenue/transaction_count:,.2f}" if transaction_count > 0 else ""
                    ]
                }
            
        except Exception as e:
            logger.error(f"Error handling date revenue query: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def _handle_customer_query(self, query: str, context: ConversationContext) -> Dict:
        """Handle customer-related query"""
        try:
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                sql = """
                    SELECT 
                        party_code AS customer_code,
                        party_desc AS customer_name,
                        COUNT(DISTINCT dnum_auto) AS total_purchases,
                        SUM(amount_local) AS total_revenue,
                        MAX(date_trans) AS last_purchase_date,
                        MIN(date_trans) AS first_purchase_date
                    FROM scm_sal_main
                    WHERE companyfn = :companyfn
                    AND tag_void_yn = 'n'
                    GROUP BY party_code, party_desc
                    ORDER BY total_purchases DESC
                    LIMIT 15
                """
                df = extractor.db_extractor.extract_data(sql, {'companyfn': self.companyfn})
                
                if df.empty:
                    return {
                        'query_type': 'top_customers',
                        'summary': 'Không tìm thấy dữ liệu khách hàng',
                        'data': {},
                        'insights': []
                    }
                
                customers = df.to_dict('records')
                
                return {
                    'query_type': 'top_customers',
                    'summary': f'Top {len(customers)} khách hàng theo số lần mua hàng',
                    'data': {
                        'top_customers': customers,
                        'total_customers': len(customers)
                    },
                    'insights': [
                        f"Khách hàng mua nhiều nhất: {customers[0]['customer_name']}",
                        f"Số lần mua: {customers[0]['total_purchases']} lần",
                        f"Tổng doanh thu: ${customers[0]['total_revenue']:,.0f}"
                    ]
                }
            
        except Exception as e:
            logger.error(f"Error handling customer query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_product_query(self, query: str, context: ConversationContext) -> Dict:
        """Handle product-related query"""
        try:
            df = self.load_data('product_analysis')
            
            result = {
                'query_type': 'product_analysis',
                'summary': 'Product analysis',
                'data': {},
                'insights': []
            }
            
            if df.empty:
                result['summary'] = 'No product data available'
                return result
            
            # Bestselling products
            if 'bestseller' in query or 'top' in query:
                revenue_col = next((col for col in ['line_amount', 'total_revenue', 'revenue'] if col in df.columns), None)
                if revenue_col:
                    top_products = df.groupby('product_name')[revenue_col].sum().sort_values(ascending=False).head(10)
                    result['data']['top_products'] = top_products.to_dict()
                    result['summary'] = 'Top 10 bestselling products'
                    if not top_products.empty:
                        result['insights'].append(f"Bestselling: {top_products.index[0]}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling product query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_sales_trend_query(self, query: str, context: ConversationContext) -> Dict:
        """Handle sales trend query"""
        try:
            df = self.load_data('sales_trend')
            
            result = {
                'query_type': 'sales_trend',
                'summary': 'Sales trend analysis',
                'data': {},
                'insights': []
            }
            
            if df.empty:
                result['summary'] = 'No trend data available'
                return result
            
            # Monthly trends
            if 'monthly' in query:
                monthly_sales = df.groupby('month')['total_revenue'].sum()
                result['data']['monthly_sales'] = monthly_sales.to_dict()
                result['summary'] = 'Monthly sales trends'
                if not monthly_sales.empty:
                    best_month = monthly_sales.idxmax()
                    result['insights'].append(f"Best month: Month {best_month}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error handling sales trend query: {str(e)}")
            return {'error': str(e)}
    
    def _handle_best_selling_products(self, query: str, context: ConversationContext) -> Dict:
        """✅ Best selling products handler"""
        try:
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                sql = """
                    SELECT 
                        stkcode_code AS stock_code,
                        stkcode_desc AS product_name,
                        SUM(amount_local) AS total_revenue,
                        SUM(qnty_total) AS total_quantity,
                        COUNT(DISTINCT uniquenum_pri) AS order_count
                    FROM scm_sal_data
                    WHERE companyfn = :companyfn
                    AND tag_void_yn = 'n'
                    GROUP BY stkcode_code, stkcode_desc
                    ORDER BY total_revenue DESC
                    LIMIT 15
                """
                df = extractor.db_extractor.extract_data(sql, {'companyfn': self.companyfn})
                
                if df.empty:
                    return {
                        'query_type': 'best_selling_products',
                        'summary': 'Không tìm thấy dữ liệu sản phẩm bán chạy',
                        'data': {},
                        'insights': []
                    }
                
                products = df.to_dict('records')
                
                return {
                    'query_type': 'best_selling_products',
                    'summary': f'Top {len(products)} sản phẩm bán chạy nhất',
                    'data': {
                        'best_selling_products': products,
                        'total_products': len(products)
                    },
                    'insights': [
                        f"Sản phẩm bán chạy nhất: {products[0]['product_name']}",
                        f"Doanh thu cao nhất: ${products[0]['total_revenue']:,.0f}",
                        f"Số lượng bán: {products[0]['total_quantity']:,} đơn vị"
                    ]
                }
        except Exception as e:
            logger.error(f"Best selling products error: {str(e)}")
            return {'error': str(e)}

    def _handle_churn_rate(self, query: str, context: ConversationContext) -> Dict:
        """📉 Customer churn rate handler"""
        try:
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                sql = """
                    WITH customer_activity AS (
                        SELECT 
                            party_code,
                            MAX(date_trans) AS last_purchase_date
                        FROM scm_sal_main
                        WHERE companyfn = :companyfn
                        AND tag_void_yn = 'n'
                        GROUP BY party_code
                    )
                    SELECT 
                        COUNT(*) AS total_customers,
                        SUM(CASE WHEN last_purchase_date < CURRENT_DATE - INTERVAL '90 days' THEN 1 ELSE 0 END) AS churned_customers
                    FROM customer_activity
                """
                df = extractor.db_extractor.extract_data(sql, {'companyfn': self.companyfn})
                
                if df.empty:
                    return {
                        'query_type': 'churn_rate',
                        'summary': 'Không tìm thấy dữ liệu churn rate',
                        'data': {},
                        'insights': []
                    }
                
                total = df.iloc[0]['total_customers']
                churned = df.iloc[0]['churned_customers']
                churn_rate = (churned / total) * 100 if total > 0 else 0
                
                return {
                    'query_type': 'churn_rate',
                    'summary': f'Tỷ lệ khách hàng rời bỏ: {churn_rate:.1f}%',
                    'data': {
                        'total_customers': total,
                        'churned_customers': churned,
                        'churn_rate_percent': round(churn_rate, 1)
                    },
                    'insights': [
                        f"Tổng khách hàng: {total:,}",
                        f"Khách hàng rời bỏ (không mua >90 ngày): {churned:,}",
                        f"Churn Rate: {churn_rate:.1f}%"
                    ]
                }
        except Exception as e:
            logger.error(f"Churn rate error: {str(e)}")
            return {'error': str(e)}

    def _handle_monthly_sales_trends(self, query: str, context: ConversationContext) -> Dict:
        """📈 Monthly sales trends handler"""
        try:
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                sql = """
                    SELECT 
                        TO_CHAR(date_trans, 'YYYY-MM') AS month,
                        SUM(amount_local) AS total_revenue,
                        COUNT(DISTINCT dnum_auto) AS total_orders,
                        COUNT(DISTINCT party_code) AS unique_customers
                    FROM scm_sal_main
                    WHERE companyfn = :companyfn
                    AND tag_void_yn = 'n'
                    AND date_trans >= CURRENT_DATE - INTERVAL '12 months'
                    GROUP BY TO_CHAR(date_trans, 'YYYY-MM')
                    ORDER BY month ASC
                """
                df = extractor.db_extractor.extract_data(sql, {'companyfn': self.companyfn})
                
                if df.empty:
                    return {
                        'query_type': 'monthly_sales_trends',
                        'summary': 'Không tìm thấy dữ liệu xu hướng doanh thu',
                        'data': {},
                        'insights': []
                    }
                
                trends = df.to_dict('records')
                
                return {
                    'query_type': 'monthly_sales_trends',
                    'summary': 'Xu hướng doanh thu theo tháng trong 12 tháng qua',
                    'data': {
                        'monthly_trends': trends,
                        'months_count': len(trends)
                    },
                    'insights': [
                        f"Dữ liệu từ {trends[0]['month']} đến {trends[-1]['month']}",
                        f"Doanh thu tháng gần nhất: ${trends[-1]['total_revenue']:,.0f}"
                    ]
                }
        except Exception as e:
            logger.error(f"Monthly sales trends error: {str(e)}")
            return {'error': str(e)}

    def _handle_sales_forecast(self, query: str, context: ConversationContext) -> Dict:
        """🔮 Sales forecast next 30 days handler"""
        try:
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                # Lấy dữ liệu 90 ngày qua để dự báo
                sql = """
                    SELECT 
                        DATE(date_trans) AS date,
                        SUM(amount_local) AS daily_revenue
                    FROM scm_sal_main
                    WHERE companyfn = :companyfn
                    AND tag_void_yn = 'n'
                    AND date_trans >= CURRENT_DATE - INTERVAL '90 days'
                    GROUP BY DATE(date_trans)
                    ORDER BY date ASC
                """
                df = extractor.db_extractor.extract_data(sql, {'companyfn': self.companyfn})
                
                if df.empty:
                    return {
                        'query_type': 'sales_forecast',
                        'summary': 'Không đủ dữ liệu để dự báo',
                        'data': {},
                        'insights': []
                    }
                
                avg_daily = df['daily_revenue'].mean()
                forecast_30d = avg_daily * 30
                growth_rate = 0.05  # Giả định tăng trưởng 5%
                
                return {
                    'query_type': 'sales_forecast',
                    'summary': f'Dự báo doanh thu 30 ngày tới: ${forecast_30d:,.0f}',
                    'data': {
                        'forecast_30_days': round(forecast_30d, 0),
                        'average_daily_revenue': round(avg_daily, 0),
                        'confidence_level': 'Medium',
                        'growth_assumption': '5%'
                    },
                    'insights': [
                        f"Doanh thu trung bình hàng ngày: ${avg_daily:,.0f}",
                        f"Dự báo 30 ngày tới: ${forecast_30d:,.0f}",
                        "Dự báo dựa trên dữ liệu 90 ngày lịch sử"
                    ]
                }
        except Exception as e:
            logger.error(f"Sales forecast error: {str(e)}")
            return {'error': str(e)}

    def _handle_active_customers(self, query: str, context: ConversationContext) -> Dict:
        """👥 Active customers per month handler"""
        try:
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                sql = """
                    SELECT 
                        DATE_FORMAT(date_trans, '%Y-%m') AS month,
                        COUNT(DISTINCT party_code) AS active_customers
                    FROM scm_sal_main
                    WHERE companyfn = :companyfn
                    AND tag_void_yn = 'n'
                    AND date_trans >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
                    GROUP BY DATE_FORMAT(date_trans, '%Y-%m')
                    ORDER BY month ASC
                """
                df = extractor.db_extractor.extract_data(sql, {'companyfn': self.companyfn})
                
                if df.empty:
                    return {
                        'query_type': 'active_customers',
                        'summary': 'Không tìm thấy dữ liệu khách hàng hoạt động',
                        'data': {},
                        'insights': []
                    }
                
                customers_monthly = df.to_dict('records')
                current_month = customers_monthly[-1]
                
                return {
                    'query_type': 'active_customers',
                    'summary': f'Số khách hàng hoạt động tháng {current_month["month"]}: {current_month["active_customers"]:,}',
                    'data': {
                        'active_customers_monthly': customers_monthly,
                        'current_month_customers': current_month['active_customers']
                    },
                    'insights': [
                        f"Khách hàng hoạt động tháng này: {current_month['active_customers']:,}",
                        f"Dữ liệu trong {len(customers_monthly)} tháng qua"
                    ]
                }
        except Exception as e:
            logger.error(f"Active customers error: {str(e)}")
            return {'error': str(e)}

    def _handle_average_order_value(self, query: str, context: ConversationContext) -> Dict:
        """💰 Average order value handler"""
        try:
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                sql = """
                    SELECT 
                        COUNT(dnum_auto) AS total_orders,
                        SUM(amount_local) AS total_revenue,
                        AVG(amount_local) AS average_order_value
                    FROM scm_sal_main
                    WHERE companyfn = :companyfn
                    AND tag_void_yn = 'n'
                    AND date_trans >= CURRENT_DATE - INTERVAL '90 days'
                """
                df = extractor.db_extractor.extract_data(sql, {'companyfn': self.companyfn})
                
                if df.empty:
                    return {
                        'query_type': 'average_order_value',
                        'summary': 'Không tìm thấy dữ liệu giá trị đơn hàng',
                        'data': {},
                        'insights': []
                    }
                
                aov = df.iloc[0]['average_order_value']
                total_orders = df.iloc[0]['total_orders']
                
                return {
                    'query_type': 'average_order_value',
                    'summary': f'Giá trị đơn hàng trung bình: ${aov:,.2f}',
                    'data': {
                        'average_order_value': round(aov, 2),
                        'total_orders_90d': total_orders,
                        'period': '90 ngày gần nhất'
                    },
                    'insights': [
                        f"AOV trung bình 90 ngày: ${aov:,.2f}",
                        f"Tổng đơn hàng trong kỳ: {total_orders:,}"
                    ]
                }
        except Exception as e:
            logger.error(f"Average order value error: {str(e)}")
            return {'error': str(e)}

    def _handle_repeat_customer_rate(self, query: str, context: ConversationContext) -> Dict:
        """🔄 Repeat customer rate handler"""
        try:
            with SalesExtractor(companyfn=self.companyfn) as extractor:
                sql = """
                    WITH customer_orders AS (
                        SELECT 
                            party_code,
                            COUNT(DISTINCT dnum_auto) AS order_count
                        FROM scm_sal_main
                        WHERE companyfn = :companyfn
                        AND tag_void_yn = 'n'
                        GROUP BY party_code
                    )
                    SELECT 
                        COUNT(*) AS total_customers,
                        SUM(CASE WHEN order_count >= 2 THEN 1 ELSE 0 END) AS repeat_customers
                    FROM customer_orders
                """
                df = extractor.db_extractor.extract_data(sql, {'companyfn': self.companyfn})
                
                if df.empty:
                    return {
                        'query_type': 'repeat_customer_rate',
                        'summary': 'Không tìm thấy dữ liệu khách hàng quay lại',
                        'data': {},
                        'insights': []
                    }
                
                total = df.iloc[0]['total_customers']
                repeat = df.iloc[0]['repeat_customers']
                repeat_rate = (repeat / total) * 100 if total > 0 else 0
                
                return {
                    'query_type': 'repeat_customer_rate',
                    'summary': f'Tỷ lệ khách hàng quay lại: {repeat_rate:.1f}%',
                    'data': {
                        'total_customers': total,
                        'repeat_customers': repeat,
                        'repeat_rate_percent': round(repeat_rate, 1)
                    },
                    'insights': [
                        f"Tổng khách hàng: {total:,}",
                        f"Khách hàng mua >=2 lần: {repeat:,}",
                        f"Tỷ lệ quay lại: {repeat_rate:.1f}%"
                    ]
                }
        except Exception as e:
            logger.error(f"Repeat customer rate error: {str(e)}")
            return {'error': str(e)}

    def _handle_general_query(self, query: str, context: ConversationContext) -> Dict:
        """Handle general query"""
        return {
            'query_type': 'general',
            'summary': 'Please ask specifically about customers, products, sales, or revenue',
            'data': {},
            'insights': []
        }
    
    # ============================================================================
    # DATA LOADING
    # ============================================================================
    
    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """
        Read data from cache or file, filtered by companyfn for data isolation
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            DataFrame filtered by companyfn
        """
        try:
            cache_key = f"{dataset_name}_{self.companyfn}" if self.companyfn else dataset_name
            if cache_key in self.data_cache:
                return self.data_cache[cache_key]
            
            import os
            from pathlib import Path
            
            data_dir = Path(self.data_path)
            for ext in ['parquet', 'csv', 'json']:
                file_path = data_dir / f"{dataset_name}.{ext}"
                if file_path.exists():
                    if ext == 'parquet':
                        df = pd.read_parquet(file_path)
                    elif ext == 'csv':
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_json(file_path)
                    
                    if self.companyfn and 'companyfn' in df.columns:
                        df = df[df['companyfn'] == self.companyfn]
                        logger.info(f"Filtered {dataset_name} by companyfn: {len(df)} records")
                    
                    self.data_cache[cache_key] = df
                    return df
            
            logger.warning(f"Dataset not found: {dataset_name}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error reading data: {str(e)}")
            return pd.DataFrame()
    
    def get_query_history(self) -> List[Dict]:
        """Get query history"""
        return self.query_history
    
    def format_response(self, result: Dict, query_text: str = "") -> str:
        """
        Format result into readable text with auto interactive chart generation
        
        Args:
            result: Result Dict
            query_text: Original user query text
            
        Returns:
            String response
        """
        try:
            # Auto generate interactive Plotly HTML chart
            try:
                from .chart_formatter import ChartFormatter
                chart_formatter = ChartFormatter()
                result = chart_formatter.format_response_with_chart(result, query_text)
            except Exception as e:
                logger.warning(f"Chart generation skipped: {e}")
            
            response = []
            
            if 'error' in result:
                return f"Error: {result['error']}"
            
            response.append(f"[Report] {result.get('summary', 'Query result')}")
            response.append("")
            
            # Insights
            if result.get('insights'):
                response.append("* Insights:")
                for insight in result['insights']:
                    if insight:
                        response.append(f"  • {insight}")
                response.append("")
            
            # Chart info if generated
            if 'chart_path' in result:
                response.append("📊 Interactive Chart Generated:")
                response.append(f"  ✅ File: {result['chart_path']}")
                response.append(f"  💡 Open in browser, hover on points for details")
                response.append("")
            
            # Data summary
            if result.get('data'):
                response.append("* Data:")
                for key, value in result['data'].items():
                    if isinstance(value, (int, float)):
                        if value > 1000000:
                            response.append(f"  • {key}: {value:,.0f}")
                        elif value > 100:
                            response.append(f"  • {key}: {value:,.2f}")
                        else:
                            response.append(f"  • {key}: {value}")
                    else:
                        response.append(f"  • {key}: {value}")
            
            return "\n".join(response)
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return f"Format error: {str(e)}"
