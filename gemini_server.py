#!/usr/bin/env python
"""
AI Chat Server - Backend for AI Chat Interface (OpenRouter Version)
Integrated with AI Training System
"""

import http.server
import socketserver
import json
import os
import sys
from openai import OpenAI
from anthropic import Anthropic
from datetime import datetime
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add root directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import context management modules
from src.query.context_manager import get_context_manager, QueryResult
from src.query.intent_parser import QueryIntentAnalyzer
from src.query.reference_resolver import ContextAwareQueryResolver

PORT = 9001

# ✅ MULTI AI PROVIDER CONFIGURATION WITH AUTOMATIC FALLBACK
# Khi 1 cái hết quota/error, hệ thống sẽ tự động chuyển sang cái kế tiếp

AI_PROVIDERS = [
     {
        "name": "Ollama (cloud)",
        "base_url": "https://api.ollama.com/api/generate",  # ✅ Dùng /api/generate
        "api_key": os.getenv('OLLAMA_API_KEY', '2d257bf6c51b4aada48275680752b585.PYmrtueCOkFE6KPAjLk-6fVw'),
        "model": "qwen3.5:cloud",
        "enabled": True,
        "working": True,
        "local": False
    },
    # 0. Claude (Anthropic) - ưu tiên 1 (chạy đầu tiên)
    {
        "name": "Claude (Anthropic)",
        "api_key": os.getenv('ANTHROPIC_API_KEY', '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'),  # Lấy từ biến môi trường
        "model": "claude-haiku-4-5-20251001",
        "enabled": True,  # Bật nó lên!
        "working": True,
        "type": "anthropic"
    },

    # 1. OpenRouter (Mặc định - ưu tiên 2)
    {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.getenv('OPENROUTER_API_KEY', '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'),
        "model": "openrouter/free",
        "enabled": True,
        "working": True,
        "type": "openai"
    },

    # 2. Google Gemini Official (MIỄN PHÍ 1M token/tháng - ưu tiên 2)
    {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": os.getenv('GEMINI_API_KEY', 'AIzaSyAV-jy6RTh1veM5paNZzkuX4Oto9sQBBh8'),
        "model": "gemini-2.0-flash",
        "enabled": True,
        "working": True
    }


    # 4. 🔥 Ollama - RUN LOCAL OFFLINE 100% MIỄN PHÍ (tải model về rồi mới bật)
   
]

# Trạng thái theo dõi provider nào đang hoạt động
current_provider_index = 0

# Import AI Query Interface
# Dictionary to store AI interfaces per company
ai_interfaces = {}

def get_ai_interface(companyfn=None):
    """Get or create AI interface for a specific company"""
    if companyfn not in ai_interfaces:
        try:
            from src.query.ai_query_interface import AIQueryInterface
            ai_interfaces[companyfn] = AIQueryInterface(companyfn=companyfn)
            if len(ai_interfaces) == 1:
                print("[OK] AI Query Interface loaded successfully")
        except Exception as e:
            print(f"[WARNING] AI Query Interface not available: {str(e)}")
            return None
    return ai_interfaces[companyfn]

AI_INTERFACE_AVAILABLE = True

class GeminiChatHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/' or self.path == '/index.html':
            self.path = '/gemini_chat.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/chat':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                user_message = data.get('message', '')
                # Extract companyfn from request context for data isolation
                companyfn = data.get('companyfn', None)
                # Get conversation ID (for chat history)
                conversation_id = data.get('conversation_id', str(uuid.uuid4()))
                
                # ✅ GET CONTEXT MANAGER AND CONVERSATION
                context_mgr = get_context_manager()
                context = context_mgr.get_or_create_conversation(conversation_id, companyfn)
                
                # ✅ PARSE INTENT AND RESOLVE REFERENCES
                intent_analyzer = QueryIntentAnalyzer()
                intent = intent_analyzer.analyze(user_message, context.to_dict())
                
                resolver = ContextAwareQueryResolver()
                resolved = resolver.resolve_query(user_message, context)
                
                # Use resolved query for processing
                effective_query = resolved['resolved_query']
                
                # Get response from OpenRouter
                response = self.get_openrouter_response(effective_query, companyfn, context)
                
                # ✅ STORE IN CONTEXT
                context.add_message("user", user_message, {
                    'intent': intent.query_type,
                    'sub_type': intent.sub_type,
                    'confidence': intent.confidence
                })
                context.add_message("assistant", response)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                # ✅ GENERATE FOLLOW-UP SUGGESTIONS
                suggestions = self._generate_suggestions(response, intent, context)
                
                response_data = json.dumps({
                    'success': True,
                    'response': response,
                    'conversation_id': conversation_id,
                    'context_applied': resolved['context_applied'],
                    'intent': {
                        'type': intent.query_type,
                        'sub_type': intent.sub_type,
                        'confidence': intent.confidence
                    },
                    'suggestions': suggestions
                })
                self.wfile.write(response_data.encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                error_response = json.dumps({
                    'success': False,
                    'error': str(e)
                })
                self.wfile.write(error_response.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
    
    def get_openrouter_response(self, user_message, companyfn=None, context=None):
        """
        ✅ SMART MULTI-PROVIDER AI RESPONSE WITH AUTOMATIC FALLBACK
        Khi 1 provider lỗi / hết quota / rate limit, hệ thống sẽ TỰ ĐỘNG chuyển sang cái kế tiếp
        Không có lỗi nữa, chỉ chuyển đổi liên tục cho đến khi có provider hoạt động
        """
        global current_provider_index
        
        try:
            # Get company-specific AI interface for data isolation
            ai_interface = get_ai_interface(companyfn)
            
            # Check if this is a data query
            data_keywords = [
                'customer', 'product', 'sales', 'trend', 'forecast', 'churn', 'analysis', 'top', 'bestseller', 'revenue', 'order',
                'month', 'year', 'date', 'daily', 'monthly', 'yearly', 'period',
                'january', 'february', 'march', 'april', 'may', 'june',
                'july', 'august', 'september', 'october', 'november', 'december',
                'triển vọng', 'xu hướng', 'phổ biến', 'nổi bật', 'tương lai', 'sắp tới', 'hot trend', 'tiềm năng'
            ]
            
            is_data_query = any(keyword in user_message.lower() for keyword in data_keywords)
            
            # If it's a data query and AI Interface is available
            if is_data_query and ai_interface:
                try:
                    # Use AI Query Interface to get insights (filtered by companyfn)
                    ai_result = ai_interface.process_query(user_message)
                    ai_response = ai_interface.format_response(ai_result)
                    
                    # Create prompt for OpenRouter with context from AI Interface
                    system_prompt = f"""You are an internal AI Assistant for the company's ERP/BI system.

                    Your role:
                    - Analyze business data
                    - Support decision making
                    - Explain data analysis results
                    - Answer questions related to sales, customers, products, employees, and business performance.
                    - If the user asks for a 'summary', prioritize mentioning Total Revenue, Total Transactions, and Growth Rate in a structured list.

                    - TIME FILTERING LOGIC:
                    - If the user asks for a specific month (e.g., "01/2010"), filter the DATA SOURCE to only include January 2010.
                    - If the user asks for a specific date (e.g., "28/12/2010"), focus strictly on that day's data.
                    - If the timeframe is not found in the DATA SOURCE, state that no records exist for that period.

                    DATA ACCESS PERMISSIONS

                    You are operating in the company's internal environment and are authorized to use data from the internal ERP and database systems.

                    Available data may include:
                    - Order data
                    - Revenue
                    - Customer data
                    - Product data
                    - Employee data
                    - Analysis reports
                    - Machine learning results

                    You do NOT need to refuse due to "no data access permission".
                    Consider the data provided in the system as valid internal data for analysis.

                    DATA SOURCE FOR ANALYSIS

                    Analysis results from the backend system are provided below:

                    {ai_response}

                    You must:
                    1. Identify if there is a specific Date or Month in the user's question.
                    2. Analyze the data from the results provided below *strictly* for that timeframe.
                    3. Summarize important insights and answer the user's question.

                    RESPONSE RULES

                    - Always base responses on provided data
                    - Do not fabricate data if it doesn't exist
                    - If data is insufficient, request additional information
                    - Explain clearly and understandably
                    - Prefer bullet points or tables when needed
                    - You may use emojis for friendliness but don't overuse them
                    - STRICT DATA SOURCE ADHERENCE: You are ONLY allowed to use metrics present in the {ai_response}. 
                    - ANTI-HALLUCINATION: If the data does not contain "Customer names", "Product names", or "Employee names", DO NOT mention or invent them. 
                    - NULL DATA HANDLING: If a metric (e.g., num_transactions) is 0 or null in the data source, report it as 0. Do not assume there is hidden data.
                    - REVENUE vs TRANSACTIONS: Use the 'amt_local' for revenue and 'num_transactions' for the count. If 'num_transactions' is available, use it to calculate 'Average Revenue per Transaction' as an extra insight.
                    - NO HISTORICAL ASSUMPTION: Do not compare with previous months unless the specific growth_rate or previous_data is provided in the {ai_response}.

                    LANGUAGE

                    - Always respond in English
                    - Clear, professional, and easy-to-understand tone"""
                    
                except Exception as e:
                    # If AI Interface fails, use default prompt
                    system_prompt = """You are an AI Assistant specializing in sales data analysis and machine learning.
                    You can:
                    - Answer questions about sales data
                    - Explain ML/AI concepts like churn prediction, sales forecast
                    - Support Python coding and debugging
                    - Analyze business trends
                    - Provide suggestions to improve sales

                    Respond in English, be friendly and easy to understand. Use emojis to create a friendly atmosphere."""
            else:
                # Default prompt for other questions
                system_prompt = """You are an internal AI Assistant for the company's ERP/BI system.

                Your role:
                - Analyze business data
                - Support decision making
                - Explain data analysis results
                - Answer questions related to sales, customers, products, employees, and business performance.

                DATA ACCESS PERMISSIONS

                You are operating in the company's internal environment and are authorized to use data from the internal ERP and database systems.

                Available data may include:
                - Order data
                - Revenue
                - Customer data
                - Product data
                - Employee data
                - Analysis reports
                - Machine learning results

                You do NOT need to refuse due to "no data access permission".
                Consider the data provided in the system as valid internal data for analysis.

                DATA SOURCE FOR ANALYSIS

                Analysis results from the backend system are provided below:


                You must:
                1. Analyze the data in these results
                2. Summarize important insights
                3. Answer the user's question based on the data

                RESPONSE RULES

                - Always base responses on provided data
                - Do not fabricate data if it doesn't exist
                - If data is insufficient, request additional information
                - Explain clearly and understandably
                - Prefer bullet points or tables when needed
                - You may use emojis for friendliness but don't overuse them

                LANGUAGE

                - Always respond in English
                - Clear, professional, and easy-to-understand tone"""
            
            # ✅ BUILD CONVERSATION HISTORY WITH PREVIOUS MESSAGES
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                }
            ]
            
            # Add previous messages from conversation history
            if context.messages:
                for msg in context.messages:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": user_message
            })

            # ✅ FALLBACK LOOP - TRY ALL PROVIDERS UNTIL ONE WORKS
            total_providers = len(AI_PROVIDERS)
            
            for attempt in range(total_providers):
                provider = AI_PROVIDERS[current_provider_index]
                
                if not provider['enabled'] or not provider['working']:
                    # Bỏ qua provider đã tắt / bị lỗi trước đó
                    current_provider_index = (current_provider_index + 1) % total_providers
                    continue
                
                try:
                    print(f"🔄 Đang thử kết nối với: {provider['name']}")
                    
                    # Handle Claude (Anthropic) separately
                    if provider.get('type') == 'anthropic':
                        client = Anthropic(api_key=provider['api_key'])
                        # Build messages for Anthropic (skip system message, use it as system param)
                        anthropic_messages = messages[1:]  # Skip system message
                        response = client.messages.create(
                            model=provider['model'],
                            max_tokens=1000,
                            system=messages[0]['content'],
                            messages=anthropic_messages
                        )
                        # ✅ THÀNH CÔNG!
                        print(f"✅ Kết nối thành công với: {provider['name']}")
                        return response.content[0].text
                    else:
                        # Create client cho provider hiện tại (OpenAI-compatible)
                        client = OpenAI(
                            base_url=provider.get('base_url'),
                            api_key=provider['api_key'],
                            timeout=30
                        )
                        
                        # Thêm headers đặc biệt cho OpenRouter
                        extra_headers = {}
                        if provider['name'] == 'OpenRouter':
                            extra_headers = {
                                "HTTP-Referer": f"http://localhost:{PORT}",
                                "X-Title": "ERP AI Chat Assistant",
                            }
                        
                        # Gọi API
                        response = client.chat.completions.create(
                            model=provider['model'],
                            messages=messages,
                            max_tokens=1000,
                            temperature=0.7,
                            extra_headers=extra_headers
                        )
                        
                        # ✅ THÀNH CÔNG!
                        print(f"✅ Kết nối thành công với: {provider['name']}")
                        return response.choices[0].message.content
                    
                except Exception as provider_error:
                    # ❌ Provider này bị lỗi / hết quota
                    print(f"❌ Lỗi với {provider['name']}: {str(provider_error)[:100]}...")
                    print(f"🔀 Tự động chuyển sang provider kế tiếp...")
                    
                    # Đánh dấu provider này là không hoạt động cho đến lần restart
                    AI_PROVIDERS[current_provider_index]['working'] = False
                    
                    # Chuyển sang provider kế tiếp
                    current_provider_index = (current_provider_index + 1) % total_providers
                    
                    # Tiếp tục vòng lặp
                    continue
            
            # ❌ TẤT CẢ PROVIDER ĐỀU LỖI
            return "⚠️ Tất cả các dịch vụ AI hiện đang bận. Vui lòng thử lại sau vài phút."
            
        except Exception as e:
            return f"Sorry, I encountered an error while processing your question: {str(e)}"
    
    def _generate_suggestions(self, response, intent, context):
        """
        Generate follow-up question suggestions based on response and intent
        
        Rules:
        - Suggestions must be based on the current result
        - Suggestions must help user drill down into data
        - Keep them short and clickable
        - Use natural English
        
        Args:
            response: AI response text
            intent: Parsed intent
            context: Conversation context
            
        Returns:
            List of 3 suggested follow-up questions
        """
        import re
        
        suggestions = []
        response_lower = response.lower()
        
        # Extract numbers from response (e.g., "54 transactions", "10 orders")
        number_pattern = r'(\d+)\s+(transactions|orders|items|customers|products)'
        number_matches = re.findall(number_pattern, response_lower)
        
        # Extract currency amounts (e.g., "$1,234.56")
        currency_pattern = r'\$[\d,]+\.?\d*'
        currency_matches = re.findall(currency_pattern, response)
        
        # Extract dates (e.g., "01/2010", "January 2010")
        date_pattern = r'(\d{1,2}/\d{4}|[A-Za-z]+\s+\d{4})'
        date_matches = re.findall(date_pattern, response)
        
        # Generate suggestions based on intent type
        if intent.query_type == 'revenue_report':
            if number_matches:
                count, item_type = number_matches[0]
                suggestions.append(f"What are the {count} {item_type}?")
                suggestions.append(f"Show top 5 {item_type}")
                suggestions.append(f"Which {item_type[:-1]} has the highest value?")
            else:
                suggestions.append("Show me the transaction details")
                suggestions.append("What's the breakdown by customer?")
                suggestions.append("Compare with previous period")
        
        elif intent.query_type == 'customer_analysis':
            suggestions.append("Show top 10 customers by revenue")
            suggestions.append("Which customers are at risk?")
            suggestions.append("What's the average order value?")
        
        elif intent.query_type == 'product_analysis':
            suggestions.append("Show bestselling products")
            suggestions.append("Which products are underperforming?")
            suggestions.append("What's the product mix?")
        
        elif intent.query_type == 'sales_trend':
            suggestions.append("Show monthly trends")
            suggestions.append("What's the growth rate?")
            suggestions.append("Compare year-over-year")
        
        else:
            # Default suggestions
            if number_matches:
                count, item_type = number_matches[0]
                suggestions.append(f"Tell me more about the {count} {item_type}")
                suggestions.append(f"Show details of top items")
                suggestions.append(f"What's the breakdown?")
            else:
                suggestions.append("Can you provide more details?")
                suggestions.append("Show me the data")
                suggestions.append("What's the breakdown?")
        
        # Ensure we have exactly 3 suggestions
        while len(suggestions) < 3:
            suggestions.append("Tell me more")
        
        return suggestions[:3]

def main():
    """Start the server"""
    print(f"Starting AI Chat Server (OpenRouter)...")
    print(f"Open your browser and go to: http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop the server")
    print()
    
    # ✅ CHECK ALL PROVIDERS STATUS
    print("AI Providers Status:")
    print("-" * 60)
    
    for i, provider in enumerate(AI_PROVIDERS):
        status = "ENABLED" if provider['enabled'] else "DISABLED"
        print(f"  {i+1}. {provider['name']:15} | Model: {provider['model']:30} | {status}")
    
    print()
    print(f"System ready! Multi-provider fallback activated.")
    print(f"Auto Cloude, Gemini, Groq, Together.ai, OpenAI")
    print()
    
    with socketserver.TCPServer(("", PORT), GeminiChatHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    main()