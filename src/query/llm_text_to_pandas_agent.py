"""
LLM Text-to-Pandas Agent
Natural language query to pandas dataframe transformation
No more hardcoded if-else handlers for each query type
"""
import logging
import pandas as pd
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import os
import openai
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

load_dotenv()

logger = logging.getLogger(__name__)


class LLMTextToPandasAgent:
    """
    AI Agent that converts natural language questions into Pandas operations
    Automatically generates and executes pandas code on real data
    No more hardcoding _handle_xxx_query() functions
    """
    
    def __init__(
        self,
        data_path: str = "data/processed",
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo-1106",
        temperature: float = 0.0
    ):
        """
        Initialize LLM Agent
        
        Args:
            data_path: Path to parquet data files
            openai_api_key: OpenAI API Key (from .env if not provided)
            model_name: OpenAI model to use
            temperature: 0 = deterministic, 1 = creative
        """
        self.data_path = data_path
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        self.temperature = temperature
        
        # Available datasets with descriptions
        self.dataset_info = {
            'sales_main': {
                'file': 'sales_main.parquet',
                'description': 'Main sales transactions, each row is one order',
                'columns': ['dnum_auto', 'party_desc', 'amount_local', 'date_trans', 'companyfn']
            },
            'sales_data': {
                'file': 'sales_data.parquet',
                'description': 'Line items per order, product details',
                'columns': ['stkcode_code', 'stkcode_desc', 'qnty_total', 'amount_local']
            },
            'customer_analysis': {
                'file': 'customer_analysis.parquet',
                'description': 'Customer segmentation, churn risk, purchase history',
                'columns': ['customer_id', 'total_purchases', 'last_purchase_date', 'churn_risk']
            },
            'product_analysis': {
                'file': 'product_analysis.parquet',
                'description': 'Product performance, sales trends, categories',
                'columns': ['product_name', 'total_revenue', 'units_sold', 'growth_rate']
            },
            'sales_trend': {
                'file': 'sales_trend.parquet',
                'description': 'Daily/Monthly aggregated sales trends',
                'columns': ['date', 'total_revenue', 'total_orders', 'unique_customers']
            },
            'customer_retention': {
                'file': 'customer_retention.parquet',
                'description': 'Customer retention, churn rate analysis',
                'columns': ['cohort_month', 'retention_rate', 'churn_rate']
            }
        }
        
        self.loaded_datasets = {}
        self.agent = None
        
        logger.info("✅ LLM Text-to-Pandas Agent initialized")
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load parquet dataset into memory"""
        if dataset_name in self.loaded_datasets:
            return self.loaded_datasets[dataset_name]
        
        info = self.dataset_info.get(dataset_name)
        if not info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        file_path = os.path.join(self.data_path, info['file'])
        df = pd.read_parquet(file_path)
        
        self.loaded_datasets[dataset_name] = df
        logger.info(f"✅ Loaded dataset {dataset_name}: {len(df)} rows")
        
        return df
    
    def load_all_datasets(self):
        """Load all available datasets"""
        for name in self.dataset_info:
            try:
                self.load_dataset(name)
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")
    
    def create_agent(self) -> Any:
        """Create LangChain Pandas Agent with all loaded datasets"""
        self.load_all_datasets()
        
        llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key
        )
        
        # Create agent with multiple dataframes
        self.agent = create_pandas_dataframe_agent(
            llm,
            list(self.loaded_datasets.values()),
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            max_iterations=5,
            early_stopping_method="generate",
            number_of_head_rows=5
        )
        
        logger.info("✅ LLM Pandas Agent created successfully")
        return self.agent
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """
        Process natural language question and return answer
        
        Args:
            question: User's natural language question
            
        Returns:
            Dict with answer, data, and insights
        """
        if not self.agent:
            self.create_agent()
        
        try:
            logger.info(f"🤖 Processing with LLM Agent: {question}")
            
            system_prompt = f"""
            You are an expert Data Analyst for Sales Data.
            Available datasets: {json.dumps(self.dataset_info, indent=2)}
            
            Answer the user's question using pandas operations on the data.
            Return only factual answers based on real data.
            Format the answer clearly with numbers, totals, and trends.
            """
            
            response = self.agent.run(f"{system_prompt}\n\nQuestion: {question}")
            
            logger.info(f"✅ LLM Agent response received")
            
            return {
                'query_type': 'llm_agent',
                'summary': response,
                'data': {},
                'insights': [
                    "Answer generated automatically from real data",
                    "No hardcoded handlers used"
                ],
                'is_llm_agent': True
            }
            
        except Exception as e:
            logger.error(f"❌ LLM Agent error: {e}")
            return {
                'error': str(e),
                'summary': "I couldn't process that question. Please try asking differently.",
                'fallback': True
            }