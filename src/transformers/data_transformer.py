"""
Data Transformer Module
Module for transforming and cleaning data for AI training
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class DataTransformer:
    """Class for transforming and cleaning data"""
    
    def __init__(self, mapping_path: str = "config/mapping.json"):
        """
        Initialize DataTransformer
        
        Args:
            mapping_path: Path to mapping file
        """
        self.mapping = self._load_mapping(mapping_path)
        self.transaction_type_mapping = self._get_transaction_type_mapping()
    
    def _load_mapping(self, mapping_path: str) -> Dict:
        """Read mapping file"""
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Mapping file not found: {mapping_path}")
            return {}
    
    def _get_transaction_type_mapping(self) -> Dict:
        """Get transaction type mapping"""
        try:
            return self.mapping.get('database_tables', {}).get('scm_sal_main', {}).get('columns', {}).get('tag_table_usage', {}).get('mapping_values', {})
        except:
            return {
                'sal_soe': 'Sales Order Entry',
                'sal_soc': 'Sales Order Confirmation',
                'sal_inv': 'Sales Invoice',
                'sal_quo': 'Sales Quotation',
                'sal_cn': 'Sales Credit Note',
                'stk_do': 'Delivery Order',
                'stk_doc': 'Delivery Order Confirmation'
            }
        
    def transform_date_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data for date-based revenue analysis
        """
        try:
            logger.info("Transforming date revenue data")
            df_clean = df.copy()
            
            # 1. Ep kieu du lieu
            df_clean['month'] = pd.to_numeric(df_clean['month'], errors='coerce').fillna(0).astype(int)
            df_clean['year'] = pd.to_numeric(df_clean['year'], errors='coerce').fillna(0).astype(int)
            df_clean['amt_local'] = pd.to_numeric(df_clean['amt_local'], errors='coerce').fillna(0)
            
            # 2. Loai bo cac dong loi
            df_clean = df_clean[(df_clean['month'] > 0) & (df_clean['year'] > 0)]
            
            # 3. Them cot bo tro
            if not df_clean.empty:
                df_clean['revenue_level'] = pd.qcut(
                    df_clean['amt_local'],
                    q=3,
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                )
                
            return df_clean

        except Exception as e:
            logger.error(f"Error transforming date revenue data: {str(e)}")
            raise
    
    def clean_sales_main(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean scm_sal_main data - preserves companyfn for data isolation
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Cleaned DataFrame with companyfn preserved
        """
        try:
            logger.info(f"Starting to clean scm_sal_main: {len(df)} records")
            
            df_clean = df.copy()
            
            # Ensure companyfn is preserved for data isolation
            if 'companyfn' not in df_clean.columns:
                logger.warning("companyfn column not found - data isolation may not work correctly")
            
            # 1. Handle null values
            df_clean['party_desc'] = df_clean['party_desc'].fillna('Unknown')
            df_clean['party_code'] = df_clean['party_code'].fillna('N/A')
            df_clean['deptunit_desc'] = df_clean['deptunit_desc'].fillna('N/A')
            df_clean['staff_code'] = df_clean['staff_code'].fillna('N/A')
            df_clean['notes_memo'] = df_clean['notes_memo'].fillna('')
            
            # 2. Convert data types
            df_clean['date_trans'] = pd.to_datetime(df_clean['date_trans'], errors='coerce')
            df_clean['date_due'] = pd.to_datetime(df_clean['date_due'], errors='coerce')
            
            df_clean['amount_local'] = pd.to_numeric(df_clean['amount_local'], errors='coerce').fillna(0)
            df_clean['amount_forex'] = pd.to_numeric(df_clean['amount_forex'], errors='coerce').fillna(0)
            df_clean['salestaxpct'] = pd.to_numeric(df_clean['salestaxpct'], errors='coerce').fillna(0)
            df_clean['curr_rate_forex_f_calc'] = pd.to_numeric(df_clean['curr_rate_forex_f_calc'], errors='coerce').fillna(1)
            
            # 3. Add transaction_type_name column
            df_clean['transaction_type_name'] = df_clean['tag_table_usage'].map(self.transaction_type_mapping).fillna('Unknown')
            
            # 4. Add temporal features
            df_clean['year'] = df_clean['date_trans'].dt.year
            df_clean['month'] = df_clean['date_trans'].dt.month
            df_clean['quarter'] = df_clean['date_trans'].dt.quarter
            df_clean['day_of_week'] = df_clean['date_trans'].dt.dayofweek
            df_clean['day_of_month'] = df_clean['date_trans'].dt.day
            df_clean['week_of_year'] = df_clean['date_trans'].dt.isocalendar().week
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            df_clean['is_month_end'] = (df_clean['day_of_month'] >= 25).astype(int)
            
            # 5. Calculate metrics
            df_clean['amount_local_log'] = np.log1p(df_clean['amount_local'].abs())
            
            # 6. Handle outliers (remove transactions with extremely high values)
            q99 = df_clean['amount_local'].quantile(0.99)
            df_clean['is_outlier'] = (df_clean['amount_local'].abs() > q99).astype(int)
            
            logger.info(f"Completed cleaning scm_sal_main: {len(df_clean)} records")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning scm_sal_main: {str(e)}")
            raise
    
    def clean_sales_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean scm_sal_data data
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"Starting to clean scm_sal_data: {len(df)} records")
            
            df_clean = df.copy()
            
            # 1. Handle null values
            df_clean['stkcode_desc'] = df_clean['stkcode_desc'].fillna('Unknown Product')
            df_clean['stkcode_code'] = df_clean['stkcode_code'].fillna('N/A')
            df_clean['brand_desc'] = df_clean['brand_desc'].fillna('No Brand')
            df_clean['stkcate_desc'] = df_clean['stkcate_desc'].fillna('No Category')
            df_clean['stkvendor_desc'] = df_clean['stkvendor_desc'].fillna('No Vendor')
            df_clean['notes_memo'] = df_clean['notes_memo'].fillna('')
            
            # 2. Convert data types
            df_clean['date_trans'] = pd.to_datetime(df_clean['date_trans'], errors='coerce')
            
            numeric_cols = [
                'qnty_total', 'qnty_uomstk', 'bal_qnty_total', 'bal_qnty_uomstk',
                'price_unitlist_forex', 'price_unitlist_local',
                'price_unitrate_forex', 'price_unitrate_local',
                'discount_pct', 'amount_forex', 'amount_local',
                'amount_tax_forex', 'amount_tax_local'
            ]
            
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
            # 3. Calculate metrics
            df_clean['line_total'] = df_clean['qnty_total'] * df_clean['price_unitrate_local']
            df_clean['discount_amount'] = df_clean['line_total'] * df_clean['discount_pct'] / 100
            df_clean['net_amount'] = df_clean['line_total'] - df_clean['discount_amount']
            
            # 4. Add temporal features
            df_clean['year'] = df_clean['date_trans'].dt.year
            df_clean['month'] = df_clean['date_trans'].dt.month
            df_clean['quarter'] = df_clean['date_trans'].dt.quarter
            
            # 5. Create product identifier
            df_clean['stkcode_code'] = df_clean['stkcode_code'].astype(str).fillna('N/A')
            df_clean['stkcode_unique'] = df_clean['stkcode_unique'].astype(str).fillna('0')
            
            df_clean['product_id'] = df_clean['stkcode_code'] + '_' + df_clean['stkcode_unique']
            
            logger.info(f"Completed cleaning scm_sal_data: {len(df_clean)} records")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error cleaning scm_sal_data: {str(e)}")
            raise
    
    def transform_customer_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data for customer analysis
        """
        try:
            logger.info("Transforming customer analysis data")
            
            df_clean = df.copy()
            
            df_clean['customer_name'] = df_clean['customer_name'].fillna('Unknown')
            df_clean['product_name'] = df_clean['product_name'].fillna('Unknown')
            df_clean['category_name'] = df_clean['category_name'].fillna('No Category')
            df_clean['brand_desc'] = df_clean['brand_desc'].fillna('No Brand')
            
            df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'], errors='coerce')
            df_clean['quantity'] = pd.to_numeric(df_clean['quantity'], errors='coerce').fillna(0)
            df_clean['unit_price'] = pd.to_numeric(df_clean['unit_price'], errors='coerce').fillna(0)
            df_clean['line_amount'] = pd.to_numeric(df_clean['line_amount'], errors='coerce').fillna(0)
            
            df_clean['year'] = df_clean['transaction_date'].dt.year
            df_clean['month'] = df_clean['transaction_date'].dt.month
            df_clean['quarter'] = df_clean['transaction_date'].dt.quarter
            df_clean['day_of_week'] = df_clean['transaction_date'].dt.dayofweek
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            
            df_clean['transaction_value_segment'] = pd.qcut(
                df_clean['line_amount'].abs(),
                q=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error transforming customer analysis data: {str(e)}")
            raise
    
    def transform_product_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data for product analysis
        """
        try:
            logger.info("Transforming product analysis data")
            
            df_clean = df.copy()
            
            df_clean['product_name'] = df_clean['product_name'].fillna('Unknown')
            df_clean['category_name'] = df_clean['category_name'].fillna('No Category')
            df_clean['brand_desc'] = df_clean['brand_desc'].fillna('No Brand')
            df_clean['vendor_name'] = df_clean['vendor_name'].fillna('No Vendor')
            
            df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'], errors='coerce')
            df_clean['quantity_sold'] = pd.to_numeric(df_clean['quantity_sold'], errors='coerce').fillna(0)
            df_clean['revenue'] = pd.to_numeric(df_clean['revenue'], errors='coerce').fillna(0)
            df_clean['avg_price'] = pd.to_numeric(df_clean['avg_price'], errors='coerce').fillna(0)
            
            df_clean['year'] = df_clean['transaction_date'].dt.year
            df_clean['month'] = df_clean['transaction_date'].dt.month
            df_clean['quarter'] = df_clean['transaction_date'].dt.quarter
            
            df_clean['sales_velocity'] = df_clean['revenue'] / df_clean['quantity_sold'].replace(0, 1)
            
            df_clean['revenue_segment'] = pd.qcut(
                df_clean['revenue'].abs(),
                q=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error transforming product analysis data: {str(e)}")
            raise
    
    def transform_customer_retention(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data for customer retention analysis
        """
        try:
            logger.info("Transforming customer retention analysis data")
            
            df_clean = df.copy()
            
            df_clean['customer_name'] = df_clean['customer_name'].fillna('Unknown')
            
            date_cols = ['first_purchase_date', 'last_purchase_date']
            for col in date_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
            
            numeric_cols = ['days_since_last_purchase', 'total_purchases', 'total_spent', 
                          'avg_purchase_value', 'purchase_frequency', 'customer_lifetime_days']
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
            df_clean['recency_score'] = pd.qcut(
                df_clean['days_since_last_purchase'],
                q=5,
                labels=[5, 4, 3, 2, 1],
                duplicates='drop'
            )
            
            df_clean['frequency_score'] = pd.qcut(
                df_clean['total_purchases'].rank(method='first'),
                q=5,
                labels=[1, 2, 3, 4, 5],
                duplicates='drop'
            )
            
            df_clean['monetary_score'] = pd.qcut(
                df_clean['total_spent'].rank(method='first'),
                q=5,
                labels=[1, 2, 3, 4, 5],
                duplicates='drop'
            )
            
            df_clean['recency_score'] = pd.to_numeric(df_clean['recency_score'], errors='coerce')
            df_clean['frequency_score'] = pd.to_numeric(df_clean['frequency_score'], errors='coerce')
            df_clean['monetary_score'] = pd.to_numeric(df_clean['monetary_score'], errors='coerce')
            
            df_clean['rfm_score'] = (
                df_clean['recency_score'] + 
                df_clean['frequency_score'] + 
                df_clean['monetary_score']
            ) / 3
            
            df_clean['customer_segment'] = pd.cut(
                df_clean['rfm_score'],
                bins=[0, 2, 3, 4, 5],
                labels=['At Risk', 'Needs Attention', 'Potential Loyalist', 'Champion']
            )
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error transforming retention data: {str(e)}")
            raise
    
    def transform_sales_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data for sales trend analysis
        """
        try:
            logger.info("Transforming sales trend data")
            
            df_clean = df.copy()
            
            # Handle null values
            df_clean['business_unit_name'] = df_clean['business_unit_name'].fillna('Unknown')
            df_clean['salesperson'] = df_clean['salesperson'].fillna('Unknown')
            df_clean['transaction_type'] = df_clean['transaction_type'].fillna('Unknown')
            
            # Convert data types
            df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'], errors='coerce')
            
            numeric_cols = ['total_transactions', 'total_revenue', 'avg_transaction_value', 'unique_customers']
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
            # Add temporal features
            df_clean['year'] = df_clean['transaction_date'].dt.year
            df_clean['month'] = df_clean['transaction_date'].dt.month
            df_clean['quarter'] = df_clean['transaction_date'].dt.quarter
            df_clean['day_of_week'] = df_clean['transaction_date'].dt.dayofweek
            df_clean['day_of_month'] = df_clean['transaction_date'].dt.day
            df_clean['week_of_year'] = df_clean['transaction_date'].dt.isocalendar().week
            df_clean['is_weekend'] = df_clean['day_of_week'].isin([5, 6]).astype(int)
            
            # Cyclical features
            df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['month'] / 12)
            df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['month'] / 12)
            df_clean['day_of_week_sin'] = np.sin(2 * np.pi * df_clean['day_of_week'] / 7)
            df_clean['day_of_week_cos'] = np.cos(2 * np.pi * df_clean['day_of_week'] / 7)
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error transforming trend data: {str(e)}")
            raise

    
    def save_transformed_data(self, df: pd.DataFrame, output_path: str, format: str = 'parquet'):
        """Save transformed data"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'parquet':
                df.to_parquet(output_path, index=False, engine='pyarrow')
            elif format == 'csv':
                df.to_csv(output_path, index=False, encoding='utf-8')
            elif format == 'json':
                df.to_json(output_path, orient='records', force_ascii=False, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data saved: {output_path} ({len(df)} records)")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def load_transformed_data(self, input_path: str, format: Optional[str] = None) -> pd.DataFrame:
        """Load transformed data"""
        try:
            if format is None:
                format = Path(input_path).suffix.lstrip('.')
            
            if format == 'parquet':
                df = pd.read_parquet(input_path)
            elif format == 'csv':
                df = pd.read_csv(input_path, encoding='utf-8')
            elif format == 'json':
                df = pd.read_json(input_path, orient='records')
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Data loaded: {input_path} ({len(df)} records)")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise  