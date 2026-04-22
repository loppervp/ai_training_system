"""
Feature Engineer Module
Module for creating features for ML models
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class for creating features for ML"""
    
    def __init__(self):
        """Initialize FeatureEngineer"""
        pass
    
    def create_rfm_features(self, df: pd.DataFrame, customer_col: str = 'customer_id') -> pd.DataFrame:
        """
        Create RFM features (Recency, Frequency, Monetary)
        
        Args:
            df: Transaction data DataFrame
            customer_col: Customer ID column name
            
        Returns:
            DataFrame with RFM features
        """
        try:
            logger.info("Creating RFM features")
            
            # Calculate current date
            current_date = df['transaction_date'].max()
            
            # Group by customer
            rfm = df.groupby(customer_col).agg({
                'transaction_date': lambda x: (current_date - x.max()).days,  # Recency
                'uniquenum_pri': 'nunique',  # Frequency
                'line_amount': 'sum'  # Monetary
            }).reset_index()
            
            rfm.columns = [customer_col, 'recency', 'frequency', 'monetary']
            
            # Calculate RFM scores
            rfm['recency_score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            rfm['monetary_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            
            # Convert to numeric
            rfm['recency_score'] = pd.to_numeric(rfm['recency_score'], errors='coerce')
            rfm['frequency_score'] = pd.to_numeric(rfm['frequency_score'], errors='coerce')
            rfm['monetary_score'] = pd.to_numeric(rfm['monetary_score'], errors='coerce')
            
            # Calculate combined RFM score
            rfm['rfm_score'] = (rfm['recency_score'] + rfm['frequency_score'] + rfm['monetary_score']) / 3
            
            # Customer segmentation
            rfm['customer_segment'] = pd.cut(
                rfm['rfm_score'],
                bins=[0, 2, 3, 4, 5],
                labels=['At Risk', 'Needs Attention', 'Potential Loyalist', 'Champion']
            )
            
            return rfm
            
        except Exception as e:
            logger.error(f"Error creating RFM features: {str(e)}")
            raise
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'transaction_date') -> pd.DataFrame:
        """
        Create temporal features from transaction date
        
        Args:
            df: Data DataFrame
            date_col: Date column name
            
        Returns:
            DataFrame with temporal features
        """
        try:
            logger.info("Creating temporal features")
            
            df_features = df.copy()
            
            # Ensure date column is datetime
            df_features[date_col] = pd.to_datetime(df_features[date_col], errors='coerce')
            
            # Create features
            df_features['year'] = df_features[date_col].dt.year
            df_features['month'] = df_features[date_col].dt.month
            df_features['quarter'] = df_features[date_col].dt.quarter
            df_features['day_of_week'] = df_features[date_col].dt.dayofweek
            df_features['day_of_month'] = df_features[date_col].dt.day
            df_features['day_of_year'] = df_features[date_col].dt.dayofyear
            df_features['week_of_year'] = df_features[date_col].dt.isocalendar().week
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
            df_features['is_month_start'] = (df_features[date_col].dt.is_month_start).astype(int)
            df_features['is_month_end'] = (df_features[date_col].dt.is_month_end).astype(int)
            df_features['is_quarter_start'] = (df_features[date_col].dt.is_quarter_start).astype(int)
            df_features['is_quarter_end'] = (df_features[date_col].dt.is_quarter_end).astype(int)
            df_features['is_year_start'] = (df_features[date_col].dt.is_year_start).astype(int)
            df_features['is_year_end'] = (df_features[date_col].dt.is_year_end).astype(int)
            
            # Create cyclical features (for ML models)
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
            df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating temporal features: {str(e)}")
            raise
    
    def create_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create product features
        
        Args:
            df: Product data DataFrame
            
        Returns:
            DataFrame with product features
        """
        try:
            logger.info("Creating product features")
            
            # Group by product
            product_features = df.groupby('product_code').agg({
                'quantity_sold': ['sum', 'mean', 'std'],
                'revenue': ['sum', 'mean', 'std'],
                'avg_price': 'mean',
                'unique_customers': 'sum',
                'num_transactions': 'sum'
            }).reset_index()
            
            # Flatten column names
            product_features.columns = [
                'product_code',
                'total_quantity_sold',
                'avg_quantity_per_transaction',
                'std_quantity',
                'total_revenue',
                'avg_revenue_per_transaction',
                'std_revenue',
                'avg_selling_price',
                'total_unique_customers',
                'total_transactions'
            ]
            
            # Calculate sales velocity (revenue/day)
            date_range = (df['transaction_date'].max() - df['transaction_date'].min()).days
            product_features['sales_velocity'] = product_features['total_revenue'] / max(date_range, 1)
            
            # Calculate customer reach
            product_features['customer_reach'] = product_features['total_unique_customers'] / product_features['total_transactions']
            
            return product_features
            
        except Exception as e:
            logger.error(f"Error creating product features: {str(e)}")
            raise
    
    def create_customer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer features
        
        Args:
            df: Customer data DataFrame
            
        Returns:
            DataFrame with customer features
        """
        try:
            logger.info("Creating customer features")
            
            # Group by customer
            customer_features = df.groupby('customer_id').agg({
                'transaction_date': ['min', 'max', 'nunique'],
                'line_amount': ['sum', 'mean', 'std'],
                'quantity': ['sum', 'mean'],
                'product_code': 'nunique',
                'category_code': 'nunique',
                'brand_code': 'nunique'
            }).reset_index()
            
            # Flatten column names
            customer_features.columns = [
                'customer_id',
                'first_purchase_date',
                'last_purchase_date',
                'total_purchase_days',
                'total_spent',
                'avg_order_value',
                'std_order_value',
                'total_quantity',
                'avg_quantity_per_order',
                'unique_products',
                'unique_categories',
                'unique_brands'
            ]
            
            # Calculate customer lifetime
            customer_features['customer_lifetime_days'] = (
                customer_features['last_purchase_date'] - customer_features['first_purchase_date']
            ).dt.days
            
            # Calculate purchase frequency
            customer_features['purchase_frequency'] = (
                customer_features['total_purchase_days'] / 
                customer_features['customer_lifetime_days'].replace(0, 1)
            )
            
            # Calculate days since last purchase
            current_date = df['transaction_date'].max()
            customer_features['days_since_last_purchase'] = (
                current_date - customer_features['last_purchase_date']
            ).dt.days
            
            # Calculate churn risk
            customer_features['churn_risk'] = pd.cut(
                customer_features['days_since_last_purchase'],
                bins=[0, 30, 60, 90, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            return customer_features
            
        except Exception as e:
            logger.error(f"Error creating customer features: {str(e)}")
            raise
    
    def create_lag_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: str = 'transaction_date',
        group_col: Optional[str] = None,
        lags: List[int] = [1, 7, 30]
    ) -> pd.DataFrame:
        """
        Create lag features for time series
        
        Args:
            df: Data DataFrame
            value_col: Value column name
            date_col: Date column name
            group_col: Group column name (if any)
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        try:
            logger.info(f"Creating lag features for {value_col}")
            
            df_features = df.copy()
            df_features = df_features.sort_values(date_col)
            
            if group_col:
                for lag in lags:
                    df_features[f'{value_col}_lag_{lag}'] = df_features.groupby(group_col)[value_col].shift(lag)
            else:
                for lag in lags:
                    df_features[f'{value_col}_lag_{lag}'] = df_features[value_col].shift(lag)
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating lag features: {str(e)}")
            raise
    
    def create_rolling_features(
        self,
        df: pd.DataFrame,
        value_col: str,
        date_col: str = 'transaction_date',
        group_col: Optional[str] = None,
        windows: List[int] = [7, 30, 90]
    ) -> pd.DataFrame:
        """
        Create rolling features for time series
        
        Args:
            df: Data DataFrame
            value_col: Value column name
            date_col: Date column name
            group_col: Group column name (if any)
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        try:
            logger.info(f"Creating rolling features for {value_col}")
            
            df_features = df.copy()
            df_features = df_features.sort_values(date_col)
            
            for window in windows:
                if group_col:
                    df_features[f'{value_col}_rolling_mean_{window}'] = df_features.groupby(group_col)[value_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    df_features[f'{value_col}_rolling_std_{window}'] = df_features.groupby(group_col)[value_col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                else:
                    df_features[f'{value_col}_rolling_mean_{window}'] = df_features[value_col].rolling(window=window, min_periods=1).mean()
                    df_features[f'{value_col}_rolling_std_{window}'] = df_features[value_col].rolling(window=window, min_periods=1).std()
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating rolling features: {str(e)}")
            raise
    
    def create_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: List[tuple]
    ) -> pd.DataFrame:
        """
        Create interaction features between features
        
        Args:
            df: Data DataFrame
            feature_pairs: List of feature pairs [(feat1, feat2), ...]
            
        Returns:
            DataFrame with interaction features
        """
        try:
            logger.info("Creating interaction features")
            
            df_features = df.copy()
            
            for feat1, feat2 in feature_pairs:
                if feat1 in df_features.columns and feat2 in df_features.columns:
                    # Multiplication interaction
                    df_features[f'{feat1}_x_{feat2}'] = df_features[feat1] * df_features[feat2]
                    
                    # Ratio interaction (avoid division by zero)
                    df_features[f'{feat1}_div_{feat2}'] = df_features[feat1] / df_features[feat2].replace(0, np.nan)
            
            return df_features
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {str(e)}")
            raise