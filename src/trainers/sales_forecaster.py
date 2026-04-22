"""
Sales Forecaster Module
Module for predicting sales revenue
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class SalesForecaster:
    """Class for predicting sales revenue"""
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize SalesForecaster
        
        Args:
            model_dir: Directory to store models
        """
        self.model_trainer = ModelTrainer(model_dir)
        self.feature_columns = [
            'year', 'month', 'quarter', 'day_of_week',
            'day_of_month', 'week_of_year', 'is_weekend',
            'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos'
        ]
        self.target_column = 'total_revenue'
    
    def prepare_forecast_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for forecast
        
        Args:
            df: Sales trend DataFrame
            
        Returns:
            Prepared DataFrame
        """
        try:
            logger.info("Preparing forecast data")
            
            df_prep = df.copy()
            
            # Ensure date column exists
            if 'transaction_date' in df_prep.columns:
                df_prep['transaction_date'] = pd.to_datetime(df_prep['transaction_date'])
                
                # Create temporal features
                df_prep['year'] = df_prep['transaction_date'].dt.year
                df_prep['month'] = df_prep['transaction_date'].dt.month
                df_prep['quarter'] = df_prep['transaction_date'].dt.quarter
                df_prep['day_of_week'] = df_prep['transaction_date'].dt.dayofweek
                df_prep['day_of_month'] = df_prep['transaction_date'].dt.day
                df_prep['week_of_year'] = df_prep['transaction_date'].dt.isocalendar().week
                df_prep['is_weekend'] = df_prep['day_of_week'].isin([5, 6]).astype(int)
                
                # Cyclical features
                df_prep['month_sin'] = np.sin(2 * np.pi * df_prep['month'] / 12)
                df_prep['month_cos'] = np.cos(2 * np.pi * df_prep['month'] / 12)
                df_prep['day_of_week_sin'] = np.sin(2 * np.pi * df_prep['day_of_week'] / 7)
                df_prep['day_of_week_cos'] = np.cos(2 * np.pi * df_prep['day_of_week'] / 7)
            
            # Handle missing values
            for col in self.feature_columns:
                if col in df_prep.columns:
                    df_prep[col] = df_prep[col].fillna(0)
            
            if self.target_column in df_prep.columns:
                df_prep[self.target_column] = df_prep[self.target_column].fillna(0)
            
            logger.info(f"Prepared {len(df_prep)} samples")
            
            return df_prep
            
        except Exception as e:
            logger.error(f"Error preparing forecast data: {str(e)}")
            raise
    
    def train(
        self,
        df: pd.DataFrame,
        algorithm: str = "random_forest"
    ) -> Dict:
        """
        Train sales forecast model
        
        Args:
            df: Data DataFrame
            algorithm: Algorithm to use
            
        Returns:
            Dict with training results
        """
        try:
            logger.info("Training sales forecast model")
            
            # Prepare features
            available_features = [col for col in self.feature_columns if col in df.columns]
            
            X, y = self.model_trainer.prepare_features(
                df,
                feature_columns=available_features,
                target_column=self.target_column,
                categorical_columns=[]
            )
            
            if y is None:
                raise ValueError(f"Target column '{self.target_column}' does not exist")
            
            # Training
            result = self.model_trainer.train_regressor(
                X, y,
                model_name="sales_forecaster",
                algorithm=algorithm
            )
            
            # Add feature information
            result['feature_columns'] = available_features
            result['avg_daily_sales'] = y.mean()
            
            logger.info(f"Completed training sales forecaster: R2={result['metrics']['r2']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training sales forecaster: {str(e)}")
            raise
    
    def forecast(
        self,
        df: pd.DataFrame,
        forecast_days: int = 30
    ) -> pd.DataFrame:
        """
        Predict sales for future dates
        
        Args:
            df: Historical data DataFrame
            forecast_days: Number of days to forecast
            
        Returns:
            DataFrame with forecasts
        """
        try:
            logger.info(f"Forecasting sales for {forecast_days} days")
            
            # Create future dates
            last_date = df['transaction_date'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Create DataFrame for future
            df_future = pd.DataFrame({'transaction_date': future_dates})
            
            # Create temporal features
            df_future['year'] = df_future['transaction_date'].dt.year
            df_future['month'] = df_future['transaction_date'].dt.month
            df_future['quarter'] = df_future['transaction_date'].dt.quarter
            df_future['day_of_week'] = df_future['transaction_date'].dt.dayofweek
            df_future['day_of_month'] = df_future['transaction_date'].dt.day
            df_future['week_of_year'] = df_future['transaction_date'].dt.isocalendar().week
            df_future['is_weekend'] = df_future['day_of_week'].isin([5, 6]).astype(int)
            df_future['month_sin'] = np.sin(2 * np.pi * df_future['month'] / 12)
            df_future['month_cos'] = np.cos(2 * np.pi * df_future['month'] / 12)
            df_future['day_of_week_sin'] = np.sin(2 * np.pi * df_future['day_of_week'] / 7)
            df_future['day_of_week_cos'] = np.cos(2 * np.pi * df_future['day_of_week'] / 7)
            
            # Prepare features
            available_features = [col for col in self.feature_columns if col in df_future.columns]
            
            X, _ = self.model_trainer.prepare_features(
                df_future,
                feature_columns=available_features,
                categorical_columns=[]
            )
            
            # Predict
            predictions = self.model_trainer.predict("sales_forecaster", X)
            
            df_future['forecasted_revenue'] = predictions
            df_future['forecast_type'] = 'daily'
            
            logger.info(f"Forecasted {len(df_future)} days")
            
            return df_future
            
        except Exception as e:
            logger.error(f"Error forecasting sales: {str(e)}")
            raise
    
    def get_forecast_insights(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame
    ) -> Dict:
        """
        Get insights from forecast
        
        Args:
            historical_df: Historical data DataFrame
            forecast_df: Forecast DataFrame
            
        Returns:
            Dict with insights
        """
        try:
            insights = {
                'historical_avg_daily_sales': historical_df[self.target_column].mean() if self.target_column in historical_df.columns else 0,
                'forecasted_avg_daily_sales': forecast_df['forecasted_revenue'].mean(),
                'forecast_period_days': len(forecast_df),
                'total_forecasted_revenue': forecast_df['forecasted_revenue'].sum(),
                'forecast_start_date': forecast_df['transaction_date'].min().isoformat() if len(forecast_df) > 0 else None,
                'forecast_end_date': forecast_df['transaction_date'].max().isoformat() if len(forecast_df) > 0 else None
            }
            
            # Calculate growth rate
            if insights['historical_avg_daily_sales'] > 0:
                insights['expected_growth_rate'] = (
                    (insights['forecasted_avg_daily_sales'] - insights['historical_avg_daily_sales']) /
                    insights['historical_avg_daily_sales'] * 100
                )
            else:
                insights['expected_growth_rate'] = 0
            
            # Weekly breakdown
            if 'transaction_date' in forecast_df.columns:
                forecast_df['week'] = forecast_df['transaction_date'].dt.isocalendar().week
                weekly_forecast = forecast_df.groupby('week')['forecasted_revenue'].sum()
                insights['weekly_forecast'] = weekly_forecast.to_dict()
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting forecast insights: {str(e)}")
            raise