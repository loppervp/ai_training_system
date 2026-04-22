"""
Churn Predictor Module
Module for predicting customer churn probability
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from .model_trainer import ModelTrainer

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Class for predicting customer churn"""
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize ChurnPredictor
        
        Args:
            model_dir: Directory to store models
        """
        self.model_trainer = ModelTrainer(model_dir)
        self.feature_columns = [
            'recency', 'frequency', 'monetary',
            'avg_order_value', 'days_since_last_purchase',
            'total_purchases', 'total_spent',
            'unique_products', 'unique_categories'
        ]
        self.target_column = 'is_churned'
    
    def prepare_churn_data(self, df: pd.DataFrame, churn_threshold_days: int = 90) -> pd.DataFrame:
        """
        Prepare data for churn prediction
        
        Args:
            df: Customer data DataFrame
            churn_threshold_days: Number of days to define churn
            
        Returns:
            Prepared DataFrame
        """
        try:
            logger.info("Preparing churn prediction data")
            
            df_prep = df.copy()
            
            # Create target variable
            if 'days_since_last_purchase' in df_prep.columns:
                df_prep['is_churned'] = (df_prep['days_since_last_purchase'] > churn_threshold_days).astype(int)
            
            # Handle missing values
            for col in self.feature_columns:
                if col in df_prep.columns:
                    df_prep[col] = df_prep[col].fillna(0)
            
            # Create additional features
            if 'total_spent' in df_prep.columns and 'total_purchases' in df_prep.columns:
                df_prep['avg_order_value'] = df_prep['total_spent'] / df_prep['total_purchases'].replace(0, 1)
            
            if 'total_purchases' in df_prep.columns and 'customer_lifetime_days' in df_prep.columns:
                df_prep['purchase_frequency'] = df_prep['total_purchases'] / df_prep['customer_lifetime_days'].replace(0, 1)
            
            logger.info(f"Prepared {len(df_prep)} samples")
            
            return df_prep
            
        except Exception as e:
            logger.error(f"Error preparing churn data: {str(e)}")
            raise
    
    def train(
        self,
        df: pd.DataFrame,
        algorithm: str = "random_forest",
        hyperparameter_tuning: bool = False
    ) -> Dict:
        """
        Train churn prediction model
        
        Args:
            df: Data DataFrame
            algorithm: Algorithm to use
            hyperparameter_tuning: Whether to tune hyperparameters
            
        Returns:
            Dict with training results
        """
        try:
            logger.info("Training churn prediction model")
            
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
            result = self.model_trainer.train_classifier(
                X, y,
                model_name="churn_predictor",
                algorithm=algorithm,
                hyperparameter_tuning=hyperparameter_tuning
            )
            
            # Add feature information
            result['feature_columns'] = available_features
            result['churn_rate'] = y.mean()
            
            logger.info(f"Completed training churn predictor: F1={result['metrics']['f1_score']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training churn predictor: {str(e)}")
            raise
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict churn for customers
        
        Args:
            df: Customer data DataFrame
            
        Returns:
            DataFrame with predictions
        """
        try:
            logger.info("Predicting churn")
            
            df_pred = df.copy()
            
            # Prepare features
            available_features = [col for col in self.feature_columns if col in df_pred.columns]
            
            X, _ = self.model_trainer.prepare_features(
                df_pred,
                feature_columns=available_features,
                categorical_columns=[]
            )
            
            # Predict
            predictions = self.model_trainer.predict("churn_predictor", X)
            
            df_pred['churn_prediction'] = predictions
            df_pred['churn_risk'] = pd.cut(
                predictions,
                bins=[-0.5, 0.5, 1.5],
                labels=['Low', 'High']
            )
            
            logger.info(f"Predicted {len(df_pred)} customers")
            
            return df_pred
            
        except Exception as e:
            logger.error(f"Error predicting churn: {str(e)}")
            raise
    
    def get_churn_insights(self, df: pd.DataFrame) -> Dict:
        """
        Get insights about churn
        
        Args:
            df: DataFrame with predictions
            
        Returns:
            Dict with insights
        """
        try:
            insights = {
                'total_customers': len(df),
                'churned_customers': df['churn_prediction'].sum() if 'churn_prediction' in df.columns else 0,
                'churn_rate': df['churn_prediction'].mean() if 'churn_prediction' in df.columns else 0,
                'high_risk_customers': len(df[df['churn_risk'] == 'High']) if 'churn_risk' in df.columns else 0
            }
            
            # Top features affecting churn
            if hasattr(self.model_trainer.models.get('churn_predictor'), 'feature_importances_'):
                importances = self.model_trainer.models['churn_predictor'].feature_importances_
                feature_importance = dict(zip(self.feature_columns[:len(importances)], importances))
                insights['top_churn_factors'] = sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting churn insights: {str(e)}")
            raise