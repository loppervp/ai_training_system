"""
Model Trainer Module
Main module for training ML models
"""

import json
import logging
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Main class for training ML models"""
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize ModelTrainer
        
        Args:
            model_dir: Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare features for training
        
        Args:
            df: Data DataFrame
            feature_columns: List of feature columns
            target_column: Target column name (if any)
            categorical_columns: List of categorical columns
            
        Returns:
            Tuple (X, y) or (X, None)
        """
        try:
            logger.info("Preparing features for training")
            
            df_prep = df.copy()
            
            # Handle categorical columns
            if categorical_columns:
                for col in categorical_columns:
                    if col in df_prep.columns:
                        if col not in self.label_encoders:
                            self.label_encoders[col] = LabelEncoder()
                            df_prep[col] = self.label_encoders[col].fit_transform(df_prep[col].astype(str))
                        else:
                            df_prep[col] = self.label_encoders[col].transform(df_prep[col].astype(str))
            
            # Get features
            available_features = [col for col in feature_columns if col in df_prep.columns]
            X = df_prep[available_features].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Scale features
            X = self.scaler.fit_transform(X)
            
            # Get target if available
            y = None
            if target_column and target_column in df_prep.columns:
                y = df_prep[target_column].values
            
            logger.info(f"Prepared {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def train_classifier(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "classifier",
        algorithm: str = "random_forest",
        test_size: float = 0.2,
        hyperparameter_tuning: bool = False
    ) -> Dict:
        """
        Train classification model
        
        Args:
            X: Features
            y: Labels
            model_name: Model name
            algorithm: Algorithm (random_forest, logistic_regression)
            test_size: Test set ratio
            hyperparameter_tuning: Whether to tune hyperparameters
            
        Returns:
            Dict with training results
        """
        try:
            logger.info(f"Training classifier: {model_name} with {algorithm}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Select algorithm
            if algorithm == "random_forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                if hyperparameter_tuning:
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 20, None],
                        'min_samples_split': [2, 5, 10]
                    }
                    model = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
            elif algorithm == "logistic_regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                raise ValueError(f"Algorithm not supported: {algorithm}")
            
            # Training
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Cross validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            # Save model
            self.models[model_name] = model
            self._save_model(model, model_name)
            
            result = {
                'model_name': model_name,
                'algorithm': algorithm,
                'metrics': metrics,
                'feature_importance': self._get_feature_importance(model, algorithm),
                'training_date': datetime.now().isoformat()
            }
            
            logger.info(f"Completed training {model_name}: F1={metrics['f1_score']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training classifier: {str(e)}")
            raise
    
    def train_regressor(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "regressor",
        algorithm: str = "random_forest",
        test_size: float = 0.2
    ) -> Dict:
        """
        Train regression model
        
        Args:
            X: Features
            y: Target values
            model_name: Model name
            algorithm: Algorithm
            test_size: Test set ratio
            
        Returns:
            Dict with training results
        """
        try:
            logger.info(f"Training regressor: {model_name} with {algorithm}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Select algorithm
            if algorithm == "random_forest":
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif algorithm == "linear_regression":
                model = LinearRegression()
            else:
                raise ValueError(f"Algorithm not supported: {algorithm}")
            
            # Training
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            # Cross validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            metrics['cv_mean'] = cv_scores.mean()
            metrics['cv_std'] = cv_scores.std()
            
            # Save model
            self.models[model_name] = model
            self._save_model(model, model_name)
            
            result = {
                'model_name': model_name,
                'algorithm': algorithm,
                'metrics': metrics,
                'feature_importance': self._get_feature_importance(model, algorithm),
                'training_date': datetime.now().isoformat()
            }
            
            logger.info(f"Completed training {model_name}: R2={metrics['r2']:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training regressor: {str(e)}")
            raise
    
    def train_clustering(
        self,
        X: np.ndarray,
        model_name: str = "cluster",
        n_clusters: int = 5
    ) -> Dict:
        """
        Train clustering model
        
        Args:
            X: Features
            model_name: Model name
            n_clusters: Number of clusters
            
        Returns:
            Dict with training results
        """
        try:
            logger.info(f"Training clustering: {model_name} with {n_clusters} clusters")
            
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            
            # Training
            labels = model.fit_predict(X)
            
            # Metrics
            metrics = {
                'inertia': model.inertia_,
                'n_clusters': n_clusters,
                'n_samples': X.shape[0]
            }
            
            # Save model
            self.models[model_name] = model
            self._save_model(model, model_name)
            
            result = {
                'model_name': model_name,
                'algorithm': 'kmeans',
                'metrics': metrics,
                'cluster_centers': model.cluster_centers_.tolist(),
                'training_date': datetime.now().isoformat()
            }
            
            logger.info(f"Completed training {model_name}: {n_clusters} clusters")
            
            return result
            
        except Exception as e:
            logger.error(f"Error training clustering: {str(e)}")
            raise
    
    def _save_model(self, model: Any, model_name: str):
        """Save model to disk"""
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model: {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_name: str) -> Any:
        """Load model from disk"""
        try:
            model_path = self.model_dir / f"{model_name}.pkl"
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            self.models[model_name] = model
            logger.info(f"Loaded model: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _get_feature_importance(self, model: Any, algorithm: str) -> Optional[Dict]:
        """Get feature importance"""
        try:
            if algorithm in ['random_forest']:
                if hasattr(model, 'feature_importances_'):
                    return {'importances': model.feature_importances_.tolist()}
            return None
        except:
            return None
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Predict with trained model"""
        try:
            if model_name not in self.models:
                self.load_model(model_name)
            
            model = self.models[model_name]
            return model.predict(X)
            
        except Exception as e:
            logger.error(f"Error predicting: {str(e)}")
            raise