"""
MLflow Tracker Module
Experiment tracking and model version management
"""
import os
import logging
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class MLFlowTracker:
    """
    MLflow experiment tracker for model version management,
    metrics logging, artifact storage and experiment comparison
    """
    
    def __init__(
        self,
        experiment_name: str = "ai_training_system",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: MLflow server URI (default: ./mlruns local)
        """
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
        self.experiment_name = experiment_name
        self.run = None
        self.run_id = None
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create experiment if not exists
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"Created new MLflow experiment: {self.experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {self.experiment_name}")
        
        mlflow.set_experiment(experiment_id=self.experiment_id)
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
        self.run_id = self.run.info.run_id
        logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")
        
        return self.run
    
    def log_params(self, params: Dict[str, Any]):
        """Log training parameters"""
        if self.run:
            mlflow.log_params(params)
            logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log evaluation metrics"""
        if self.run:
            mlflow.log_metrics(metrics, step=step)
            logger.info(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log file artifact"""
        if self.run:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
    
    def log_model(
        self,
        model,
        model_name: str,
        registered_model_name: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Log and register scikit-learn model"""
        if self.run:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name,
                registered_model_name=registered_model_name
            )
            
            if metrics:
                for metric_name, value in metrics.items():
                    mlflow.set_tag(f"metric_{metric_name}", f"{value:.6f}")
            
            logger.info(f"✅ Model logged: {model_name}")
            if registered_model_name:
                logger.info(f"✅ Model registered: {registered_model_name}")
    
    def end_run(self, status: str = "FINISHED"):
        """End current MLflow run"""
        if self.run:
            mlflow.end_run(status=status)
            logger.info(f"Ended MLflow run with status: {status}")
            self.run = None
            self.run_id = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.end_run("FINISHED")
        else:
            logger.error(f"Run failed with exception: {exc_val}")
            self.end_run("FAILED")