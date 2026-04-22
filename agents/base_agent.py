"""
Base Agent Class
Core interface for all agents in the development team
"""

import os
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AgentTask(BaseModel):
    """Task model for agent execution"""
    task_id: str = Field(description="Unique task identifier")
    task_type: str = Field(description="Type of task: analyze, code, test, review, plan")
    description: str = Field(description="Task description")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input parameters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    priority: int = Field(default=3, description="Task priority 1=highest 5=lowest")


class AgentResult(BaseModel):
    """Result model returned from agent execution"""
    task_id: str
    success: bool
    output: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    execution_time: float = 0.0
    next_actions: List[AgentTask] = Field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, role: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.role = role
        self.config = config or {}
        self.working_directory = os.getcwd()
        self.memory = []
        self.capabilities = []
        
        logger.info(f"🤖 Agent initialized: {self.name} ({self.role})")
    
    @abstractmethod
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute assigned task"""
        pass
    
    def add_capability(self, capability: str):
        """Add capability to agent"""
        if capability not in self.capabilities:
            self.capabilities.append(capability)
    
    def can_handle(self, task: AgentTask) -> bool:
        """Check if agent can handle this task type"""
        return task.task_type in self.capabilities
    
    def log_action(self, message: str):
        """Log agent action"""
        logger.info(f"[{self.name}] {message}")
    
    def create_success_result(self, task: AgentTask, output: Dict[str, Any]) -> AgentResult:
        """Create success result object"""
        return AgentResult(
            task_id=task.task_id,
            success=True,
            output=output
        )
    
    def create_error_result(self, task: AgentTask, errors: List[str]) -> AgentResult:
        """Create error result object"""
        return AgentResult(
            task_id=task.task_id,
            success=False,
            errors=errors
        )
    
    def __repr__(self) -> str:
        return f"<Agent name={self.name} role={self.role} capabilities={len(self.capabilities)}>"