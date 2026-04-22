"""
Agent Orchestrator
Task manager that coordinates work between different agents in the team
"""

import time
import uuid
import logging
from typing import Dict, List, Any, Optional
from queue import PriorityQueue
from threading import Thread
from .base_agent import BaseAgent, AgentTask, AgentResult

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """Orchestrates the entire agent team workflow"""
    
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.task_queue = PriorityQueue()
        self.completed_tasks: Dict[str, AgentResult] = {}
        self.running = False
        self.worker_thread: Optional[Thread] = None
        
        logger.info("🏛️ Agent Orchestrator initialized")
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent to the team"""
        self.agents.append(agent)
        logger.info(f"✅ Registered agent: {agent.name} ({agent.role})")
    
    def submit_task(self, task: AgentTask) -> str:
        """Submit a task to the orchestrator"""
        # Add to priority queue (priority, task)
        self.task_queue.put((task.priority, task))
        logger.info(f"📋 Task submitted: {task.task_id} - {task.description[:60]}...")
        return task.task_id
    
    def start(self):
        """Start the orchestrator worker thread"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("🚀 Agent Orchestrator started")
    
    def stop(self):
        """Stop the orchestrator"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("🛑 Agent Orchestrator stopped")
    
    def _worker_loop(self):
        """Main worker loop processing tasks"""
        while self.running:
            if self.task_queue.empty():
                time.sleep(0.5)
                continue
            
            # Get highest priority task
            priority, task = self.task_queue.get()
            
            try:
                # Find suitable agent
                suitable_agent = self._find_suitable_agent(task)
                
                if not suitable_agent:
                    logger.warning(f"⚠️ No agent found for task: {task.task_id}")
                    continue
                
                # Execute task
                logger.info(f"⚡ Assigning task {task.task_id} to {suitable_agent.name}")
                start_time = time.time()
                
                result = suitable_agent.execute(task)
                result.execution_time = time.time() - start_time
                
                # Store result
                self.completed_tasks[task.task_id] = result
                
                if result.success:
                    logger.info(f"✅ Task {task.task_id} completed successfully in {result.execution_time:.2f}s")
                    
                    # Submit next actions
                    for next_task in result.next_actions:
                        self.submit_task(next_task)
                else:
                    logger.error(f"❌ Task {task.task_id} failed: {result.errors}")
                
            except Exception as e:
                logger.error(f"🔥 Error processing task {task.task_id}: {str(e)}", exc_info=True)
            
            finally:
                self.task_queue.task_done()
    
    def _find_suitable_agent(self, task: AgentTask) -> Optional[BaseAgent]:
        """Find the best agent for this task"""
        for agent in self.agents:
            if agent.can_handle(task):
                return agent
        return None
    
    def process_markdown_file(self, file_path: str) -> str:
        """Convenience method: Analyze a markdown file and run all tasks"""
        task_id = str(uuid.uuid4())[:8]
        
        task = AgentTask(
            task_id=task_id,
            task_type="analyze",
            description=f"Analyze markdown file: {file_path}",
            input_data={'file_path': file_path},
            priority=1
        )
        
        self.submit_task(task)
        return task_id
    
    def wait_completion(self):
        """Wait for all tasks to complete"""
        self.task_queue.join()
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            'agents': len(self.agents),
            'queue_size': self.task_queue.qsize(),
            'completed_tasks': len(self.completed_tasks),
            'running': self.running
        }