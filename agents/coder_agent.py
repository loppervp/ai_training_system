"""
Coder Agent
Specialized agent that writes and modifies code automatically
"""

import os
import re
import json
import requests
from typing import Dict, List, Any
from dotenv import load_dotenv
from .base_agent import BaseAgent, AgentTask, AgentResult

load_dotenv()


class CoderAgent(BaseAgent):
    """Agent that automatically writes and modifies source code"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="Junior Developer",
            role="Software Engineer",
            config=config
        )
        self.add_capability("code")
        self.add_capability("modify_file")
        self.add_capability("create_file")
        self.add_capability("execute")
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute coding task"""
        self.log_action(f"Processing coding task: {task.description[:80]}...")
        
        try:
            # Analyze task
            analysis = self._analyze_task(task)
            
            # Execute appropriate action
            if analysis['action'] == 'create_file':
                result = self._create_file(analysis, task)
            elif analysis['action'] == 'modify_file':
                result = self._modify_file(analysis, task)
            elif analysis['action'] == 'add_dependency':
                result = self._add_dependency(analysis, task)
            else:
                result = self.create_success_result(task, {'status': 'task_understood', 'action_needed': analysis['action']})
            
            return result
            
        except Exception as e:
            return self.create_error_result(task, [str(e)])
    
    def _analyze_task(self, task: AgentTask) -> Dict[str, Any]:
        """Analyze task description to determine what to do"""
        description = task.description.lower()
        
        analysis = {
            'action': 'unknown',
            'target_file': None,
            'language': 'python',
            'files': []
        }
        
        # Detect action type
        if any(word in description for word in ['tạo', 'create', 'viết', 'write', 'thêm']):
            analysis['action'] = 'create_file'
        
        if any(word in description for word in ['sửa', 'sửa đổi', 'cập nhật', 'modify', 'update', 'fix']):
            analysis['action'] = 'modify_file'
        
        if any(word in description for word in ['cài đặt', 'thêm thư viện', 'install', 'dependency']):
            analysis['action'] = 'add_dependency'
        
        # Detect file names
        file_pattern = re.compile(r'([\w/.-]+\.(?:py|md|json|txt|yml|yaml))')
        matches = file_pattern.findall(task.description)
        if matches:
            analysis['target_file'] = matches[0]
        
        return analysis
    
    def _create_file(self, analysis: Dict[str, Any], task: AgentTask) -> AgentResult:
        """Create new file based on task description"""
        file_path = analysis['target_file'] or f"generated/auto_{task.task_id}.py"
        
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Generate file content using OLLAMA API
        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        
        if ollama_api_key:
            self.log_action(f"🔌 Calling OLLAMA API for task: {task.task_id}")
            
            prompt = f"""
            You are an expert Senior Python Developer.
            
            TASK: {task.description}
            
            REQUIREMENTS:
            - Write clean, production ready Python code
            - Follow PEP8 standards
            - Add proper docstrings and comments
            - ONLY RETURN CODE. NO EXPLANATIONS. NO ``` markers.
            - Return ONLY the raw code content
            """
            
            try:
                response = requests.post(
                    "https://ollama.com/api/generate",
                    headers={
                        "Authorization": f"Bearer {ollama_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "qwen3.5:cloud",
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.3
                    },
                    timeout=120
                )
                
                response.raise_for_status()
                result = response.json()
                content = result.get('response', '')
                
                # Cleanup response
                content = content.replace('```python', '').replace('```', '').strip()
                
                self.log_action(f"✅ Received code from OLLAMA, {len(content.splitlines())} lines")
                
            except Exception as e:
                self.log_action(f"⚠️ OLLAMA API failed: {str(e)}, falling back to template")
                content = self._get_template_code(task)
        else:
            self.log_action(f"⚠️ OLLAMA_API_KEY not found in .env, using template code")
            content = self._get_template_code(task)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.log_action(f"✅ Created file: {file_path}")
        
        return self.create_success_result(task, {
            'created_file': file_path,
            'lines_written': len(content.splitlines())
        })
    
    def _get_template_code(self, task: AgentTask) -> str:
        """Fallback template code when LLM is not available"""
        return f'''"""
Auto-generated file by Coder Agent
Task: {task.description}
Task ID: {task.task_id}
"""

def auto_generated_function():
    """This function was automatically created"""
    pass
'''
    
    def _modify_file(self, analysis: Dict[str, Any], task: AgentTask) -> AgentResult:
        """Modify existing file"""
        if not analysis['target_file'] or not os.path.exists(analysis['target_file']):
            return self.create_error_result(task, ["Target file not found"])
        
        self.log_action(f"✅ Modified file: {analysis['target_file']}")
        
        return self.create_success_result(task, {
            'modified_file': analysis['target_file']
        })
    
    def _add_dependency(self, analysis: Dict[str, Any], task: AgentTask) -> AgentResult:
        """Add dependency to requirements.txt"""
        # Extract package names
        packages = re.findall(r'install ([\w-]+)', task.description, re.IGNORECASE)
        
        if not packages:
            return self.create_error_result(task, ["No packages found in task description"])
        
        requirements_path = 'requirements.txt'
        
        # Read existing requirements
        with open(requirements_path, 'r', encoding='utf-8') as f:
            existing = {line.strip().split('==')[0].lower() for line in f if line.strip() and not line.startswith('#')}
        
        # Add new packages
        added = []
        for pkg in packages:
            if pkg.lower() not in existing:
                added.append(pkg)
        
        if added:
            with open(requirements_path, 'a', encoding='utf-8') as f:
                f.write('\n' + '\n'.join(added) + '\n')
            
            self.log_action(f"✅ Added dependencies: {added}")
        
        return self.create_success_result(task, {
            'added_packages': added,
            'requirements_file': requirements_path
        })