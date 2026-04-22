"""
Markdown Analysis Agent
Specialized agent that reads and analyzes .md files, extracts requirements, tasks, test cases
"""

import re
import os
from typing import Dict, List, Any
from markdown import markdown
from bs4 import BeautifulSoup
from .base_agent import BaseAgent, AgentTask, AgentResult


class MarkdownAnalysisAgent(BaseAgent):
    """Agent that reads markdown files and extracts actionable tasks"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="Markdown Analyst",
            role="Requirements Engineer",
            config=config
        )
        self.add_capability("analyze")
        self.add_capability("parse_markdown")
    
    def execute(self, task: AgentTask) -> AgentResult:
        """Execute markdown analysis task"""
        self.log_action(f"Analyzing markdown file: {task.input_data.get('file_path')}")
        
        file_path = task.input_data.get('file_path')
        
        if not file_path or not os.path.exists(file_path):
            return self.create_error_result(task, [f"File not found: {file_path}"])
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract all information
            analysis = self._analyze_content(content)
            analysis['file_path'] = file_path
            analysis['file_size'] = os.path.getsize(file_path)
            
            # Generate sub tasks
            tasks = self._generate_tasks(analysis, task)
            
            result = self.create_success_result(task, analysis)
            result.next_actions = tasks
            
            self.log_action(f"✅ Markdown analysis complete. Found {len(tasks)} actionable tasks.")
            
            return result
            
        except Exception as e:
            return self.create_error_result(task, [str(e)])
    
    def _analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze markdown content and extract structure"""
        
        # Extract headers
        headers = self._extract_headers(content)
        
        # Extract checklists and todos
        checklists = self._extract_checklists(content)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)
        
        # Extract links and references
        links = self._extract_links(content)
        
        # Extract requirements
        requirements = self._extract_requirements(content)
        
        return {
            'headers': headers,
            'checklists': checklists,
            'code_blocks': code_blocks,
            'links': links,
            'requirements': requirements,
            'word_count': len(content.split()),
            'line_count': len(content.splitlines())
        }
    
    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """Extract all headers from markdown"""
        headers = []
        for match in re.finditer(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE):
            headers.append({
                'level': len(match.group(1)),
                'text': match.group(2).strip(),
                'position': match.start()
            })
        return headers
    
    def _extract_checklists(self, content: str) -> List[Dict[str, Any]]:
        """Extract todo checklist items"""
        checklists = []
        
        # Match MARKDOWN checkboxes (support BOTH | and normal lists)
        for match in re.finditer(r'^(?:\|[^|]+\|\s*)*(\s*)-\s+\[([ xX])\]\s+(.+)$', content, re.MULTILINE):
            checklists.append({
                'indent': len(match.group(1)),
                'completed': match.group(2).lower() == 'x',
                'text': match.group(3).strip()
            })
        
        # Also match table checkboxes
        for match in re.finditer(r'\|\s*(\d+)\s*\|\s*([^\|]+?)\s*\|\s*([^\|]+?)\s*\|\s*(⏳|✅|⌛|❌)\s*\|', content, re.MULTILINE):
            if match:
                status = match.group(4).strip()
                checklists.append({
                    'indent': 0,
                    'completed': status == '✅',
                    'text': match.group(2).strip(),
                    'estimated_time': match.group(3).strip(),
                    'status': status
                })
        
        # ALSO EXTRACT TASKS FROM ALL MARKDOWN HEADERS
        header_pattern = re.compile(r'^#{1,4}\s*(.+?)$', re.MULTILINE)
        for match in header_pattern.finditer(content):
            header_text = match.group(1).strip()
            if len(header_text) > 10 and not header_text.lower().startswith(('mục tiêu', 'roadmap', 'lợi ích', 'gợi ý')):
                # Check if this header has incomplete items below it
                checklists.append({
                    'indent': 0,
                    'completed': False,
                    'text': header_text,
                    'source': 'header'
                })
        
        return checklists
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks"""
        code_blocks = []
        
        for match in re.finditer(r'```(\w*)\n([\s\S]*?)\n```', content):
            code_blocks.append({
                'language': match.group(1),
                'code': match.group(2).strip(),
                'line_count': len(match.group(2).splitlines())
            })
        
        return code_blocks
    
    def _extract_links(self, content: str) -> List[Dict[str, Any]]:
        """Extract markdown links"""
        links = []
        
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', content):
            links.append({
                'text': match.group(1),
                'url': match.group(2)
            })
        
        return links
    
    def _extract_requirements(self, content: str) -> List[str]:
        """Extract requirements and action items"""
        requirements = []
        
        # Look for requirement patterns
        requirement_patterns = [
            r'^(?:✅|❌|⚠️|🔹|▪️|->|-)\s*(?:Cần|Phải|Nên|Cần phải|Phải có)\s+(.+)$',
            r'^.*?(?:phải|cần|should|must|required)\s+(.+?)(?:\.|$)',
        ]
        
        for pattern in requirement_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE):
                req = match.group(1).strip()
                if len(req) > 10 and req not in requirements:
                    requirements.append(req)
        
        return requirements
    
    def _generate_tasks(self, analysis: Dict[str, Any], parent_task: AgentTask) -> List[AgentTask]:
        """Generate actionable tasks from analysis"""
        tasks = []
        task_id_base = parent_task.task_id
        
        # Create tasks from checklists
        for idx, item in enumerate(analysis['checklists']):
            if not item['completed']:
                tasks.append(AgentTask(
                    task_id=f"{task_id_base}-task-{idx}",
                    task_type="execute",
                    description=item['text'],
                    priority=parent_task.priority,
                    context={'source': 'checklist_item'}
                ))
        
        return tasks