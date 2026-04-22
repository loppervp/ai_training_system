"""
Conversation Context Manager
Manages conversation history, query results, and entity extraction for follow-up questions
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Represents a single query result"""
    query_id: str
    query_text: str
    query_type: str
    timestamp: str
    data: Dict[str, Any]
    summary: str
    insights: List[str] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class ConversationContext:
    """Manages conversation state and history"""
    conversation_id: str
    companyfn: Optional[str] = None
    messages: List[Dict] = field(default_factory=list)
    query_results: List[QueryResult] = field(default_factory=list)
    active_filters: Dict[str, Any] = field(default_factory=dict)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.messages.append(message)
        logger.debug(f"Added {role} message: {content[:50]}...")
    
    def add_query_result(self, result: QueryResult):
        """Add query result to history"""
        self.query_results.append(result)
        
        # Update extracted entities
        if result.entities:
            self.extracted_entities.update(result.entities)
        
        logger.debug(f"Added query result: {result.query_type}")
    
    def get_last_query_result(self) -> Optional[QueryResult]:
        """Get the most recent query result"""
        return self.query_results[-1] if self.query_results else None
    
    def get_last_query_data(self) -> Optional[Dict]:
        """Get data from last query"""
        last = self.get_last_query_result()
        return last.data if last else None
    
    def set_filter(self, filter_name: str, filter_value: Any):
        """Set an active filter"""
        self.active_filters[filter_name] = filter_value
        logger.debug(f"Set filter: {filter_name} = {filter_value}")
    
    def get_filter(self, filter_name: str) -> Optional[Any]:
        """Get an active filter"""
        return self.active_filters.get(filter_name)
    
    def clear_filters(self):
        """Clear all active filters"""
        self.active_filters.clear()
        logger.debug("Cleared all filters")
    
    def get_context_summary(self) -> str:
        """Get a summary of current context for AI"""
        summary = []
        
        if self.query_results:
            summary.append(f"Previous queries: {len(self.query_results)}")
            last = self.get_last_query_result()
            if last:
                summary.append(f"Last query type: {last.query_type}")
                summary.append(f"Last query: {last.query_text}")
        
        if self.active_filters:
            summary.append(f"Active filters: {self.active_filters}")
        
        if self.extracted_entities:
            summary.append(f"Extracted entities: {self.extracted_entities}")
        
        return "\n".join(summary) if summary else "No previous context"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'conversation_id': self.conversation_id,
            'companyfn': self.companyfn,
            'messages': self.messages,
            'query_results': [r.to_dict() for r in self.query_results],
            'active_filters': self.active_filters,
            'extracted_entities': self.extracted_entities,
            'created_at': self.created_at
        }


class ContextManager:
    """Manages multiple conversations"""
    
    def __init__(self):
        """Initialize context manager"""
        self.conversations: Dict[str, ConversationContext] = {}
        logger.info("ContextManager initialized")
    
    def create_conversation(self, conversation_id: str, companyfn: Optional[str] = None) -> ConversationContext:
        """Create a new conversation"""
        context = ConversationContext(
            conversation_id=conversation_id,
            companyfn=companyfn
        )
        self.conversations[conversation_id] = context
        logger.info(f"Created conversation: {conversation_id}")
        return context
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get existing conversation"""
        return self.conversations.get(conversation_id)
    
    def get_or_create_conversation(self, conversation_id: str, companyfn: Optional[str] = None) -> ConversationContext:
        """Get existing conversation or create new one"""
        if conversation_id not in self.conversations:
            return self.create_conversation(conversation_id, companyfn)
        return self.conversations[conversation_id]
    
    def delete_conversation(self, conversation_id: str):
        """Delete a conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id}")
    
    def clear_all(self):
        """Clear all conversations"""
        self.conversations.clear()
        logger.info("Cleared all conversations")
    
    def get_conversation_count(self) -> int:
        """Get number of active conversations"""
        return len(self.conversations)


# Global context manager instance
_context_manager = ContextManager()


def get_context_manager() -> ContextManager:
    """Get global context manager instance"""
    return _context_manager
