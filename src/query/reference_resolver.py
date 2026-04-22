"""
Reference Resolver Module
Resolves pronouns and implicit references using conversation context
"""

import logging
from typing import Dict, List, Optional, Any
from .context_manager import ConversationContext, QueryResult

logger = logging.getLogger(__name__)


class ReferenceResolver:
    """Resolves pronouns and references from conversation context"""
    
    # Pronoun mappings
    PRONOUN_GROUPS = {
        'they': ['they', 'them', 'their'],
        'that': ['that', 'those'],
        'this': ['this', 'these'],
        'it': ['it', 'its'],
        'he': ['he', 'him', 'his'],
        'she': ['she', 'her', 'hers']
    }
    
    def __init__(self):
        """Initialize reference resolver"""
        logger.info("ReferenceResolver initialized")
    
    def resolve(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Resolve references in query using context
        
        Args:
            query: User query with potential references
            context: Conversation context with history
            
        Returns:
            Dict with resolved references and updated query
        """
        resolved = {
            'original_query': query,
            'resolved_query': query,
            'references_found': [],
            'resolutions': {}
        }
        
        if not context.query_results:
            logger.debug("No previous query results to resolve references")
            return resolved
        
        # Get last query result
        last_result = context.get_last_query_result()
        if not last_result:
            return resolved
        
        # Detect and resolve pronouns
        pronouns_found = self._detect_pronouns(query)
        if pronouns_found:
            resolved['references_found'] = pronouns_found
            resolved['resolutions'] = self._resolve_pronouns(pronouns_found, last_result, context)
            resolved['resolved_query'] = self._apply_resolutions(query, resolved['resolutions'])
        
        return resolved
    
    def _detect_pronouns(self, query: str) -> List[str]:
        """Detect pronouns in query"""
        pronouns = []
        query_lower = query.lower()
        
        for pronoun_group, variants in self.PRONOUN_GROUPS.items():
            for variant in variants:
                if variant in query_lower:
                    if pronoun_group not in pronouns:
                        pronouns.append(pronoun_group)
        
        return pronouns
    
    def _resolve_pronouns(self, pronouns: List[str], last_result: QueryResult, context: ConversationContext) -> Dict[str, Any]:
        """Resolve pronouns to actual entities"""
        resolutions = {}
        
        for pronoun in pronouns:
            resolution = self._resolve_single_pronoun(pronoun, last_result, context)
            if resolution:
                resolutions[pronoun] = resolution
        
        return resolutions
    
    def _resolve_single_pronoun(self, pronoun: str, last_result: QueryResult, context: ConversationContext) -> Optional[Dict[str, Any]]:
        """Resolve a single pronoun"""
        
        # "They/them/those" - usually refers to list from previous query
        if pronoun in ['they', 'that', 'this']:
            if last_result.query_type == 'customer_analysis':
                if 'top_customers' in last_result.data:
                    customers = list(last_result.data['top_customers'].keys())
                    return {
                        'type': 'customer_list',
                        'values': customers,
                        'description': f"Top customers from previous query"
                    }
            elif last_result.query_type == 'product_analysis':
                if 'top_products' in last_result.data:
                    products = list(last_result.data['top_products'].keys())
                    return {
                        'type': 'product_list',
                        'values': products,
                        'description': f"Top products from previous query"
                    }
        
        # "It" - usually refers to a single metric or entity
        if pronoun == 'it':
            if last_result.summary:
                return {
                    'type': 'metric',
                    'value': last_result.summary,
                    'description': "Result from previous query"
                }
        
        return None
    
    def _apply_resolutions(self, query: str, resolutions: Dict[str, Any]) -> str:
        """Apply resolutions to query"""
        resolved_query = query
        
        for pronoun, resolution in resolutions.items():
            if resolution['type'] == 'customer_list':
                # Replace pronoun with customer list
                customer_str = ", ".join(resolution['values'][:3])  # First 3
                if len(resolution['values']) > 3:
                    customer_str += f", and {len(resolution['values']) - 3} more"
                resolved_query = resolved_query.replace(pronoun, f"({customer_str})")
            
            elif resolution['type'] == 'product_list':
                # Replace pronoun with product list
                product_str = ", ".join(resolution['values'][:3])  # First 3
                if len(resolution['values']) > 3:
                    product_str += f", and {len(resolution['values']) - 3} more"
                resolved_query = resolved_query.replace(pronoun, f"({product_str})")
        
        return resolved_query
    
    def resolve_implicit_references(self, query: str, context: ConversationContext) -> str:
        """
        Resolve implicit references like "that month", "those transactions"
        
        Args:
            query: User query
            context: Conversation context
            
        Returns:
            Resolved query with explicit references
        """
        resolved = query
        
        # "that month" -> specific month from context
        if 'that month' in query.lower():
            if 'month' in context.extracted_entities:
                month = context.extracted_entities['month']
                year = context.extracted_entities.get('year', '')
                resolved = resolved.replace('that month', f"{month}/{year}")
        
        # "that year" -> specific year from context
        if 'that year' in query.lower():
            if 'year' in context.extracted_entities:
                year = context.extracted_entities['year']
                resolved = resolved.replace('that year', f"{year}")
        
        # "those transactions" -> transactions from previous result
        if 'those transactions' in query.lower():
            last_result = context.get_last_query_result()
            if last_result and 'transaction_count' in last_result.data:
                count = last_result.data['transaction_count']
                resolved = resolved.replace('those transactions', f"the {count} transactions")
        
        return resolved


class ContextAwareQueryResolver:
    """Combines reference resolution with context awareness"""
    
    def __init__(self):
        """Initialize resolver"""
        self.reference_resolver = ReferenceResolver()
        logger.info("ContextAwareQueryResolver initialized")
    
    def resolve_query(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """
        Resolve query with full context awareness
        
        Args:
            query: User query
            context: Conversation context
            
        Returns:
            Dict with resolved query and context information
        """
        result = {
            'original_query': query,
            'resolved_query': query,
            'context_applied': False,
            'context_info': {}
        }
        
        # Step 1: Resolve pronouns
        pronoun_resolution = self.reference_resolver.resolve(query, context)
        result['resolved_query'] = pronoun_resolution['resolved_query']
        
        if pronoun_resolution['references_found']:
            result['context_applied'] = True
            result['context_info']['pronouns_resolved'] = pronoun_resolution['resolutions']
        
        # Step 2: Resolve implicit references
        implicit_resolved = self.reference_resolver.resolve_implicit_references(
            result['resolved_query'],
            context
        )
        if implicit_resolved != result['resolved_query']:
            result['resolved_query'] = implicit_resolved
            result['context_applied'] = True
            result['context_info']['implicit_references_resolved'] = True
        
        # Step 3: Apply active filters from context
        if context.active_filters:
            result['context_info']['active_filters'] = context.active_filters
            result['context_applied'] = True
        
        logger.debug(f"Resolved query: {result['resolved_query']}")
        return result
    
    def get_context_prompt(self, context: ConversationContext) -> str:
        """
        Generate a prompt snippet for AI with context information
        
        Args:
            context: Conversation context
            
        Returns:
            String to include in AI prompt
        """
        if not context.query_results:
            return ""
        
        prompt_parts = []
        prompt_parts.append("## Conversation Context")
        prompt_parts.append(f"Previous queries: {len(context.query_results)}")
        
        last_result = context.get_last_query_result()
        if last_result:
            prompt_parts.append(f"Last query type: {last_result.query_type}")
            prompt_parts.append(f"Last query: {last_result.query_text}")
            prompt_parts.append(f"Last result summary: {last_result.summary}")
        
        if context.active_filters:
            prompt_parts.append(f"Active filters: {context.active_filters}")
        
        if context.extracted_entities:
            prompt_parts.append(f"Extracted entities: {context.extracted_entities}")
        
        return "\n".join(prompt_parts)
