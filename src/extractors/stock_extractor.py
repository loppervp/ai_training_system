"""
Stock Extractor Module
Specialized module for extracting stock/inventory data from stkm_main_all
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from .database_extractor import DatabaseExtractor

logger = logging.getLogger(__name__)


class StockExtractor:
    """Class for extracting stock/inventory data with company-based data isolation"""

    def __init__(self, config_path: str = "config/database.json", companyfn: Optional[str] = None):
        """
        Initialize StockExtractor

        Args:
            config_path: Path to configuration file
            companyfn: Company code for data isolation (unique per company)
        """
        self.db_extractor = DatabaseExtractor(config_path)
        self.mapping = self._load_mapping()
        self.companyfn = companyfn

    def _load_mapping(self) -> Dict:
        """Read mapping file"""
        try:
            with open("config/mapping.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning("Mapping file mapping.json not found")
            return {}

    def extract_stock_main(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        transaction_types: Optional[List[str]] = None,
        include_void: bool = False
    ) -> pd.DataFrame:
        """
        Extract data from stkm_main_all table with company-based data isolation

        Args:
            companyfn: Company code (uses instance companyfn if not provided)
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            transaction_types: Transaction types (stk_doc, stk_do, stk_gvn, etc.)
            include_void: Include voided transactions

        Returns:
            DataFrame containing data filtered by companyfn
        """
        try:
            effective_companyfn = companyfn or self.companyfn

            query = """
                SELECT
                    uniquenum_pri,
                    companyfn,
                    tag_table_usage,
                    date_trans,
                    location_code,
                    stkcode_code,
                    stkcode_unique,
                    balance_qnty_uom_stk_code,
                    party_unique,
                    party_code,
                    party_desc,
                    staff_code,
                    staff_unique,
                    tag_void_yn,
                    amount_local,
                    amount_forex,
                    curr_short_forex
                FROM stkm_main_all
                WHERE 1=1
            """

            params = {}

            if effective_companyfn:
                query += " AND companyfn = :companyfn"
                params['companyfn'] = effective_companyfn

            if not include_void:
                query += " AND tag_void_yn = 'n'"

            if transaction_types:
                placeholders = ", ".join([f":type_{i}" for i in range(len(transaction_types))])
                query += f" AND tag_table_usage IN ({placeholders})"
                for i, t in enumerate(transaction_types):
                    params[f"type_{i}"] = t

            if date_from:
                query += " AND date_trans >= :date_from"
                params['date_from'] = date_from

            if date_to:
                query += " AND date_trans <= :date_to"
                params['date_to'] = date_to

            query += " ORDER BY date_trans DESC"

            logger.info(f"Extracting stkm_main_all | params: {params}")
            result = self.db_extractor.extract_data(query, params)
            logger.info(f"Result: {len(result)} records")
            return result

        except Exception as e:
            logger.error(f"Error extracting stkm_main_all: {str(e)}")
            raise



    def close(self):
        """Close connection"""
        self.db_extractor.close()
