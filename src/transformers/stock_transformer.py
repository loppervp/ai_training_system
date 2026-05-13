"""
Stock Transformer Module
Module for transforming and cleaning stock/inventory data from stkm_main_all
"""

import json
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StockTransformer:
    """Class for transforming and cleaning stock/inventory data"""

    def __init__(self, mapping_path: str = "config/mapping.json"):
        self.mapping = self._load_mapping(mapping_path)
        self.transaction_type_mapping = self._get_transaction_type_mapping()

    def _load_mapping(self, mapping_path: str) -> Dict:
        """Read mapping file"""
        try:
            with open(mapping_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Mapping file not found: {mapping_path}")
            return {}

    def _get_transaction_type_mapping(self) -> Dict:
        """Get stock transaction type mapping from mapping.json"""
        try:
            return (
                self.mapping
                .get('database_tables', {})
                .get('stkm_main_all', {})
                .get('columns', {})
                .get('tag_table_usage', {})
                .get('mapping_values', {})
            )
        except Exception:
            return {
                'stk_do':    'Delivery Order',
                'stk_doc':   'Delivery Order Confirm',
                'stk_grn':   'Goods Receipt',
                'stk_gvn':   'Goods Valuation',
                'stk_adji':  'Stock Adjustment (Incr Qnty)',
                'stk_adjd':  'Stock Adjustment (Decr Qnty)',
                'stk_trnc':  'Transfer Confirmation',
                'stk_trnr':  'Transfer Request',
                'stk_retc':  'Sales Return',
                'stk_retv':  'Return To Vendor',
                'stk_asma':  'Work Order In Progress',
                'stk_asmc':  'Work Order Completion',
                'stk_asmr':  'Work Order Request',
                'stk_reclas': 'Item Reclassification With Cost',
                'stk_mi':    'Material Stock In',
                'stk_mw':    'Material Stock Out',
                'stk_open':  'Stock Opening Balance',
                'stk_take':  'Stock Take',
                'stk_unm':   'Reverse BOM / Unassemble',
            }

    # -------------------------------------------------------------------------
    # CLEAN RAW TABLES
    # -------------------------------------------------------------------------

    def clean_stock_main(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean stkm_main_all data

        Args:
            df: Raw DataFrame from stkm_main_all

        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"Starting to clean stkm_main_all: {len(df)} records")

            df_clean = df.copy()

            if 'companyfn' not in df_clean.columns:
                logger.warning("companyfn column not found - data isolation may not work correctly")

            # 1. Handle null values
            df_clean['party_desc']    = df_clean['party_desc'].fillna('Unknown')
            df_clean['party_code']    = df_clean['party_code'].fillna('N/A')
            df_clean['staff_code']    = df_clean['staff_code'].fillna('N/A')
            df_clean['location_code'] = df_clean['location_code'].fillna('N/A')
            df_clean['stkcode_code']  = df_clean['stkcode_code'].fillna('N/A')
            df_clean['curr_short_forex'] = df_clean['curr_short_forex'].fillna('N/A') if 'curr_short_forex' in df_clean.columns else df_clean.get('curr_short_forex', 'N/A')

            # 2. Convert data types
            df_clean['date_trans'] = pd.to_datetime(df_clean['date_trans'], errors='coerce')

            df_clean['amount_local']              = pd.to_numeric(df_clean['amount_local'],              errors='coerce').fillna(0)
            df_clean['amount_forex']              = pd.to_numeric(df_clean['amount_forex'],              errors='coerce').fillna(0)
            df_clean['balance_qnty_uom_stk_code'] = pd.to_numeric(df_clean['balance_qnty_uom_stk_code'], errors='coerce').fillna(0)

            # 3. Add transaction_type_name column
            df_clean['transaction_type_name'] = df_clean['tag_table_usage'].map(self.transaction_type_mapping).fillna('Unknown')

            # 4. Create product identifier
            df_clean['stkcode_code']   = df_clean['stkcode_code'].astype(str)
            df_clean['stkcode_unique'] = df_clean['stkcode_unique'].astype(str).fillna('0')
            df_clean['product_id']     = df_clean['stkcode_code'] + '_' + df_clean['stkcode_unique']

            # 5. Add temporal features
            df_clean['year']         = df_clean['date_trans'].dt.year
            df_clean['month']        = df_clean['date_trans'].dt.month
            df_clean['quarter']      = df_clean['date_trans'].dt.quarter
            df_clean['day_of_week']  = df_clean['date_trans'].dt.dayofweek
            df_clean['day_of_month'] = df_clean['date_trans'].dt.day
            df_clean['week_of_year'] = df_clean['date_trans'].dt.isocalendar().week
            df_clean['is_weekend']   = df_clean['day_of_week'].isin([5, 6]).astype(int)
            df_clean['is_month_end'] = (df_clean['day_of_month'] >= 25).astype(int)

            # 6. Calculate metrics
            df_clean['amount_local_log'] = np.log1p(df_clean['amount_local'].abs())

            # 7. Flag outliers
            q99 = df_clean['amount_local'].quantile(0.99)
            df_clean['is_outlier'] = (df_clean['amount_local'].abs() > q99).astype(int)

            logger.info(f"Completed cleaning stkm_main_all: {len(df_clean)} records")
            return df_clean

        except Exception as e:
            logger.error(f"Error cleaning stkm_main_all: {str(e)}")
            raise