"""
Purchase Transformer Module
Module for transforming and cleaning purchase data from scm_pur_main and scm_pur_data
"""

import json
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class PurchaseTransformer:
    """Class for transforming and cleaning purchase data"""

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
        """Get purchase transaction type mapping from mapping.json"""
        try:
            return (
                self.mapping
                .get('database_tables', {})
                .get('scm_pur_main', {})
                .get('columns', {})
                .get('tag_table_usage', {})
                .get('mapping_values', {})
            )
        except Exception:
            return {
                'pur_po':  'Purchase Order',
                'pur_poc': 'Purchase Order Confirm',
                'pur_inv': 'Purchase Invoice',
                'pur_pr': 'Purchase Requisition',
                'pur_dn': 'Purchase Debit Note',
                'pur_cn': 'Purchase Credit Note',
            }

    # -------------------------------------------------------------------------
    # CLEAN RAW TABLES
    # -------------------------------------------------------------------------

    def clean_purchase_main(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean scm_pur_main data

        Args:
            df: Raw DataFrame from scm_pur_main

        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"Starting to clean scm_pur_main: {len(df)} records")

            df_clean = df.copy()

            if 'companyfn' not in df_clean.columns:
                logger.warning("companyfn column not found - data isolation may not work correctly")

            # 1. Handle null values
            df_clean['party_desc']     = df_clean['party_desc'].fillna('Unknown')
            df_clean['party_code']     = df_clean['party_code'].fillna('N/A')
            df_clean['deptunit_desc']  = df_clean['deptunit_desc'].fillna('N/A')
            df_clean['staff_code']     = df_clean['staff_code'].fillna('N/A')
            df_clean['notes_memo']     = df_clean['notes_memo'].fillna('')
            df_clean['location_code']  = df_clean['location_code'].fillna('N/A')
            df_clean['creditterm_desc'] = df_clean['creditterm_desc'].fillna('N/A') if 'creditterm_desc' in df_clean.columns else df_clean.get('creditterm_desc', 'N/A')

            # 2. Convert data types
            df_clean['date_trans'] = pd.to_datetime(df_clean['date_trans'], errors='coerce')
            df_clean['date_due']   = pd.to_datetime(df_clean['date_due'],   errors='coerce') if 'date_due' in df_clean.columns else None

            df_clean['amount_local']           = pd.to_numeric(df_clean['amount_local'],           errors='coerce').fillna(0)
            df_clean['amount_forex']           = pd.to_numeric(df_clean['amount_forex'],           errors='coerce').fillna(0)
            df_clean['curr_rate_forex_f_calc'] = pd.to_numeric(df_clean.get('curr_rate_forex_f_calc', 1), errors='coerce').fillna(1)

            # 3. Add transaction_type_name column
            df_clean['transaction_type_name'] = df_clean['tag_table_usage'].map(self.transaction_type_mapping).fillna('Unknown')

            # 4. Add temporal features
            df_clean['year']         = df_clean['date_trans'].dt.year
            df_clean['month']        = df_clean['date_trans'].dt.month
            df_clean['quarter']      = df_clean['date_trans'].dt.quarter
            df_clean['day_of_week']  = df_clean['date_trans'].dt.dayofweek
            df_clean['day_of_month'] = df_clean['date_trans'].dt.day
            df_clean['week_of_year'] = df_clean['date_trans'].dt.isocalendar().week
            df_clean['is_weekend']   = df_clean['day_of_week'].isin([5, 6]).astype(int)
            df_clean['is_month_end'] = (df_clean['day_of_month'] >= 25).astype(int)

            # 5. Calculate metrics
            df_clean['amount_local_log'] = np.log1p(df_clean['amount_local'].abs())

            # 6. Flag outliers
            q99 = df_clean['amount_local'].quantile(0.99)
            df_clean['is_outlier'] = (df_clean['amount_local'].abs() > q99).astype(int)

            logger.info(f"Completed cleaning scm_pur_main: {len(df_clean)} records")
            return df_clean

        except Exception as e:
            logger.error(f"Error cleaning scm_pur_main: {str(e)}")
            raise

    def clean_purchase_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean scm_pur_data data

        Args:
            df: Raw DataFrame from scm_pur_data

        Returns:
            Cleaned DataFrame
        """
        try:
            logger.info(f"Starting to clean scm_pur_data: {len(df)} records")

            df_clean = df.copy()

            # 1. Handle null values
            df_clean['stkcode_desc']  = df_clean['stkcode_desc'].fillna('Unknown Product')
            df_clean['stkcode_code']  = df_clean['stkcode_code'].fillna('N/A')
            df_clean['brand_desc']    = df_clean['brand_desc'].fillna('No Brand')
            df_clean['stkcate_desc']  = df_clean['stkcate_desc'].fillna('No Category')
            df_clean['stkvendor_desc'] = df_clean['stkvendor_desc'].fillna('No Vendor')
            df_clean['notes_memo']    = df_clean['notes_memo'].fillna('')

            # 2. Convert data types
            df_clean['date_trans'] = pd.to_datetime(df_clean['date_trans'], errors='coerce')

            numeric_cols = [
                'qnty_total', 'qnty_uomstk', 'bal_qnty_total', 'bal_qnty_uomstk',
                'price_unitlist_forex', 'price_unitlist_local',
                'price_unitrate_forex', 'price_unitrate_local',
                'discount_pct', 'amount_forex', 'amount_local',
                'amount_tax_forex', 'amount_tax_local'
            ]
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

            # 3. Calculate metrics
            df_clean['line_total']       = df_clean['qnty_total'] * df_clean['price_unitrate_local']
            df_clean['discount_amount']  = df_clean['line_total'] * df_clean['discount_pct'] / 100
            df_clean['net_amount']       = df_clean['line_total'] - df_clean['discount_amount']

            # 4. Add temporal features
            df_clean['year']    = df_clean['date_trans'].dt.year
            df_clean['month']   = df_clean['date_trans'].dt.month
            df_clean['quarter'] = df_clean['date_trans'].dt.quarter

            # 5. Create product identifier
            df_clean['stkcode_code']   = df_clean['stkcode_code'].astype(str).fillna('N/A')
            df_clean['stkcode_unique'] = df_clean['stkcode_unique'].astype(str).fillna('0')
            df_clean['product_id']     = df_clean['stkcode_code'] + '_' + df_clean['stkcode_unique']

            logger.info(f"Completed cleaning scm_pur_data: {len(df_clean)} records")
            return df_clean

        except Exception as e:
            logger.error(f"Error cleaning scm_pur_data: {str(e)}")
            raise

    # -------------------------------------------------------------------------
    # TRANSFORM AI DATASETS
    # -------------------------------------------------------------------------

    def transform_vendor_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform vendor analysis data
        """
        try:
            logger.info("Transforming vendor analysis data")

            df_clean = df.copy()

            df_clean['vendor_name']   = df_clean['vendor_name'].fillna('Unknown')
            df_clean['product_name']  = df_clean['product_name'].fillna('Unknown')
            df_clean['category_name'] = df_clean['category_name'].fillna('No Category')
            df_clean['brand_desc']    = df_clean['brand_desc'].fillna('No Brand')

            df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'], errors='coerce')
            df_clean['quantity']   = pd.to_numeric(df_clean['quantity'],   errors='coerce').fillna(0)
            df_clean['unit_price'] = pd.to_numeric(df_clean['unit_price'], errors='coerce').fillna(0)
            df_clean['line_amount'] = pd.to_numeric(df_clean['line_amount'], errors='coerce').fillna(0)

            df_clean['year']        = df_clean['transaction_date'].dt.year
            df_clean['month']       = df_clean['transaction_date'].dt.month
            df_clean['quarter']     = df_clean['transaction_date'].dt.quarter
            df_clean['day_of_week'] = df_clean['transaction_date'].dt.dayofweek
            df_clean['is_weekend']  = df_clean['day_of_week'].isin([5, 6]).astype(int)

            df_clean['transaction_value_segment'] = pd.qcut(
                df_clean['line_amount'].abs(),
                q=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )

            return df_clean

        except Exception as e:
            logger.error(f"Error transforming vendor analysis data: {str(e)}")
            raise

    def transform_purchase_product_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform product analysis for purchase
        """
        try:
            logger.info("Transforming purchase product analysis data")

            df_clean = df.copy()

            df_clean['product_name']  = df_clean['product_name'].fillna('Unknown')
            df_clean['category_name'] = df_clean['category_name'].fillna('No Category')
            df_clean['brand_desc']    = df_clean['brand_desc'].fillna('No Brand')
            df_clean['vendor_name']   = df_clean['vendor_name'].fillna('No Vendor')

            df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'], errors='coerce')
            df_clean['quantity_purchased'] = pd.to_numeric(df_clean['quantity_sold'],  errors='coerce').fillna(0)
            df_clean['cost']               = pd.to_numeric(df_clean['revenue'],        errors='coerce').fillna(0)
            df_clean['avg_price']          = pd.to_numeric(df_clean['avg_price'],      errors='coerce').fillna(0)

            df_clean['year']    = df_clean['transaction_date'].dt.year
            df_clean['month']   = df_clean['transaction_date'].dt.month
            df_clean['quarter'] = df_clean['transaction_date'].dt.quarter

            df_clean['cost_velocity'] = df_clean['cost'] / df_clean['quantity_purchased'].replace(0, 1)

            df_clean['cost_segment'] = pd.qcut(
                df_clean['cost'].abs(),
                q=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )

            return df_clean

        except Exception as e:
            logger.error(f"Error transforming purchase product analysis: {str(e)}")
            raise

    def transform_purchase_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform purchase trend data
        """
        try:
            logger.info("Transforming purchase trend data")

            df_clean = df.copy()

            df_clean['business_unit_name'] = df_clean['business_unit_name'].fillna('Unknown')
            df_clean['staff_code']         = df_clean['staff_code'].fillna('Unknown')
            df_clean['transaction_type']   = df_clean['transaction_type'].fillna('Unknown')

            df_clean['transaction_date'] = pd.to_datetime(df_clean['transaction_date'], errors='coerce')

            numeric_cols = ['total_transactions', 'total_cost', 'avg_transaction_value', 'unique_vendors']
            for col in numeric_cols:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

            df_clean['year']         = df_clean['transaction_date'].dt.year
            df_clean['month']        = df_clean['transaction_date'].dt.month
            df_clean['quarter']      = df_clean['transaction_date'].dt.quarter
            df_clean['day_of_week']  = df_clean['transaction_date'].dt.dayofweek
            df_clean['day_of_month'] = df_clean['transaction_date'].dt.day
            df_clean['week_of_year'] = df_clean['transaction_date'].dt.isocalendar().week
            df_clean['is_weekend']   = df_clean['day_of_week'].isin([5, 6]).astype(int)

            # Cyclical features
            df_clean['month_sin']        = np.sin(2 * np.pi * df_clean['month'] / 12)
            df_clean['month_cos']        = np.cos(2 * np.pi * df_clean['month'] / 12)
            df_clean['day_of_week_sin']  = np.sin(2 * np.pi * df_clean['day_of_week'] / 7)
            df_clean['day_of_week_cos']  = np.cos(2 * np.pi * df_clean['day_of_week'] / 7)

            return df_clean

        except Exception as e:
            logger.error(f"Error transforming purchase trend data: {str(e)}")
            raise

    def transform_date_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform monthly cost data
        """
        try:
            logger.info("Transforming date cost data")

            df_clean = df.copy()

            df_clean['month']    = pd.to_numeric(df_clean['month'],    errors='coerce').fillna(0).astype(int)
            df_clean['year']     = pd.to_numeric(df_clean['year'],     errors='coerce').fillna(0).astype(int)
            df_clean['amt_local'] = pd.to_numeric(df_clean['amt_local'], errors='coerce').fillna(0)

            df_clean = df_clean[(df_clean['month'] > 0) & (df_clean['year'] > 0)]

            if not df_clean.empty:
                df_clean['cost_level'] = pd.qcut(
                    df_clean['amt_local'],
                    q=3,
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                )

            return df_clean

        except Exception as e:
            logger.error(f"Error transforming date cost data: {str(e)}")
            raise