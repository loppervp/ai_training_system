"""
Purchase Extractor Module
Specialized module for extracting purchase data from scm_pur_main and scm_pur_data
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from .database_extractor import DatabaseExtractor

logger = logging.getLogger(__name__)


class PurchaseExtractor:
    """Class for extracting purchase data with company-based data isolation"""

    def __init__(self, config_path: str = "config/database.json", companyfn: Optional[str] = None):
        """
        Initialize PurchaseExtractor

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

    def extract_purchase_main(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        transaction_types: Optional[List[str]] = None,
        include_void: bool = False
    ) -> pd.DataFrame:
        """
        Extract data from scm_pur_main table with company-based data isolation

        Args:
            companyfn: Company code (uses instance companyfn if not provided)
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            transaction_types: Transaction types (pur_po, pur_grn, pur_inv, etc.)
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
                    dnum_auto,
                    tag_table_usage,
                    party_unique,
                    party_code,
                    party_desc,
                    date_trans,
                    date_due,
                    amount_local,
                    amount_forex,
                    curr_short_forex,
                    curr_rate_forex_f_calc,
                    deptunit_code,
                    deptunit_desc,
                    staff_code,
                    staff_unique,
                    location_code,
                    creditterm_desc,
                    notes_memo,
                    tag_void_yn
                FROM scm_pur_main
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

            logger.info(f"Extracting scm_pur_main | params: {params}")
            result = self.db_extractor.extract_data(query, params)
            logger.info(f"Result: {len(result)} records")
            return result

        except Exception as e:
            logger.error(f"Error extracting scm_pur_main: {str(e)}")
            raise

    def extract_purchase_data(
        self,
        companyfn: Optional[str] = None,
        uniquenum_pri: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        product_codes: Optional[List[str]] = None,
        include_void: bool = False
    ) -> pd.DataFrame:
        """
        Extract data from scm_pur_data table

        Args:
            companyfn: Company code
            uniquenum_pri: Specific order ID
            date_from: Start date
            date_to: End date
            product_codes: List of product codes
            include_void: Include voided transactions

        Returns:
            DataFrame containing data
        """
        try:
            query = """
                SELECT
                    idcode,
                    uniquenum_uniq,
                    uniquenum_pri,
                    companyfn,
                    tag_table_usage,
                    row_item_num,
                    stkcode_code,
                    stkcode_unique,
                    stkcode_desc,
                    skucode_code,
                    brand_code,
                    brand_unique,
                    brand_desc,
                    stkcate_code,
                    stkcate_unique,
                    stkcate_desc,
                    stkvendor_code,
                    stkvendor_desc,
                    qnty_total,
                    qnty_uomstk,
                    bal_qnty_total,
                    bal_qnty_uomstk,
                    uom_stk_code,
                    price_unitlist_forex,
                    price_unitlist_local,
                    price_unitrate_forex,
                    price_unitrate_local,
                    discount_pct,
                    amount_forex,
                    amount_local,
                    amount_tax_forex,
                    amount_tax_local,
                    date_trans,
                    party_unique,
                    party_code,
                    party_desc,
                    staff_code,
                    staff_unique,
                    deptunit_code,
                    location_code,
                    notes_memo,
                    tag_void_yn
                FROM scm_pur_data
                WHERE 1=1
            """

            params = {}

            if companyfn:
                query += " AND companyfn = :companyfn"
                params['companyfn'] = companyfn

            if uniquenum_pri:
                query += " AND uniquenum_pri = :uniquenum_pri"
                params['uniquenum_pri'] = uniquenum_pri

            if not include_void:
                query += " AND tag_void_yn = 'n'"

            if product_codes:
                placeholders = ", ".join([f":prod_{i}" for i in range(len(product_codes))])
                query += f" AND stkcode_code IN ({placeholders})"
                for i, code in enumerate(product_codes):
                    params[f"prod_{i}"] = code

            if date_from:
                query += " AND date_trans >= :date_from"
                params['date_from'] = date_from

            if date_to:
                query += " AND date_trans <= :date_to"
                params['date_to'] = date_to

            query += " ORDER BY date_trans DESC, row_item_num"

            logger.info("Extracting scm_pur_data")
            return self.db_extractor.extract_data(query, params)

        except Exception as e:
            logger.error(f"Error extracting scm_pur_data: {str(e)}")
            raise

    def extract_purchase_with_details(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        transaction_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract complete purchase data (JOIN scm_pur_main and scm_pur_data)

        Args:
            companyfn: Company code
            date_from: Start date
            date_to: End date
            transaction_types: Transaction types

        Returns:
            DataFrame containing complete data
        """
        try:
            query = """
                SELECT
                    main.uniquenum_pri,
                    main.companyfn,
                    main.dnum_auto,
                    main.tag_table_usage as main_transaction_type,
                    main.party_unique as main_party_unique,
                    main.party_code as main_party_code,
                    main.party_desc as main_party_desc,
                    main.date_trans as main_date_trans,
                    main.amount_local as main_amount_local,
                    main.amount_forex as main_amount_forex,
                    main.curr_short_forex,
                    main.deptunit_code as main_deptunit_code,
                    main.deptunit_desc as main_deptunit_desc,
                    main.staff_code as main_staff_code,
                    main.staff_unique as main_staff_unique,
                    main.location_code as main_location_code,
                    main.creditterm_desc,
                    main.notes_memo as main_notes_memo,
                    main.tag_void_yn as main_tag_void_yn,

                    data.uniquenum_uniq,
                    data.row_item_num,
                    data.stkcode_code,
                    data.stkcode_unique,
                    data.stkcode_desc,
                    data.skucode_code,
                    data.brand_code,
                    data.brand_desc,
                    data.stkcate_code,
                    data.stkcate_desc,
                    data.stkvendor_code,
                    data.stkvendor_desc,
                    data.qnty_total,
                    data.qnty_uomstk,
                    data.bal_qnty_total,
                    data.uom_stk_code,
                    data.price_unitrate_forex,
                    data.price_unitrate_local,
                    data.discount_pct,
                    data.amount_forex as line_amount_forex,
                    data.amount_local as line_amount_local,
                    data.amount_tax_forex,
                    data.amount_tax_local,
                    data.notes_memo as line_notes_memo

                FROM scm_pur_main main
                INNER JOIN scm_pur_data data
                    ON main.uniquenum_pri = data.uniquenum_pri
                    AND main.companyfn = data.companyfn
                WHERE main.tag_void_yn = 'n'
                    AND data.tag_void_yn = 'n'
            """

            params = {}

            if companyfn:
                query += " AND main.companyfn = :companyfn"
                params['companyfn'] = companyfn

            if transaction_types:
                placeholders = ", ".join([f":type_{i}" for i in range(len(transaction_types))])
                query += f" AND main.tag_table_usage IN ({placeholders})"
                for i, t in enumerate(transaction_types):
                    params[f"type_{i}"] = t

            if date_from:
                query += " AND main.date_trans >= :date_from"
                params['date_from'] = date_from

            if date_to:
                query += " AND main.date_trans <= :date_to"
                params['date_to'] = date_to

            query += " ORDER BY main.date_trans DESC, data.row_item_num"

            logger.info("Extracting complete purchase data with JOIN")
            return self.db_extractor.extract_data(query, params)

        except Exception as e:
            logger.error(f"Error extracting complete purchase data: {str(e)}")
            raise

    def extract_vendor_analysis_data(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract data for vendor analysis (same sales customer analysis)

        Returns:
            DataFrame with vendor analysis data
        """
        try:
            query = """
                SELECT
                    main.party_unique as vendor_id,
                    main.party_code as vendor_code,
                    main.party_desc as vendor_name,
                    main.date_trans as transaction_date,
                    EXTRACT(YEAR FROM main.date_trans) as year,
                    EXTRACT(MONTH FROM main.date_trans) as month,
                    EXTRACT(QUARTER FROM main.date_trans) as quarter,
                    EXTRACT(DOW FROM main.date_trans) as day_of_week,
                    data.stkcode_code as product_code,
                    data.stkcode_desc as product_name,
                    data.stkcate_code as category_code,
                    data.stkcate_desc as category_name,
                    data.brand_code,
                    data.brand_desc,
                    data.qnty_total as quantity,
                    data.price_unitrate_local as unit_price,
                    data.amount_local as line_amount,
                    main.tag_table_usage as transaction_type,
                    main.deptunit_code as business_unit,
                    main.staff_code as staff_code
                FROM scm_pur_main main
                INNER JOIN scm_pur_data data
                    ON main.uniquenum_pri = data.uniquenum_pri
                    AND main.companyfn = data.companyfn
                WHERE main.tag_void_yn = 'n'
                    AND data.tag_void_yn = 'n'
                    AND main.tag_table_usage IN ('pur_po', 'pur_poc', 'pur_inv')
            """

            params = {}

            if companyfn:
                query += " AND main.companyfn = :companyfn"
                params['companyfn'] = companyfn

            if date_from:
                query += " AND main.date_trans >= :date_from"
                params['date_from'] = date_from

            if date_to:
                query += " AND main.date_trans <= :date_to"
                params['date_to'] = date_to

            query += " ORDER BY main.date_trans DESC"

            logger.info("Extracting vendor analysis data")
            return self.db_extractor.extract_data(query, params)

        except Exception as e:
            logger.error(f"Error extracting vendor analysis data: {str(e)}")
            raise

    def extract_purchase_trend_data(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract data for purchase trend analysis

        Returns:
            DataFrame with trend data
        """
        try:
            query = """
                SELECT
                    date_trans as transaction_date,
                    EXTRACT(YEAR FROM date_trans) as year,
                    EXTRACT(MONTH FROM date_trans) as month,
                    EXTRACT(QUARTER FROM date_trans) as quarter,
                    EXTRACT(DOW FROM date_trans) as day_of_week,
                    EXTRACT(DAY FROM date_trans) as day_of_month,
                    EXTRACT(WEEK FROM date_trans) as week_of_year,
                    deptunit_code as business_unit,
                    deptunit_desc as business_unit_name,
                    staff_code as staff_code,
                    tag_table_usage as transaction_type,
                    COUNT(*) as total_transactions,
                    SUM(amount_local) as total_cost,
                    AVG(amount_local) as avg_transaction_value,
                    COUNT(DISTINCT party_unique) as unique_vendors
                FROM scm_pur_main
                WHERE tag_void_yn = 'n'
                    AND tag_table_usage IN ('pur_inv', 'pur_poc')
            """

            params = {}

            if companyfn:
                query += " AND companyfn = :companyfn"
                params['companyfn'] = companyfn

            if date_from:
                query += " AND date_trans >= :date_from"
                params['date_from'] = date_from

            if date_to:
                query += " AND date_trans <= :date_to"
                params['date_to'] = date_to

            query += """
                GROUP BY
                    date_trans,
                    EXTRACT(YEAR FROM date_trans),
                    EXTRACT(MONTH FROM date_trans),
                    EXTRACT(QUARTER FROM date_trans),
                    EXTRACT(DOW FROM date_trans),
                    EXTRACT(DAY FROM date_trans),
                    EXTRACT(WEEK FROM date_trans),
                    deptunit_code,
                    deptunit_desc,
                    staff_code,
                    tag_table_usage
                ORDER BY date_trans DESC
            """

            logger.info("Extracting purchase trend data")
            return self.db_extractor.extract_data(query, params)

        except Exception as e:
            logger.error(f"Error extracting purchase trend data: {str(e)}")
            raise

    def extract_date_cost_data(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract monthly cost summary (same as sales extract_date_revenue_data

        Returns:
            DataFrame with monthly cost data
        """
        try:
            query = """
                SELECT
                    companyfn,
                    EXTRACT(MONTH FROM date_trans)::int AS month,
                    EXTRACT(YEAR FROM date_trans)::int AS year,
                    SUM(amount_local) AS amt_local,
                    COUNT(DISTINCT uniquenum_pri) AS num_transactions
                FROM scm_pur_main
                WHERE (tag_table_usage = 'pur_poc' or (tag_table_usage = 'pur_po' and tag_closed02_yn = 'n'))
                AND tag_void_yn = 'n'
            """

            params = {}

            if companyfn:
                query += " AND companyfn = :companyfn"
                params['companyfn'] = companyfn

            if date_from:
                query += " AND date_trans >= :date_from"
                params['date_from'] = date_from

            if date_to:
                query += " AND date_trans <= :date_to"
                params['date_to'] = date_to

            query += """
                GROUP BY companyfn, EXTRACT(MONTH FROM date_trans), EXTRACT(YEAR FROM date_trans)
                ORDER BY year DESC, month DESC
            """

            logger.info("Extracting purchase cost data by date")
            return self.db_extractor.extract_data(query, params)

        except Exception as e:
            logger.error(f"Error extracting purchase cost data: {str(e)}")
            raise

    def close(self):
        """Close connection"""
        self.db_extractor.close()
