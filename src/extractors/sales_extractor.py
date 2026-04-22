"""
Sales Extractor Module
Specialized module for extracting sales data from scm_sal_main and scm_sal_data
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from .database_extractor import DatabaseExtractor

logger = logging.getLogger(__name__)


class SalesExtractor:
    """Class for extracting sales data with company-based data isolation"""
    
    def __init__(self, config_path: str = "config/database.json", companyfn: Optional[str] = None):
        """
        Initialize SalesExtractor
        
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
    
    def extract_sales_main(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        transaction_types: Optional[List[str]] = None,
        include_void: bool = False
    ) -> pd.DataFrame:
        """
        Extract data from scm_sal_main table with company-based data isolation
        
        Args:
            companyfn: Company code (uses instance companyfn if not provided)
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            transaction_types: Transaction types (sal_soe, sal_soc, sal_inv, etc.)
            include_void: Include voided transactions
            
        Returns:
            DataFrame containing data filtered by companyfn
        """
        try:
            # Use instance companyfn if not provided (for data isolation)
            effective_companyfn = companyfn or self.companyfn
            
            # Build filters
            filters = {}
            
            if effective_companyfn:
                filters['companyfn'] = effective_companyfn
            
            if not include_void:
                filters['tag_void_yn'] = 'n'
            
            if transaction_types:
                filters['tag_table_usage'] = transaction_types
            
            # Build query with date filter
            # Only select columns that exist in database
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
                    delivtype_desc,
                    sendby_desc,
                    notes_memo,
                    tag_void_yn,
                    salestaxpct
                FROM scm_sal_main
                WHERE 1=1
            """
            
            params = {}
            
            if effective_companyfn:
                query += " AND companyfn = :companyfn"
                params['companyfn'] = effective_companyfn
            
            # Temporarily remove tag_void_yn filter for debugging
            # if not include_void:
            #     query += " AND tag_void_yn = 'n'"
            
            # Temporarily remove transaction_types filter for debugging
            # if transaction_types:
            #     placeholders = ", ".join([f":type_{i}" for i in range(len(transaction_types))])
            #     query += f" AND tag_table_usage IN ({placeholders})"
            #     for i, t in enumerate(transaction_types):
            #         params[f"type_{i}"] = t
            
            if date_from:
                query += " AND date_trans >= :date_from"
                params['date_from'] = date_from
            
            if date_to:
                query += " AND date_trans <= :date_to"
                params['date_to'] = date_to
            
            query += " ORDER BY date_trans DESC"
            
            logger.info(f"Extracting scm_sal_main with query: {query}")
            logger.info(f"Params: {params}")
            
            result = self.db_extractor.extract_data(query, params)
            logger.info(f"Result: {len(result)} records")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting scm_sal_main: {str(e)}")
            raise
    
    def extract_sales_data(
        self,
        companyfn: Optional[str] = None,
        uniquenum_pri: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        product_codes: Optional[List[str]] = None,
        include_void: bool = False
    ) -> pd.DataFrame:
        """
        Extract data from scm_sal_data table
        
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
            # Only select columns that exist in database
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
                    gst_taxa_code,
                    date_trans,
                    party_unique,
                    party_code,
                    party_desc,
                    staff_code,
                    staff_unique,
                    deptunit_code,
                    location_code,
                    notes_memo,
                    tag_void_yn,
                    tag_item_taxable_yn,
                    tag_closedmain_yn
                FROM scm_sal_data
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
            
            logger.info(f"Extracting scm_sal_data")
            
            return self.db_extractor.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting scm_sal_data: {str(e)}")
            raise
    
    def extract_sales_with_details(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        transaction_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Extract complete sales data (JOIN scm_sal_main and scm_sal_data)
        
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
                    main.entprojfn_code,
                    main.location_code as main_location_code,
                    main.creditterm_desc,
                    main.delivtype_desc,
                    main.notes_memo as main_notes_memo,
                    main.tag_void_yn as main_tag_void_yn,
                    main.salestaxpct,
                    
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
                    data.gst_taxa_code,
                    data.tag_item_taxable_yn,
                    data.notes_memo as line_notes_memo
                    
                FROM scm_sal_main main
                INNER JOIN scm_sal_data data 
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
            
            logger.info("Extracting complete sales data with JOIN")
            
            return self.db_extractor.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting complete sales data: {str(e)}")
            raise
    
    def extract_customer_analysis_data(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract data for customer analysis
        
        Args:
            companyfn: Company code
            date_from: Start date
            date_to: End date
            
        Returns:
            DataFrame with customer analysis data
        """
        try:
            query = """
                SELECT 
                    main.party_unique as customer_id,
                    main.party_code as customer_code,
                    main.party_desc as customer_name,
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
                    main.staff_code as salesperson
                FROM scm_sal_main main
                INNER JOIN scm_sal_data data 
                    ON main.uniquenum_pri = data.uniquenum_pri
                    AND main.companyfn = data.companyfn
                WHERE main.tag_void_yn = 'n'
                    AND data.tag_void_yn = 'n'
                    AND main.tag_table_usage IN ('sal_soe', 'sal_soc', 'sal_inv')
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
            
            logger.info("Extracting customer analysis data")
            
            return self.db_extractor.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting customer analysis data: {str(e)}")
            raise
    
    def extract_customer_retention_data(
        self,
        companyfn: Optional[str] = None,
        lookback_days: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data for customer retention analysis
        
        Args:
            companyfn: Company code
            lookback_days: Number of days to look back (None = unlimited)
            
        Returns:
            DataFrame with customer retention data
        """
        try:
            query = """
                WITH customer_stats AS (
                    SELECT 
                        party_unique as customer_id,
                        party_code as customer_code,
                        party_desc as customer_name,
                        MIN(date_trans) as first_purchase_date,
                        MAX(date_trans) as last_purchase_date,
                        COUNT(DISTINCT uniquenum_pri) as total_purchases,
                        SUM(amount_local) as total_spent,
                        AVG(amount_local) as avg_purchase_value,
                        COUNT(DISTINCT EXTRACT(YEAR FROM date_trans) * 100 + EXTRACT(MONTH FROM date_trans)) as active_months
                    FROM scm_sal_main
                    WHERE tag_void_yn = 'n'
                        AND tag_table_usage = 'sal_inv'
                    GROUP BY party_unique, party_code, party_desc
                ),
                max_date AS (
                    SELECT MAX(last_purchase_date) as max_last_purchase FROM customer_stats
                )
                SELECT 
                    cs.customer_id,
                    cs.customer_code,
                    cs.customer_name,
                    cs.first_purchase_date,
                    cs.last_purchase_date,
                    EXTRACT(DAYS FROM md.max_last_purchase - cs.last_purchase_date) as days_since_last_purchase,
                    cs.total_purchases,
                    cs.total_spent,
                    cs.avg_purchase_value,
                    CASE 
                        WHEN cs.total_purchases > 0 THEN 
                            cs.total_purchases::float / NULLIF(cs.active_months, 0)
                        ELSE 0 
                    END as purchase_frequency,
                    EXTRACT(DAYS FROM cs.last_purchase_date - cs.first_purchase_date) as customer_lifetime_days,
                    CASE 
                        WHEN EXTRACT(DAYS FROM md.max_last_purchase - cs.last_purchase_date) > 365 THEN 1 
                        ELSE 0 
                    END as is_churned
                FROM customer_stats cs
                CROSS JOIN max_date md
                ORDER BY cs.total_spent DESC
            """
            
            params = {}
            
            if companyfn:
                # Modify query to include companyfn filter
                query = query.replace(
                    "WHERE tag_void_yn = 'n'",
                    f"WHERE tag_void_yn = 'n' AND companyfn = :companyfn"
                )
                params['companyfn'] = companyfn
            
            logger.info("Extracting customer retention analysis data")
            
            return self.db_extractor.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting customer retention data: {str(e)}")
            raise
    
    def extract_product_analysis_data(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract data for product analysis
        
        Args:
            companyfn: Company code
            date_from: Start date
            date_to: End date
            
        Returns:
            DataFrame with product analysis data
        """
        try:
            query = """
                SELECT 
                    data.stkcode_code as product_code,
                    data.stkcode_desc as product_name,
                    data.stkcate_code as category_code,
                    data.stkcate_desc as category_name,
                    data.brand_code,
                    data.brand_desc,
                    data.stkvendor_code as vendor_code,
                    data.stkvendor_desc as vendor_name,
                    main.date_trans as transaction_date,
                    EXTRACT(YEAR FROM main.date_trans) as year,
                    EXTRACT(MONTH FROM main.date_trans) as month,
                    EXTRACT(QUARTER FROM main.date_trans) as quarter,
                    SUM(data.qnty_total) as quantity_sold,
                    SUM(data.amount_local) as revenue,
                    AVG(data.price_unitrate_local) as avg_price,
                    COUNT(DISTINCT main.party_unique) as unique_customers,
                    COUNT(DISTINCT main.uniquenum_pri) as num_transactions
                FROM scm_sal_main main
                INNER JOIN scm_sal_data data 
                    ON main.uniquenum_pri = data.uniquenum_pri
                    AND main.companyfn = data.companyfn
                WHERE main.tag_void_yn = 'n'
                    AND data.tag_void_yn = 'n'
                    AND main.tag_table_usage IN ('sal_soe', 'sal_soc', 'sal_inv')
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
            
            query += """
                GROUP BY 
                    data.stkcode_code,
                    data.stkcode_desc,
                    data.stkcate_code,
                    data.stkcate_desc,
                    data.brand_code,
                    data.brand_desc,
                    data.stkvendor_code,
                    data.stkvendor_desc,
                    main.date_trans,
                    EXTRACT(YEAR FROM main.date_trans),
                    EXTRACT(MONTH FROM main.date_trans),
                    EXTRACT(QUARTER FROM main.date_trans)
                ORDER BY revenue DESC
            """
            
            logger.info("Extracting product analysis data")
            
            return self.db_extractor.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting product analysis data: {str(e)}")
            raise
    
    def extract_sales_trend_data(
        self,
        companyfn: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract data for sales trend analysis
        
        Args:
            companyfn: Company code
            date_from: Start date
            date_to: End date
            
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
                    staff_code as salesperson,
                    tag_table_usage as transaction_type,
                    COUNT(*) as total_transactions,
                    SUM(amount_local) as total_revenue,
                    AVG(amount_local) as avg_transaction_value,
                    COUNT(DISTINCT party_unique) as unique_customers
                FROM scm_sal_main
                WHERE tag_void_yn = 'n'
                    AND tag_table_usage IN ('sal_inv', 'sal_soc')
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
            
            logger.info("Extracting sales trend data")
            
            return self.db_extractor.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting trend data: {str(e)}")
            raise
        
    def extract_date_revenue_data(
        self,
        companyfn: Optional[str] = None,
        uniquenum_pri: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        product_codes: Optional[List[str]] = None,
        include_void: bool = False
    ) -> pd.DataFrame:
        """
        Extract data from scm_sal_data table
        
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
            # Only select columns that exist in database
            query = """
                SELECT 
                    companyfn,
                    EXTRACT(MONTH FROM date_trans)::int AS month,
                    EXTRACT(YEAR FROM date_trans)::int AS year,
                    SUM(amount_local) AS amt_local,
                    COUNT(DISTINCT uniquenum_pri) AS num_transactions
                FROM scm_sal_main
                WHERE (tag_table_usage = 'sal_soc' or (tag_table_usage = 'sal_soe' and tag_closed02_yn = 'n'))
                AND tag_deleted_yn = 'n'
                AND tag_void_yn = 'n'
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
            
            query += """
                GROUP BY companyfn, EXTRACT(MONTH FROM date_trans), EXTRACT(YEAR FROM date_trans)
                ORDER BY year DESC, month DESC
            """
            logger.info(f"Extracting scm_sal_data")
            
            return self.db_extractor.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting scm_sal_data: {str(e)}")
            raise
    
    def get_available_date_range(self, companyfn: Optional[str] = None) -> Tuple[str, str]:
        """
        Lấy khoảng thời gian dữ liệu thực tế có sẵn trong database
        Returns:
            Tuple (min_date, max_date) dạng 'YYYY-MM-DD'
        """
        try:
            query = """
                SELECT 
                    MIN(date_trans) as min_date,
                    MAX(date_trans) as max_date
                FROM scm_sal_main
                WHERE tag_void_yn = 'n'
            """
            
            params = {}
            if companyfn or self.companyfn:
                effective_companyfn = companyfn or self.companyfn
                query += " AND companyfn = :companyfn"
                params['companyfn'] = effective_companyfn
            
            result = self.db_extractor.extract_data(query, params)
            
            if len(result) == 0:
                return ('2009-01-01', '2026-12-31')
            
            min_date = result.iloc[0]['min_date'].strftime('%Y-%m-%d')
            max_date = result.iloc[0]['max_date'].strftime('%Y-%m-%d')
            
            return (min_date, max_date)
            
        except Exception as e:
            logger.error(f"Error getting date range: {str(e)}")
            return ('2009-01-01', '2026-12-31')

    def close(self):
        """Close connection"""
        self.db_extractor.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
