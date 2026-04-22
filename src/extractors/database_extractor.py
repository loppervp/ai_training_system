"""
Database Extractor Module
Module for extracting data from PostgreSQL database
"""

import json
import logging
import re
import gc
import sys
from typing import Dict, List, Optional, Any, Iterator, Callable
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()

logger = logging.getLogger(__name__)


class DatabaseExtractor:
    """Class for extracting data from database"""
    
    def __init__(self, config_path: str = "config/database.json"):
        """
        Initialize DatabaseExtractor
        
        Args:
            config_path: Path to database configuration file
        """
        # Handle relative path from ai_training_system directory
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not os.path.isabs(config_path):
            config_path = os.path.join(base_dir, config_path)
        
        self.config = self._load_config(config_path)
        self.engine = None
        self.session = None
        self._connect()
    
    def _load_config(self, config_path: str) -> Dict:
        """Read configuration file and override with environment variables"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Override database config with environment variables if present
            db_config = config.get('source_database') or config.get('database', {})
            
            # Map environment variables to config keys
            env_mapping = {
                'DB_TYPE': 'type',
                'DB_HOST': 'host',
                'DB_PORT': 'port',
                'DB_NAME': 'database',
                'DB_USER': 'username',
                'DB_PASSWORD': 'password',
                'DB_SCHEMA': 'schema',
                'DB_POOL_SIZE': 'pool_size',
                'DB_MAX_OVERFLOW': 'max_overflow',
                'DB_POOL_TIMEOUT': 'pool_timeout',
                'DB_ECHO': 'echo'
            }
            
            # Pattern to match ${VAR_NAME:-default} syntax
            env_pattern = re.compile(r'\$\{([^}:-]+)(?::-([^}]+))?\}')
            
            # First process all values and resolve environment placeholders
            for key, value in db_config.items():
                if isinstance(value, str):
                    match = env_pattern.match(value)
                    if match:
                        env_var, default_val = match.groups()
                        env_value = os.getenv(env_var, default_val)
                        
                        # Skip password required field if no value
                        if env_var == 'DB_PASSWORD' and env_value is None:
                            logger.warning(f"Environment variable {env_var} not set!")
                        
                        if env_value is not None:
                            db_config[key] = env_value

            # Apply direct environment variables override
            for env_var, config_key in env_mapping.items():
                env_value = os.getenv(env_var)
                if env_value is not None:
                    # Convert to correct type
                    if config_key in ['port', 'pool_size', 'max_overflow', 'pool_timeout']:
                        env_value = int(env_value)
                    elif config_key == 'echo':
                        env_value = env_value.lower() in ('true', '1', 'yes')
                    
                    db_config[config_key] = env_value
            
            # Final type conversion for integer fields
            for int_key in ['port', 'pool_size', 'max_overflow', 'pool_timeout']:
                if int_key in db_config and isinstance(db_config[int_key], str):
                    db_config[int_key] = int(db_config[int_key])
            
            # Final type conversion for boolean field
            if 'echo' in db_config and isinstance(db_config['echo'], str):
                db_config['echo'] = db_config['echo'].lower() in ('true', '1', 'yes')
            
            # Update back the config
            if 'source_database' in config:
                config['source_database'] = db_config
            else:
                config['database'] = db_config
            
            logger.info("Database configuration loaded successfully (with environment variables override)")
            return config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error reading JSON file: {config_path}")
            raise
    
    def _connect(self):
        """Connect to database"""
        try:
            # Support both 'database' and 'source_database'
            db_config = self.config.get('source_database') or self.config.get('database')
            
            # Create connection string
            connection_string = (
                f"postgresql://{db_config['username']}:{db_config['password']}"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            )
            
            # Create engine
            self.engine = create_engine(
                connection_string,
                pool_size=db_config.get('pool_size', 10),
                max_overflow=db_config.get('max_overflow', 20),
                pool_timeout=db_config.get('pool_timeout', 30),
                echo=db_config.get('echo', False)
            )
            
            # Create session
            Session = sessionmaker(bind=self.engine)
            self.session = Session()
            
            logger.info("Database connection successful")
            
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
    
    def extract_data(
        self,
        query: str,
        params: Optional[Dict] = None,
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data using SQL query
        
        Args:
            query: SQL query
            params: Query parameters
            chunk_size: Chunk size for reading large data
            
        Returns:
            DataFrame containing data
        """
        try:
            if chunk_size:
                # Read in chunks for large data
                chunks = []
                total_records = 0
                for chunk in pd.read_sql(
                    text(query),
                    self.engine,
                    params=params,
                    chunksize=chunk_size
                ):
                    chunks.append(chunk)
                    total_records += len(chunk)
                    logger.info(f"Read {len(chunk)} records | Total: {total_records} | Memory used: {sys.getsizeof(chunk) / 1024 / 1024:.2f} MB")
                    
                    # Force garbage collection to free memory
                    gc.collect()
                
                df = pd.concat(chunks, ignore_index=True)
                del chunks
                gc.collect()
            else:
                # Read all
                df = pd.read_sql(text(query), self.engine, params=params)
            
            logger.info(f"Successfully extracted {len(df)} records | Total memory: {sys.getsizeof(df) / 1024 / 1024:.2f} MB")
            return df
            
        except Exception as e:
            logger.error(f"Data extraction error: {str(e)}")
            raise

    def extract_data_stream(
        self,
        query: str,
        params: Optional[Dict] = None,
        chunk_size: int = 50000
    ) -> Iterator[pd.DataFrame]:
        """
        Stream data extraction with memory optimization (NO OOM)
        Does NOT load all data into memory - returns iterator of chunks
        
        Args:
            query: SQL query
            params: Query parameters
            chunk_size: Chunk size for reading large data
            
        Yields:
            DataFrame chunks one by one
        """
        try:
            logger.info(f"Starting streaming extraction with chunk size: {chunk_size}")
            total_records = 0
            
            for chunk in pd.read_sql(
                text(query),
                self.engine,
                params=params,
                chunksize=chunk_size
            ):
                total_records += len(chunk)
                logger.info(f"Streaming chunk | Records: {len(chunk)} | Total: {total_records} | Memory: {sys.getsizeof(chunk) / 1024 / 1024:.2f} MB")
                
                yield chunk
                
                # Force clean memory before next chunk
                del chunk
                gc.collect()
            
            logger.info(f"Streaming completed. Total records extracted: {total_records}")
            
        except Exception as e:
            logger.error(f"Streaming extraction error: {str(e)}")
            raise

    def extract_to_parquet(
        self,
        output_path: str,
        query: str,
        params: Optional[Dict] = None,
        chunk_size: int = 100000,
        compression: str = 'snappy'
    ) -> int:
        """
        ✅ OOM PROOF: Extract large dataset directly to Parquet file
        Never loads full dataset into memory
        
        Args:
            output_path: Path to save parquet file
            query: SQL query
            params: Query parameters
            chunk_size: Chunk size for reading
            compression: Compression algorithm
            
        Returns:
            Total number of records extracted
        """
        try:
            logger.info(f"Starting OOM-safe extraction to Parquet: {output_path}")
            total_records = 0
            writer = None
            
            for chunk in pd.read_sql(
                text(query),
                self.engine,
                params=params,
                chunksize=chunk_size
            ):
                total_records += len(chunk)
                logger.info(f"Processing chunk | Records: {len(chunk)} | Total: {total_records}")
                
                table = pa.Table.from_pandas(chunk)
                
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression=compression
                    )
                
                writer.write_table(table)
                
                # Clean memory aggressively
                del chunk
                del table
                gc.collect()
            
            if writer:
                writer.close()
            
            file_size = os.path.getsize(output_path) / 1024 / 1024
            logger.info(f"✅ Extraction completed successfully!")
            logger.info(f"Total records: {total_records}")
            logger.info(f"Output file: {output_path}")
            logger.info(f"File size: {file_size:.2f} MB")
            
            return total_records
            
        except Exception as e:
            logger.error(f"Parquet extraction error: {str(e)}")
            if 'writer' in locals() and writer:
                writer.close()
            raise
    
    def extract_table(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data from table
        
        Args:
            table_name: Table name
            columns: List of columns to retrieve
            filters: Filter conditions
            limit: Record limit
            
        Returns:
            DataFrame containing data
        """
        try:
            # Build query
            if columns:
                cols = ", ".join(columns)
            else:
                cols = "*"
            
            query = f"SELECT {cols} FROM {table_name}"
            
            # Add filter conditions
            if filters:
                conditions = []
                params = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        placeholders = ", ".join([f":{key}_{i}" for i in range(len(value))])
                        conditions.append(f"{key} IN ({placeholders})")
                        for i, v in enumerate(value):
                            params[f"{key}_{i}"] = v
                    else:
                        conditions.append(f"{key} = :{key}")
                        params[key] = value
                
                query += " WHERE " + " AND ".join(conditions)
            else:
                params = {}
            
            # Add limit
            if limit:
                query += f" LIMIT {limit}"
            
            logger.info(f"Query: {query}")
            
            return self.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting table {table_name}: {str(e)}")
            raise
    
    def extract_with_join(
        self,
        main_table: str,
        join_table: str,
        join_condition: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract data with JOIN
        
        Args:
            main_table: Main table
            join_table: Join table
            join_condition: Join condition
            columns: List of columns
            filters: Filter conditions
            limit: Record limit
            
        Returns:
            DataFrame containing data
        """
        try:
            if columns:
                cols = ", ".join(columns)
            else:
                cols = "*"
            
            query = f"""
                SELECT {cols}
                FROM {main_table}
                INNER JOIN {join_table} ON {join_condition}
            """
            
            # Add filter conditions
            if filters:
                conditions = []
                params = {}
                for key, value in filters.items():
                    if isinstance(value, list):
                        placeholders = ", ".join([f":{key}_{i}" for i in range(len(value))])
                        conditions.append(f"{key} IN ({placeholders})")
                        for i, v in enumerate(value):
                            params[f"{key}_{i}"] = v
                    else:
                        conditions.append(f"{key} = :{key}")
                        params[key] = value
                
                query += " WHERE " + " AND ".join(conditions)
            else:
                params = {}
            
            if limit:
                query += f" LIMIT {limit}"
            
            return self.extract_data(query, params)
            
        except Exception as e:
            logger.error(f"Error extracting with JOIN: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """
        Execute any query
        
        Args:
            query: SQL query
            params: Parameters
            
        Returns:
            Query result
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                return result.fetchall()
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict:
        """
        Get table information
        
        Args:
            table_name: Table name
            
        Returns:
            Dict containing table information
        """
        try:
            query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """
            
            result = self.execute_query(query, {"table_name": table_name})
            
            columns = []
            for row in result:
                columns.append({
                    "name": row[0],
                    "type": row[1],
                    "nullable": row[2] == "YES",
                    "default": row[3]
                })
            
            return {
                "table_name": table_name,
                "columns": columns,
                "column_count": len(columns)
            }
            
        except Exception as e:
            logger.error(f"Error getting table info {table_name}: {str(e)}")
            raise
    
    def close(self):
        """Close connection"""
        if self.session:
            self.session.close()
        if self.engine:
            self.engine.dispose()
        logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @staticmethod
    def csv_to_parquet(
        csv_path: str,
        output_path: str,
        chunk_size: int = 500000,
        compression: str = 'snappy'
    ) -> int:
        """
        ✅ OOM PROOF: Convert large CSV file to Parquet format
        Never loads full CSV into memory
        
        Args:
            csv_path: Path to input CSV file
            output_path: Path to output Parquet file
            chunk_size: Number of rows per chunk
            compression: Compression algorithm
            
        Returns:
            Total number of rows converted
        """
        try:
            logger.info(f"Converting CSV to Parquet: {csv_path}")
            total_rows = 0
            writer = None
            
            for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
                total_rows += len(chunk)
                logger.info(f"Processing chunk | Rows: {len(chunk)} | Total: {total_rows}")
                
                table = pa.Table.from_pandas(chunk)
                
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        table.schema,
                        compression=compression
                    )
                
                writer.write_table(table)
                
                del chunk
                del table
                gc.collect()
            
            if writer:
                writer.close()
            
            original_size = os.path.getsize(csv_path) / 1024 / 1024
            new_size = os.path.getsize(output_path) / 1024 / 1024
            ratio = (1 - new_size / original_size) * 100
            
            logger.info(f"✅ CSV conversion completed successfully!")
            logger.info(f"Total rows: {total_rows}")
            logger.info(f"Original CSV size: {original_size:.2f} MB")
            logger.info(f"Parquet size: {new_size:.2f} MB")
            logger.info(f"✅ Space saved: {ratio:.1f}%")
            
            return total_rows
            
        except Exception as e:
            logger.error(f"CSV to Parquet conversion error: {str(e)}")
            if 'writer' in locals() and writer:
                writer.close()
            raise
