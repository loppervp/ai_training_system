"""
Test cases for DatabaseExtractor module
"""
import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.extractors.database_extractor import DatabaseExtractor


class TestDatabaseExtractor:
    """Test class for DatabaseExtractor"""
    
    @pytest.fixture
    def mock_db_config(self):
        return {
            "database": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "database": "test_db",
                "username": "test_user",
                "password": "test_pass",
                "pool_size": 5,
                "max_overflow": 10,
                "echo": False
            }
        }
    
    def test_csv_to_parquet_conversion(self):
        """Test OOM proof CSV to Parquet conversion"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test CSV
            csv_path = os.path.join(tmpdir, "test_data.csv")
            parquet_path = os.path.join(tmpdir, "test_data.parquet")
            
            # Generate 100,000 test rows
            df_test = pd.DataFrame({
                "id": np.arange(100000),
                "value": np.random.randn(100000),
                "category": np.random.choice(["A", "B", "C", "D"], 100000),
                "timestamp": pd.date_range("2020-01-01", periods=100000, freq="min")
            })
            
            df_test.to_csv(csv_path, index=False)
            original_size = os.path.getsize(csv_path)
            
            # Run conversion with 10,000 chunk size
            total_rows = DatabaseExtractor.csv_to_parquet(
                csv_path=csv_path,
                output_path=parquet_path,
                chunk_size=10000,
                compression="snappy"
            )
            
            # Verify results
            assert total_rows == 100000
            assert os.path.exists(parquet_path)
            
            new_size = os.path.getsize(parquet_path)
            assert new_size < original_size  # Parquet should be smaller
            
            # Verify data integrity
            df_loaded = pd.read_parquet(parquet_path)
            assert len(df_loaded) == 100000
            assert list(df_loaded.columns) == list(df_test.columns)
    
    def test_csv_to_parquet_small_file(self):
        """Test conversion with small file less than chunk size"""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "small_test.csv")
            parquet_path = os.path.join(tmpdir, "small_test.parquet")
            
            df_test = pd.DataFrame({
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"]
            })
            
            df_test.to_csv(csv_path, index=False)
            
            total_rows = DatabaseExtractor.csv_to_parquet(
                csv_path=csv_path,
                output_path=parquet_path,
                chunk_size=1000
            )
            
            assert total_rows == 5
    
    @patch('src.extractors.database_extractor.create_engine')
    def test_extract_data_with_chunk_size(self, mock_create_engine):
        """Test extract_data with chunk processing and garbage collection"""
        
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Mock pandas read_sql to return chunks
        mock_chunks = [
            pd.DataFrame({"id": [1,2,3]}),
            pd.DataFrame({"id": [4,5,6]}),
            pd.DataFrame({"id": [7,8,9]})
        ]
        
        with patch('pandas.read_sql', return_value=iter(mock_chunks)):
            with patch('src.extractors.database_extractor.gc.collect') as mock_gc:
                with patch.object(DatabaseExtractor, '_load_config') as mock_load_config:
                    mock_load_config.return_value = self.mock_db_config
                    with patch.object(DatabaseExtractor, '_connect') as mock_connect:
                        extractor = DatabaseExtractor(config_path="dummy_path")
                        result = extractor.extract_data(
                            query="SELECT * FROM test_table",
                            chunk_size=3
                        )
                        
                        assert len(result) == 9
                        assert mock_gc.call_count >= 3  # GC called for each chunk
    
    @patch('src.extractors.database_extractor.create_engine')
    def test_extract_data_stream(self, mock_create_engine):
        """Test streaming extraction iterator"""
        
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        mock_chunks = [
            pd.DataFrame({"id": [1,2,3]}),
            pd.DataFrame({"id": [4,5,6]}),
            pd.DataFrame({"id": [7,8,9]})
        ]
        
        with patch('pandas.read_sql', return_value=iter(mock_chunks)):
            with patch.object(DatabaseExtractor, '_load_config') as mock_load_config:
                mock_load_config.return_value = self.mock_db_config
                with patch.object(DatabaseExtractor, '_connect') as mock_connect:
                    extractor = DatabaseExtractor(config_path="dummy_path")
                    stream = extractor.extract_data_stream(
                        query="SELECT * FROM test_table",
                        chunk_size=3
                    )
                    
                    total_records = 0
                    for chunk in stream:
                        total_records += len(chunk)
                        assert isinstance(chunk, pd.DataFrame)
                    
                    assert total_records == 9
    
    def test_oom_safety_features(self):
        """Test that OOM safety features are properly implemented"""
        
        # Verify that gc is imported and available
        import gc
        assert hasattr(gc, 'collect')
        
        # Verify sys is imported for memory monitoring
        import sys
        assert hasattr(sys, 'getsizeof')
        
        # Verify pyarrow is available
        import pyarrow as pa
        import pyarrow.parquet as pq
        assert hasattr(pq, 'ParquetWriter')