"""
AI Training System - Main Entry Point
AI training system for sales data analysis
"""

import logging
import argparse
import sys
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_extraction(args):
    """Run data extraction"""
    from src.extractors.sales_extractor import SalesExtractor
    from src.transformers.data_transformer import DataTransformer
    
    logger.info("Starting data extraction")
    
    extractor = SalesExtractor()
    transformer = DataTransformer()
    
    # Nếu không truyền date range thì lấy tất cả dữ liệu có sẵn
    if not args.date_from or not args.date_to:
        # Lấy khoảng thời gian thực tế từ database
        min_date, max_date = extractor.get_available_date_range()
        date_from = args.date_from or min_date
        date_to = args.date_to or max_date
        logger.info(f"Auto detected full date range: {date_from} to {max_date}")
    else:
        date_from = args.date_from
        date_to = args.date_to
        logger.info(f"Using custom date range: {date_from} to {date_to}")
    
    # Extract raw data
    df_main = extractor.extract_sales_main(date_from=date_from, date_to=date_to)
    df_data = extractor.extract_sales_data(date_from=date_from, date_to=date_to)
    
    # Extract AI datasets
    df_customer = extractor.extract_customer_analysis_data(date_from=date_from, date_to=date_to)
    df_product = extractor.extract_product_analysis_data(date_from=date_from, date_to=date_to)
    df_trend = extractor.extract_sales_trend_data(date_from=date_from, date_to=date_to)
    df_retention = extractor.extract_customer_retention_data(lookback_days=None)
    df_revenue = extractor.extract_date_revenue_data(date_from=date_from, date_to=date_to)
    
    # Transform raw data
    df_main_clean = transformer.clean_sales_main(df_main)
    df_data_clean = transformer.clean_sales_data(df_data)
    
    # Transform AI datasets
    df_customer_clean = transformer.transform_customer_analysis(df_customer)
    try:
        df_product_clean = transformer.transform_product_analysis(df_product)
    except Exception as e:
        logger.error(f"Error in Product module: {str(e)}")
        df_product_clean = None
    df_trend_clean = transformer.transform_sales_trend(df_trend)
    df_retention_clean = transformer.transform_customer_retention(df_retention)
    try:
        df_revenue_clean = transformer.transform_date_revenue(df_revenue)
    except Exception as e:
        logger.error(f"Error in Revenue module: {str(e)}")
        df_revenue_clean = None
    
    # Save raw data
    transformer.save_transformed_data(df_main_clean, 'data/processed/sales_main.parquet')
    transformer.save_transformed_data(df_data_clean, 'data/processed/sales_data.parquet')
    
    # Save AI datasets
    if df_customer_clean is not None:
        transformer.save_transformed_data(df_customer_clean, 'data/processed/customer_analysis.parquet')
    if df_product_clean is not None:
        transformer.save_transformed_data(df_product_clean, 'data/processed/product_analysis.parquet')
    if df_trend_clean is not None:
        transformer.save_transformed_data(df_trend_clean, 'data/processed/sales_trend.parquet')
    if df_retention_clean is not None:
        transformer.save_transformed_data(df_retention_clean, 'data/processed/customer_retention.parquet')
    if df_revenue_clean is not None:
        transformer.save_transformed_data(df_revenue_clean, 'data/processed/revenue_report_by_date.parquet')
    
    extractor.close()
    
    logger.info(f"Extraction completed: {len(df_main)} main, {len(df_data)} detail records")
    
    # AUTO CREATE AI DATASETS AFTER EXTRACTION
    logger.info("🔄 Auto creating AI analysis datasets...")
    try:
        import sys
        import subprocess
        subprocess.run([sys.executable, "create_datasets.py"], check=True)
        logger.info("✅ All AI datasets created successfully!")
    except Exception as e:
        logger.error(f"❌ Error creating datasets: {e}")


def run_trend_analysis(args):
    """Run product trend analysis and generate top 10 potential products"""
    from src.extractors.sales_extractor import SalesExtractor
    from src.analysis.product_trend_analyzer import ProductTrendAnalyzer
    import json
    
    logger.info("=== Bắt đầu phân tích xu hướng sản phẩm ===")
    
    with SalesExtractor() as extractor:
        analyzer = ProductTrendAnalyzer(extractor)
        
        result = analyzer.analyze_product_trends(
            companyfn=args.companyfn,
            days_history=args.days,
            top_n=args.top
        )
        
        if result['status'] == 'success':
            print("\n" + "="*60)
            print("✅ PHÂN TÍCH XU HƯỚNG THÀNH CÔNG")
            print("="*60)
            print(f"📅 Thời gian phân tích: {result['analysis_period']['days_analyzed']} ngày")
            print(f"📊 Tổng sản phẩm phân tích: {result['total_products_analyzed']}")
            print(f"📈 Xu hướng thị trường chung: {result['overall_market_trend']['average_product_growth']}%")
            print("\n🏆 TOP 10 SẢN PHẨM TRIỂN VỌNG NHẤT:")
            print("-"*60)
            
            for idx, product in enumerate(result['top_potential_products'], 1):
                print(f"\n{idx}. {product['stkcode_desc']}")
                print(f"   Mã: {product['stkcode_code']} | Nhóm: {product['stkcate_desc']}")
                print(f"   ⭐ Điểm triển vọng: {product['potential_score']}/100")
                print(f"   📈 Tăng trưởng: {round(product['growth_rate']*100, 1)}% | Đà tăng: {round(product['momentum'], 2)}x")
                print(f"   💰 Doanh thu: {int(product['total_revenue']):,} | Đơn hàng: {product['transactions_count']}")
            
            # Lưu kết quả ra file
            output_file = f'data/analysis/product_trends_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            print("\n" + "-"*60)
            print(f"💾 Kết quả chi tiết đã được lưu tại: {output_file}")
            print("="*60)
            
        else:
            logger.warning("Không có dữ liệu để phân tích")
    
    return result


def run_training(args):
    """Run model training with MLflow experiment tracking"""
    from src.extractors.sales_extractor import SalesExtractor
    from src.transformers.data_transformer import DataTransformer
    from src.trainers.churn_predictor import ChurnPredictor
    from src.trainers.sales_forecaster import SalesForecaster
    from src.mlops.mlflow_tracker import MLFlowTracker
    
    logger.info("Starting model training with MLflow tracking")
    
    # Initialize MLflow tracker
    tracker = MLFlowTracker()
    
    with tracker:
        tracker.log_params({
            "model_type": args.model,
            "timestamp": datetime.now().isoformat()
        })
        
        extractor = SalesExtractor()
        transformer = DataTransformer()
        
        # Churn prediction
        if args.model in ['churn', 'all']:
            logger.info("Training churn prediction model")
            df_retention = extractor.extract_customer_retention_data()
            df_retention_clean = transformer.transform_customer_retention(df_retention)
            
            churn_predictor = ChurnPredictor()
            df_retention_prep = churn_predictor.prepare_churn_data(df_retention_clean)
            result = churn_predictor.train(df_retention_prep)
            
            logger.info(f"Churn model: F1={result['metrics']['f1_score']:.4f}")
            
            # Log to MLflow
            tracker.log_metrics({
                "churn_f1_score": result['metrics']['f1_score'],
                "churn_accuracy": result['metrics'].get('accuracy', 0),
                "churn_precision": result['metrics'].get('precision', 0),
                "churn_recall": result['metrics'].get('recall', 0)
            })
            
            tracker.log_model(
                model=churn_predictor.model,
                model_name="churn_predictor",
                registered_model_name="ChurnPredictor",
                metrics=result['metrics']
            )
        
        # Sales forecast
        if args.model in ['forecast', 'all']:
            logger.info("Training sales forecast model")
            df_trend = extractor.extract_sales_trend_data()
            df_trend_clean = transformer.transform_sales_trend(df_trend)
            
            forecaster = SalesForecaster()
            df_trend_prep = forecaster.prepare_forecast_data(df_trend_clean)
            result = forecaster.train(df_trend_prep)
            
            logger.info(f"Forecast model: R2={result['metrics']['r2']:.4f}")
            
            # Log to MLflow
            tracker.log_metrics({
                "forecast_r2": result['metrics']['r2'],
                "forecast_rmse": result['metrics'].get('rmse', 0),
                "forecast_mae": result['metrics'].get('mae', 0)
            })
            
            tracker.log_model(
                model=forecaster.model,
                model_name="sales_forecaster",
                registered_model_name="SalesForecaster",
                metrics=result['metrics']
            )
        
        extractor.close()
    
    logger.info("✅ Training completed. All data logged to MLflow.")
    logger.info("💡 Run `mlflow ui` to view experiment dashboard")


def run_query(args):
    """Run AI query"""
    from src.query.ai_query_interface import AIQueryInterface
    
    logger.info("Starting AI query")
    
    interface = AIQueryInterface(companyfn=args.companyfn)
    
    if args.interactive:
        # Interactive mode
        print("\n=== AI Query Interface ===")
        print("Enter questions about sales data (type 'quit' to exit)")
        print("Examples: 'Top customers by purchases', 'Monthly sales trends'")
        print()
        
        while True:
            try:
                query = input("Question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if not query:
                    continue
                
                result = interface.process_query(query)
                response = interface.format_response(result)
                print(f"\n{response}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
    else:
        # Single query
        query = args.query or "Sales overview"
        result = interface.process_query(query)
        response = interface.format_response(result)
        print(response)


def run_scheduled(args):
    """Run scheduled training"""
    from scripts.scheduled_training import main as scheduled_main
    
    logger.info("Starting scheduled training service")
    scheduled_main()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AI Training System - Sales Data Analysis System'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Extract command
    extract_parser = subparsers.add_parser('extract', help='Extract and transform data')
    extract_parser.add_argument('--date-from', help='Start date (YYYY-MM-DD)')
    extract_parser.add_argument('--date-to', help='End date (YYYY-MM-DD)')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--model', choices=['churn', 'forecast', 'all'], default='all',
                             help='Model to train')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query AI for insights')
    query_parser.add_argument('--query', '-q', help='Query string')
    query_parser.add_argument('--companyfn', '-c', help='Company code for data isolation')
    query_parser.add_argument('--interactive', '-i', action='store_true',
                             help='Interactive mode')
    
    # Trend Analysis command
    trend_parser = subparsers.add_parser('trend', help='Phân tích xu hướng sản phẩm và top 10 triển vọng')
    trend_parser.add_argument('--days', '-d', type=int, default=90, help='Số ngày lịch sử phân tích (mặc định 90)')
    trend_parser.add_argument('--top', '-t', type=int, default=10, help='Số sản phẩm top cần lấy (mặc định 10)')
    trend_parser.add_argument('--companyfn', '-c', help='Mã công ty (tùy chọn)')

    # Scheduled command
    scheduled_parser = subparsers.add_parser('scheduled', help='Run scheduled training')

    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'extract':
            run_extraction(args)
        elif args.command == 'train':
            run_training(args)
        elif args.command == 'query':
            run_query(args)
        elif args.command == 'trend':
            run_trend_analysis(args)
        elif args.command == 'scheduled':
            run_scheduled(args)
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()