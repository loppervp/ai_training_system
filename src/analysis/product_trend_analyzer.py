"""
Product Trend Analyzer Module
Phân tích xu hướng bán hàng và dự đoán top 10 sản phẩm triển vọng dựa trên dữ liệu đơn hàng lịch sử
"""
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class ProductTrendAnalyzer:
    """
    Class phân tích trend sản phẩm và dự đoán tiềm năng trong tương lai
    """
    
    def __init__(self, sales_extractor):
        """
        Khởi tạo analyzer với SalesExtractor đã kết nối
        """
        self.sales_extractor = sales_extractor
        self.weight_factors = {
            'recent_growth': 0.35,      # Tốc độ tăng trưởng gần đây
            'momentum': 0.25,           # Đà tăng trong 30 ngày qua
            'volume_consistency': 0.20, # Độ ổn định doanh số
            'profitability': 0.15,      # Lợi nhuận
            'new_customer_rate': 0.05   # Tỷ lệ khách hàng mới mua sản phẩm
        }
    
    def analyze_product_trends(self, 
                              companyfn: Optional[str] = None,
                              days_history: int = 90,
                              top_n: int = 10) -> Dict:
        """
        Phân tích toàn bộ sản phẩm và trả về top N sản phẩm triển vọng nhất
        
        Args:
            companyfn: Mã công ty
            days_history: Số ngày lịch sử phân tích
            top_n: Số lượng sản phẩm top cần trả về
            
        Returns:
            Dictionary kết quả phân tích
        """
        logger.info(f"Bắt đầu phân tích trend sản phẩm với {days_history} ngày lịch sử")
        
        # Tính khoảng ngày
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_history)
        
        # Lấy dữ liệu bán hàng chi tiết
        sales_data = self.sales_extractor.extract_sales_data(
            companyfn=companyfn,
            date_from=start_date.strftime("%Y-%m-%d"),
            date_to=end_date.strftime("%Y-%m-%d"),
            include_void=False
        )
        
        if sales_data.empty:
            logger.warning("Không tìm thấy dữ liệu bán hàng trong khoảng thời gian này")
            return {
                'status': 'no_data',
                'products': [],
                'analysis_period': {
                    'from': start_date.isoformat(),
                    'to': end_date.isoformat()
                }
            }
        
        logger.info(f"Đã lấy {len(sales_data)} dòng dữ liệu bán hàng")
        
        # Nhóm dữ liệu theo sản phẩm
        product_stats = self._calculate_product_statistics(sales_data, days_history)
        
        # Tính điểm triển vọng cho mỗi sản phẩm
        scored_products = self._calculate_potential_score(product_stats)
        
        # Sắp xếp và lấy top N
        scored_products.sort(key=lambda x: x['potential_score'], reverse=True)
        top_products = scored_products[:top_n]
        
        # Tính các chỉ số tổng quan
        overall_stats = self._calculate_overall_stats(product_stats)
        
        result = {
            'status': 'success',
            'analysis_period': {
                'from': start_date.isoformat(),
                'to': end_date.isoformat(),
                'days_analyzed': days_history
            },
            'total_products_analyzed': len(product_stats),
            'top_potential_products': top_products,
            'overall_market_trend': overall_stats,
            'factor_weights': self.weight_factors,
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"Phân tích hoàn tất. Tìm thấy {len(top_products)} sản phẩm triển vọng hàng đầu")
        return result
    
    def _calculate_product_statistics(self, sales_data: pd.DataFrame, days_history: int) -> Dict:
        """
        Tính toán các chỉ số thống kê cho từng sản phẩm
        """
        product_map = defaultdict(lambda: {
            'total_quantity': 0,
            'total_revenue': 0,
            'transactions_count': 0,
            'daily_sales': defaultdict(int),
            'weekly_sales': [],
            'unique_customers': set(),
            'first_sale_date': None,
            'last_sale_date': None,
            'price_history': []
        })
        
        mid_point = days_history // 2
        current_date = datetime.now()
        
        for _, row in sales_data.iterrows():
            product_key = row['stkcode_unique']
            product = product_map[product_key]
            
            # Thông tin cơ bản
            product['stkcode_code'] = row['stkcode_code']
            product['stkcode_desc'] = row['stkcode_desc']
            product['brand_desc'] = row.get('brand_desc', '')
            product['stkcate_desc'] = row.get('stkcate_desc', '')
            
            # Doanh số
            quantity = float(row['qnty_total']) if pd.notna(row['qnty_total']) else 0
            revenue = quantity * (float(row['price_unitlist_local']) if pd.notna(row['price_unitlist_local']) else 0)
            
            product['total_quantity'] += quantity
            product['total_revenue'] += revenue
            product['transactions_count'] += 1
            
            # Khách hàng
            if 'party_unique' in row and pd.notna(row['party_unique']):
                product['unique_customers'].add(row['party_unique'])
            
            # Ngày giao dịch
            trans_date = pd.to_datetime(row['date_trans']) if 'date_trans' in row else current_date
            date_key = trans_date.strftime("%Y-%m-%d")
            product['daily_sales'][date_key] += quantity
            
            if product['first_sale_date'] is None or trans_date < product['first_sale_date']:
                product['first_sale_date'] = trans_date
            if product['last_sale_date'] is None or trans_date > product['last_sale_date']:
                product['last_sale_date'] = trans_date
            
            if pd.notna(row['price_unitlist_local']):
                product['price_history'].append(float(row['price_unitlist_local']))
        
        # Tính các chỉ số phái sinh
        for product_key, product in product_map.items():
            # Tính tăng trưởng nửa kỳ đầu và nửa kỳ cuối
            daily_sales = sorted(product['daily_sales'].items())
            split_point = len(daily_sales) // 2
            
            first_half_qty = sum(qty for date, qty in daily_sales[:split_point])
            second_half_qty = sum(qty for date, qty in daily_sales[split_point:])
            
            if first_half_qty > 0:
                product['growth_rate'] = (second_half_qty - first_half_qty) / first_half_qty
            else:
                product['growth_rate'] = 1.0 if second_half_qty > 0 else 0
            
            # Tính đà tăng (momentum) 30 ngày gần nhất
            recent_days = 30
            recent_cutoff = current_date - timedelta(days=recent_days)
            recent_qty = sum(qty for date, qty in daily_sales if pd.to_datetime(date) >= recent_cutoff)
            older_qty = product['total_quantity'] - recent_qty
            
            if older_qty > 0:
                product['momentum'] = (recent_qty / recent_days) / (older_qty / max(days_history - recent_days, 1))
            else:
                product['momentum'] = 2.0 if recent_qty > 0 else 0
            
            # Độ ổn định (coefficient of variation)
            daily_values = list(product['daily_sales'].values())
            if len(daily_values) > 1:
                mean_sales = np.mean(daily_values)
                std_sales = np.std(daily_values)
                product['consistency'] = 1 / (1 + (std_sales / mean_sales if mean_sales > 0 else 1))
            else:
                product['consistency'] = 0.5
            
            # Số khách hàng độc nhất
            product['unique_customer_count'] = len(product['unique_customers'])
            del product['unique_customers']
            
            # Giá trung bình
            if product['price_history']:
                product['avg_price'] = np.mean(product['price_history'])
            del product['price_history']
            del product['daily_sales']
        
        return product_map
    
    def _calculate_potential_score(self, product_stats: Dict) -> List[Dict]:
        """
        Tính điểm triển vọng tổng hợp cho mỗi sản phẩm
        """
        scored = []
        
        # Chuẩn hóa tất cả các chỉ số về thang điểm 0-1
        all_products = list(product_stats.values())
        
        # Lấy giá trị max để chuẩn hóa
        max_growth = max((p['growth_rate'] for p in all_products), default=1)
        max_momentum = max((p['momentum'] for p in all_products), default=1)
        max_revenue = max((p['total_revenue'] for p in all_products), default=1)
        
        for product in all_products:
            # Chuẩn hóa từng yếu tố
            norm_growth = min(product['growth_rate'] / max_growth if max_growth > 0 else 0, 1)
            norm_momentum = min(product['momentum'] / max_momentum if max_momentum > 0 else 0, 1)
            norm_consistency = product['consistency']
            norm_volume = min(product['total_revenue'] / max_revenue if max_revenue > 0 else 0, 1)
            norm_customer = min(product['unique_customer_count'] / 50, 1)
            
            # Tính điểm tổng hợp theo trọng số
            total_score = (
                norm_growth * self.weight_factors['recent_growth'] +
                norm_momentum * self.weight_factors['momentum'] +
                norm_consistency * self.weight_factors['volume_consistency'] +
                norm_volume * self.weight_factors['profitability'] +
                norm_customer * self.weight_factors['new_customer_rate']
            )
            
            product['potential_score'] = round(total_score * 100, 2)
            product['factor_breakdown'] = {
                'recent_growth': round(norm_growth * 100, 1),
                'momentum': round(norm_momentum * 100, 1),
                'consistency': round(norm_consistency * 100, 1),
                'volume': round(norm_volume * 100, 1),
                'customer_base': round(norm_customer * 100, 1)
            }
            
            scored.append(product)
        
        return scored
    
    def _calculate_overall_stats(self, product_stats: Dict) -> Dict:
        """
        Tính các chỉ số xu hướng thị trường tổng thể
        """
        if not product_stats:
            return {}
            
        all_products = list(product_stats.values())
        
        avg_growth = np.mean([p['growth_rate'] for p in all_products])
        positive_growth_count = sum(1 for p in all_products if p['growth_rate'] > 0)
        declining_count = sum(1 for p in all_products if p['growth_rate'] < 0)
        
        return {
            'average_product_growth': round(avg_growth * 100, 2),
            'growing_products_count': positive_growth_count,
            'declining_products_count': declining_count,
            'total_products': len(all_products),
            'market_momentum': round((positive_growth_count / len(all_products)) * 100, 1)
        }