import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geodetector import factor_detector, interaction_detector
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class AgriculturalLandAnalysis:
    def __init__(self, file_path):
        """
        初始化分析类
        """
        self.file_path = file_path
        self.data = None
        self.provinces = [
            '全国总计', '北京', '天津', '河北', '山西', '内蒙古', '辽宁', '吉林', '黑龙江',
            '上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '河南', '湖北', '湖南',
            '广东', '广西', '海南', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃',
            '青海', '宁夏', '新疆'
        ]
        self.variables = [
            '信息熵_H', '年均温', '年降水量', '人均GDP', '城镇化率', 
            '经济增长非农占比', '化肥投入强度', '农业机械化水平', '灌溉保障水平', '政策因子'
        ]
        self.years = list(range(1924, 2025))
        
    def load_and_process_data(self):
        """
        加载和处理数据
        """
        print("正在加载数据...")
        
        # 读取Excel文件
        df = pd.read_excel(self.file_path, header=None)
        
        # 创建空的数据框
        all_data = []
        
        # 定义每个省份的列范围
        province_cols = {}
        start_col = 1  # B列是1 (0-indexed)
        
        for i, province in enumerate(self.provinces):
            province_cols[province] = list(range(start_col, start_col + 10))
            start_col += 10
        
        # 提取每个省份的数据
        for province, cols in province_cols.items():
            # 提取该省份的数据 (第3行到第103行，对应1924-2024年)
            province_data = df.iloc[2:103, cols].copy()  # 2:103 对应第3-103行
            
            # 设置列名
            province_data.columns = self.variables
            
            # 添加省份和年份列
            province_data['省份'] = province
            province_data['年份'] = self.years
            
            all_data.append(province_data)
        
        # 合并所有数据
        self.data = pd.concat(all_data, ignore_index=True)
        
        print(f"数据加载完成！共 {len(self.data)} 行记录")
        print(f"省份数量: {self.data['省份'].nunique()}")
        print(f"年份范围: {self.data['年份'].min()} - {self.data['年份'].max()}")
        
        return self.data
    
    def data_exploration(self):
        """
        数据探索和描述性统计
        """
        print("\n=== 数据探索 ===")
        
        # 检查缺失值
        missing_data = self.data.isnull().sum()
        print("缺失值统计:")
        print(missing_data)
        
        # 描述性统计
        print("\n描述性统计:")
        desc_stats = self.data[self.variables].describe()
        print(desc_stats)
        
        # 保存描述性统计
        desc_stats.to_excel('描述性统计.xlsx')
        
        return desc_stats
    
    def data_standardization(self):
        """
        数据标准化处理
        """
        print("\n=== 数据标准化 ===")
        
        # 需要标准化的变量（排除政策因子和因变量）
        variables_to_standardize = [
            '年均温', '年降水量', '人均GDP', '城镇化率', 
            '经济增长非农占比', '化肥投入强度', '农业机械化水平', '灌溉保障水平'
        ]
        
        # 标准化处理 (Z-score标准化)
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        self.data_std = self.data.copy()
        
        for var in variables_to_standardize:
            self.data_std[f'{var}_标准化'] = scaler.fit_transform(
                self.data_std[[var]].fillna(self.data_std[var].mean())
            )
        
        print("数据标准化完成！")
        
        return self.data_std
    
    def temporal_analysis(self):
        """
        时空演变分析 - 计算重心迁移轨迹
        注意：由于缺少具体的地类面积数据，这里提供框架代码
        """
        print("\n=== 时空演变分析 ===")
        
        # 这里需要您提供六类农用地的具体面积数据
        # 如果数据不可用，可以跳过此部分或使用现有变量进行分析
        
        # 示例：计算信息熵的时空变化
        entropy_by_year = self.data.groupby('年份')['信息熵_H'].agg(['mean', 'std']).reset_index()
        
        # 绘制信息熵随时间变化
        plt.figure(figsize=(12, 6))
        plt.plot(entropy_by_year['年份'], entropy_by_year['mean'], linewidth=2)
        plt.fill_between(entropy_by_year['年份'], 
                        entropy_by_year['mean'] - entropy_by_year['std'],
                        entropy_by_year['mean'] + entropy_by_year['std'], 
                        alpha=0.3)
        plt.title('信息熵随时间变化趋势 (1924-2024)')
        plt.xlabel('年份')
        plt.ylabel('信息熵')
        plt.grid(True, alpha=0.3)
        plt.savefig('信息熵随时间变化.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("时空分析完成！")
        
        return entropy_by_year
    
    def define_historical_periods(self):
        """
        定义历史阶段
        """
        periods = {
            '1924-1949': (1924, 1949),
            '1950-1977': (1950, 1977),
            '1978-1999': (1978, 1999),
            '2000-2012': (2000, 2012),
            '2013-2024': (2013, 2024)
        }
        return periods
    
    def run_geodetector_analysis(self):
        """
        运行地理探测器分析
        """
        print("\n=== 地理探测器分析 ===")
        
        periods = self.define_historical_periods()
        
        # 定义自变量和因变量
        independent_vars = [
            '年均温', '年降水量', '人均GDP', '城镇化率', 
            '经济增长非农占比', '化肥投入强度', '农业机械化水平', 
            '灌溉保障水平', '政策因子'
        ]
        
        dependent_var = '信息熵_H'
        
        all_results = {}
        
        for period_name, (start_year, end_year) in periods.items():
            print(f"\n分析时期: {period_name}")
            
            # 筛选时期数据
            period_data = self.data_std[
                (self.data_std['年份'] >= start_year) & 
                (self.data_std['年份'] <= end_year)
            ].copy()
            
            if len(period_data) == 0:
                print(f"时期 {period_name} 无数据，跳过")
                continue
            
            # 准备地理探测器输入数据
            # 对连续变量进行离散化处理（地理探测器要求）
            for var in independent_vars:
                if var != '政策因子':  # 政策因子已经是分类变量
                    # 使用分位数离散化
                    period_data[f'{var}_离散'] = pd.qcut(
                        period_data[var], 
                        q=5, 
                        labels=False, 
                        duplicates='drop'
                    )
                else:
                    period_data[f'{var}_离散'] = period_data[var]
            
            try:
                # 因子探测
                factor_detection = geodetector.factor_detector(
                    period_data[[f'{var}_离散' for var in independent_vars]],
                    period_data[dependent_var]
                )
                
                # 交互探测
                interaction_detection = geodetector.interaction_detector(
                    period_data[[f'{var}_离散' for var in independent_vars]],
                    period_data[dependent_var]
                )
                
                all_results[period_name] = {
                    'factor': factor_detection,
                    'interaction': interaction_detection
                }
                
                print(f"时期 {period_name} 分析完成")
                
            except Exception as e:
                print(f"时期 {period_name} 分析出错: {e}")
                continue
        
        return all_results
    
    def visualize_geodetector_results(self, results):
        """
        可视化地理探测器结果
        """
        print("\n=== 可视化结果 ===")
        
        periods = self.define_historical_periods()
        independent_vars = [
            '年均温', '年降水量', '人均GDP', '城镇化率', 
            '经济增长非农占比', '化肥投入强度', '农业机械化水平', 
            '灌溉保障水平', '政策因子'
        ]
        
        # 1. 各时期因子q值柱状图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (period_name, period_results) in enumerate(results.items()):
            if i >= len(axes):
                break
                
            if period_results:
                q_values = period_results['factor']['q_stat']
                
                ax = axes[i]
                bars = ax.bar(range(len(q_values)), q_values)
                ax.set_title(f'{period_name} 因子q值')
                ax.set_xlabel('因子')
                ax.set_ylabel('q值')
                ax.set_xticks(range(len(independent_vars)))
                ax.set_xticklabels(independent_vars, rotation=45)
                
                # 在柱子上添加数值
                for bar, value in zip(bars, q_values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 隐藏多余的子图
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('各时期因子q值.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 交互作用热力图
        for period_name, period_results in results.items():
            if period_results:
                interaction_matrix = period_results['interaction']
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(interaction_matrix, 
                           annot=True, 
                           fmt='.3f',
                           cmap='YlOrRd',
                           xticklabels=independent_vars,
                           yticklabels=independent_vars)
                plt.title(f'{period_name} 因子交互作用热力图')
                plt.xticks(rotation=45)
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.savefig(f'{period_name}_交互作用热力图.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print("可视化完成！")
    
    def export_results(self, results):
        """
        导出所有结果
        """
        print("\n=== 导出结果 ===")
        
        # 导出处理后的数据
        self.data_std.to_excel('标准化数据.xlsx', index=False)
        
        # 导出地理探测器结果
        with pd.ExcelWriter('地理探测器结果.xlsx') as writer:
            for period_name, period_results in results.items():
                if period_results:
                    # 因子探测结果
                    factor_df = pd.DataFrame({
                        '因子': [
                            '年均温', '年降水量', '人均GDP', '城镇化率', 
                            '经济增长非农占比', '化肥投入强度', '农业机械化水平', 
                            '灌溉保障水平', '政策因子'
                        ],
                        'q值': period_results['factor']['q_stat']
                    })
                    factor_df.to_excel(writer, sheet_name=f'{period_name}_因子探测', index=False)
                    
                    # 交互探测结果
                    interaction_df = pd.DataFrame(
                        period_results['interaction'],
                        index=[
                            '年均温', '年降水量', '人均GDP', '城镇化率', 
                            '经济增长非农占比', '化肥投入强度', '农业机械化水平', 
                            '灌溉保障水平', '政策因子'
                        ],
                        columns=[
                            '年均温', '年降水量', '人均GDP', '城镇化率', 
                            '经济增长非农占比', '化肥投入强度', '农业机械化水平', 
                            '灌溉保障水平', '政策因子'
                        ]
                    )
                    interaction_df.to_excel(writer, sheet_name=f'{period_name}_交互探测')
        
        print("结果导出完成！")
    
    def run_complete_analysis(self):
        """
        运行完整分析流程
        """
        print("开始完整的农用地分析流程...")
        
        # 1. 数据加载与处理
        self.load_and_process_data()
        
        # 2. 数据探索
        self.data_exploration()
        
        # 3. 数据标准化
        self.data_standardization()
        
        # 4. 时空演变分析
        self.temporal_analysis()
        
        # 5. 地理探测器分析
        results = self.run_geodetector_analysis()
        
        # 6. 结果可视化
        self.visualize_geodetector_results(results)
        
        # 7. 导出结果
        self.export_results(results)
        
        print("\n=== 分析完成！ ===")
        print("生成的文件:")
        print("- 描述性统计.xlsx")
        print("- 标准化数据.xlsx")
        print("- 信息熵随时间变化.png")
        print("- 各时期因子q值.png")
        print("- 各时期交互作用热力图.png")
        print("- 地理探测器结果.xlsx")

# 主程序
if __name__ == "__main__":
    # 设置文件路径 - 请修改为您的实际文件路径
    file_path = r'D:\原E盘\大学\SRT\影响因子处理\影响因子+H值.xlsx'
    
    # 创建分析实例
    analyzer = AgriculturalLandAnalysis(file_path)
    
    # 运行完整分析
    analyzer.run_complete_analysis()