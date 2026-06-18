import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class StageGeodetectorAnalysis:
    """
    专门用于分阶段地理探测器分析的类
    """
    
    def __init__(self, file_path):
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
        """加载和处理数据"""
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
            province_data = df.iloc[2:103, cols].copy()
            
            # 设置列名
            province_data.columns = self.variables
            
            # 添加省份和年份列
            province_data['省份'] = province
            province_data['年份'] = self.years
            
            all_data.append(province_data)
        
        # 合并所有数据
        self.data = pd.concat(all_data, ignore_index=True)
        
        print(f"数据加载完成！共 {len(self.data)} 行记录")
        return self.data
    
    def define_historical_periods(self):
        """定义五个历史阶段"""
        return {
            '1924-1949': (1924, 1949),
            '1950-1977': (1950, 1977),
            '1978-1999': (1978, 1999),
            '2000-2012': (2000, 2012),
            '2013-2024': (2013, 2024)
        }
    
    def calculate_q_value(self, x, y):
        """
        计算单个因子的q值
        q = 1 - (SSW / SST)
        其中SSW是层内方差，SST是总方差
        """
        # 确保数据没有缺失值
        valid_mask = ~x.isnull() & ~y.isnull()
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) == 0:
            return 0
        
        # 对连续变量进行离散化（5类）
        if x_clean.nunique() > 5:
            x_discrete = pd.qcut(x_clean, q=5, labels=False, duplicates='drop')
        else:
            x_discrete = x_clean
        
        # 分组计算
        groups = y_clean.groupby(x_discrete)
        
        # 计算层内方差和(SSW)和总方差(SST)
        SSW = 0
        for name, group in groups:
            SSW += len(group) * group.var()
        
        SST = len(y_clean) * y_clean.var()
        
        # 计算q值
        if SST == 0:
            q = 0
        else:
            q = 1 - (SSW / SST)
        
        return max(0, min(q, 1))  # 确保q值在0-1范围内
    
    def analyze_stage_q_values(self):
        """分析五个阶段各影响因子的q值"""
        print("开始分阶段地理探测器分析...")
        
        periods = self.define_historical_periods()
        
        # 定义自变量（排除因变量信息熵_H）
        independent_vars = [
            '年均温', '年降水量', '人均GDP', '城镇化率', 
            '经济增长非农占比', '化肥投入强度', '农业机械化水平', 
            '灌溉保障水平', '政策因子'
        ]
        
        dependent_var = '信息熵_H'
        
        # 存储各阶段结果
        stage_results = {}
        
        for period_name, (start_year, end_year) in periods.items():
            print(f"\n分析时期: {period_name}")
            
            # 筛选时期数据
            period_data = self.data[
                (self.data['年份'] >= start_year) & 
                (self.data['年份'] <= end_year)
            ].copy()
            
            if len(period_data) == 0:
                print(f"时期 {period_name} 无数据，跳过")
                continue
            
            # 计算各因子的q值
            q_values = {}
            for var in independent_vars:
                q_val = self.calculate_q_value(period_data[var], period_data[dependent_var])
                q_values[var] = q_val
                print(f"  {var}: q = {q_val:.4f}")
            
            stage_results[period_name] = q_values
        
        return stage_results
    
    def visualize_stage_q_values(self, stage_results):
        """可视化各阶段q值结果"""
        print("\n正在生成可视化结果...")
        
        periods = list(stage_results.keys())
        variables = list(next(iter(stage_results.values())).keys())
        
        # 创建数据框用于绘图
        q_df = pd.DataFrame(stage_results).T
        
        # 1. 各阶段q值热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(q_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'q值'}, vmin=0, vmax=1)
        plt.title('各历史阶段影响因子q值热力图')
        plt.tight_layout()
        plt.savefig('各阶段q值热力图.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 各阶段q值柱状图对比
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(variables)))
        
        for i, (period, q_values) in enumerate(stage_results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            bars = ax.bar(range(len(variables)), list(q_values.values()), 
                         color=colors, alpha=0.8)
            ax.set_title(f'{period}时期因子q值', fontsize=14, fontweight='bold')
            ax.set_xlabel('影响因素')
            ax.set_ylabel('q值')
            ax.set_xticks(range(len(variables)))
            ax.set_xticklabels(variables, rotation=45, ha='right')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # 在柱子上添加数值
            for bar, value in zip(bars, q_values.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 隐藏多余的子图
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('各阶段q值柱状图.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 各因子q值随时间变化趋势图
        plt.figure(figsize=(14, 8))
        
        # 为每个变量绘制趋势线
        for var in variables:
            q_trend = [stage_results[period][var] for period in periods]
            plt.plot(periods, q_trend, 'o-', linewidth=2, markersize=8, label=var)
        
        plt.title('各影响因子q值历史演变趋势', fontsize=16, fontweight='bold')
        plt.xlabel('历史时期')
        plt.ylabel('q值')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig('q值历史演变趋势.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("可视化完成！")
        
        return q_df
    
    def export_results(self, stage_results, q_df):
        """导出分析结果"""
        print("\n正在导出结果...")
        
        # 1. 导出详细结果到Excel
        with pd.ExcelWriter('分阶段q值分析结果.xlsx') as writer:
            # 各阶段q值汇总表
            summary_df = pd.DataFrame(stage_results)
            summary_df.to_excel(writer, sheet_name='各阶段q值汇总')
            
            # 各阶段详细统计
            for period, q_values in stage_results.items():
                period_df = pd.DataFrame({
                    '影响因素': list(q_values.keys()),
                    'q值': list(q_values.values()),
                    '解释力排名': pd.Series(list(q_values.values())).rank(ascending=False).astype(int)
                })
                period_df = period_df.sort_values('q值', ascending=False)
                period_df.to_excel(writer, sheet_name=f'{period}详细结果', index=False)
        
        # 2. 导出q值数据框
        q_df.to_excel('各阶段q值矩阵.xlsx')
        
        # 3. 生成分析报告
        self.generate_analysis_report(stage_results)
        
        print("结果导出完成！")
    
    def generate_analysis_report(self, stage_results):
        """生成简要分析报告"""
        report_lines = []
        report_lines.append("农用地结构影响因素分阶段地理探测器分析报告")
        report_lines.append("=" * 50)
        report_lines.append(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"数据范围: 1924-2024年，{len(self.provinces)}个省份")
        report_lines.append("")
        
        for period, q_values in stage_results.items():
            report_lines.append(f"{period}时期分析结果:")
            report_lines.append("-" * 30)
            
            # 按q值排序
            sorted_factors = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
            
            for i, (factor, q_val) in enumerate(sorted_factors, 1):
                strength = "强" if q_val > 0.5 else "中等" if q_val > 0.3 else "弱"
                report_lines.append(f"  {i}. {factor}: q={q_val:.4f} ({strength}解释力)")
            
            # 识别主导因素
            top_factors = [factor for factor, q_val in sorted_factors if q_val > 0.3]
            if top_factors:
                report_lines.append(f"  主导因素: {', '.join(top_factors)}")
            report_lines.append("")
        
        # 保存报告
        with open('分析报告.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 打印报告摘要
        print("\n分析报告摘要:")
        print("=" * 40)
        for line in report_lines[:10]:
            print(line)
    
    def run_complete_analysis(self):
        """运行完整的分阶段分析"""
        print("开始分阶段地理探测器分析...")
        print("=" * 50)
        
        # 1. 加载数据
        self.load_and_process_data()
        
        # 2. 分阶段计算q值
        stage_results = self.analyze_stage_q_values()
        
        if not stage_results:
            print("没有获得有效结果，请检查数据！")
            return
        
        # 3. 可视化结果
        q_df = self.visualize_stage_q_values(stage_results)
        
        # 4. 导出结果
        self.export_results(stage_results, q_df)
        
        print("\n" + "=" * 50)
        print("分阶段地理探测器分析完成！")
        print("生成的文件:")
        print("- 分阶段q值分析结果.xlsx")
        print("- 各阶段q值矩阵.xlsx")
        print("- 各阶段q值热力图.png")
        print("- 各阶段q值柱状图.png")
        print("- q值历史演变趋势.png")
        print("- 分析报告.txt")

# 主程序
def main():
    """主运行函数"""
    # 请修改为您的实际文件路径
    file_path = r'D:\原E盘\大学\SRT\影响因子处理\影响因子+H值.xlsx'
    
    try:
        print("农用地结构影响因素分阶段分析")
        print("专注于五个历史阶段的q值计算")
        
        # 创建分析器实例
        analyzer = StageGeodetectorAnalysis(file_path)
        
        # 运行分析
        analyzer.run_complete_analysis()
        
        print("\n分析成功完成！")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        print("请检查文件路径是否正确")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()