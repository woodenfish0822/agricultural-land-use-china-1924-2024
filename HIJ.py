import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

# ==================== 配置区域 ====================
INPUT_FILE = r"D:\原E盘\大学\SRT\数据处理\数据修正\1\总数据库.xlsx"
OUTPUT_TABLE = "HJI_计算结果.xlsx"
OUTPUT_FIG_PREFIX = "HJI_趋势_"

# 地类名称映射（工作表名 -> 统一名称）
SHEET_MAP = {
    '耕': '耕地',
    '园': '园地',
    '林': '林地',
    '草': '牧草地',
    '水': '养捕水面',
    '设': '设施农用地'
}
N_CLASSES = 6
H_MAX = np.log(N_CLASSES)

# 关键年份
KEY_YEARS = [1924, 1949, 1978, 2000, 2024]

# 板块划分
REGIONS = {
    '东部': ['北京', '天津', '河北', '上海', '江苏', '浙江', '福建', '山东', '广东', '海南'],
    '中部': ['山西', '安徽', '江西', '河南', '湖北', '湖南'],
    '西部': ['内蒙古', '广西', '重庆', '四川', '贵州', '云南', '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'],
    '东北': ['辽宁', '吉林', '黑龙江']
}

# ==================== 工具函数 ====================
def calculate_hji(areas):
    """
    输入六类面积（数组或列表），返回 (H, J, I)
    增加类型检查和调试信息
    """
    try:
        # 强制转换为浮点数组
        areas = np.asarray(areas, dtype=float)
        if np.any(np.isnan(areas)):
            print(f"警告：数据包含NaN: {areas}")
            return np.nan, np.nan, np.nan
        total = np.sum(areas)
        if total <= 0 or not np.isfinite(total):
            return np.nan, np.nan, np.nan
        p = areas / total
        positive = p[p > 0]
        if len(positive) == 0:
            return np.nan, np.nan, np.nan
        h = -np.sum(positive * np.log(positive))
        j = h / H_MAX
        i = H_MAX - h
        return h, j, i
    except Exception as e:
        print(f"计算HJI时出错，输入数据: {areas}")
        raise e

# ==================== 读取数据 ====================
print("正在读取Excel文件...")
data_frames = {}
for sheet, land_name in SHEET_MAP.items():
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet, index_col=0)
    df.columns = df.columns.astype(int)
    df.index = df.index.str.strip()
    data_frames[land_name] = df

provinces = data_frames['耕地'].index.tolist()
years = data_frames['耕地'].columns.tolist()
print(f"读取完成，共有 {len(provinces)} 个省份/地区，年份范围 {min(years)}-{max(years)}")

# 构建长表
long_list = []
for land_name, df in data_frames.items():
    long = df.stack().reset_index()
    long.columns = ['省份', '年份', land_name]
    long_list.append(long)

df_all = reduce(lambda left, right: pd.merge(left, right, on=['省份','年份'], how='outer'), long_list)
for col in SHEET_MAP.values():
    df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

# ==================== 1. 计算各省份每年的HJI ====================
print("正在计算各省份HJI...")
hji_list = []
for (province, year), group in df_all.groupby(['省份', '年份']):
    areas = group.iloc[0][list(SHEET_MAP.values())].values
    h, j, i = calculate_hji(areas)
    hji_list.append({
        '年份': year,
        '省份': province,
        'H': h,
        'J': j,
        'I': i
    })
df_province_hji = pd.DataFrame(hji_list)

# ==================== 2. 计算全国每年的HJI ====================
print("正在计算全国HJI...")
national_data = df_all[df_all['省份'] == '全国总计'].copy()
national_hji = []
for _, row in national_data.iterrows():
    year = row['年份']
    areas = row[list(SHEET_MAP.values())].values
    h, j, i = calculate_hji(areas)
    national_hji.append({
        '年份': year,
        '省份': '全国',
        'H': h,
        'J': j,
        'I': i
    })
df_national_hji = pd.DataFrame(national_hji)

df_all_hji = pd.concat([df_province_hji, df_national_hji], ignore_index=True)
df_all_hji.to_excel(OUTPUT_TABLE, index=False)
print(f"HJI计算结果已保存至 {OUTPUT_TABLE}")

# ==================== 3. 计算各板块每年的HJI ====================
print("正在计算四大板块HJI...")
province_data = df_all[df_all['省份'] != '全国总计'].copy()

region_hji_list = []
for region_name, province_list in REGIONS.items():
    region_df = province_data[province_data['省份'].isin(province_list)]
    region_yearly = region_df.groupby('年份')[list(SHEET_MAP.values())].sum().reset_index()
    for _, row in region_yearly.iterrows():
        year = row['年份']
        areas = row[list(SHEET_MAP.values())].values
        h, j, i = calculate_hji(areas)
        region_hji_list.append({
            '年份': year,
            '板块': region_name,
            'H': h,
            'J': j,
            'I': i
        })

df_region_hji = pd.DataFrame(region_hji_list)

# 将全国数据加入板块数据（列名统一为['年份','板块','H','J','I']）
df_national_for_region = df_national_hji.rename(columns={'省份': '板块'})
df_region_hji = pd.concat([df_region_hji, df_national_for_region], ignore_index=True)

# ==================== 4. 提取关键年份数据并绘图 ====================
print("正在绘制关键年份趋势图...")
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

key_data = df_region_hji[df_region_hji['年份'].isin(KEY_YEARS)].copy()
key_data.sort_values(['板块', '年份'], inplace=True)

all_regions = key_data['板块'].unique()

for region in all_regions:
    region_data = key_data[key_data['板块'] == region]
    if region_data.empty:
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(region_data['年份'], region_data['H'], marker='o', label='信息熵 H', linewidth=2)
    ax.plot(region_data['年份'], region_data['J'], marker='s', label='均衡度 J', linewidth=2)
    ax.plot(region_data['年份'], region_data['I'], marker='^', label='优势度 I', linewidth=2)

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title(f'{region}地区农用地结构HJI趋势（关键年份）', fontsize=14)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(KEY_YEARS)
    ax.set_xticklabels(KEY_YEARS, rotation=45)

    plt.tight_layout()
    output_file = f"{OUTPUT_FIG_PREFIX}{region}.png"
    plt.savefig(output_file, dpi=300)
    plt.show()
    print(f"已保存 {region} 趋势图至 {output_file}")

# ==================== 5. 简单分析报告 ====================
print("\n" + "="*50)
print("简单分析报告")
print("="*50)

national_key = key_data[key_data['板块']=='全国'].set_index('年份')
if not national_key.empty:
    h_start = national_key.loc[KEY_YEARS[0], 'H']
    h_end = national_key.loc[KEY_YEARS[-1], 'H']
    j_start = national_key.loc[KEY_YEARS[0], 'J']
    j_end = national_key.loc[KEY_YEARS[-1], 'J']
    i_start = national_key.loc[KEY_YEARS[0], 'I']
    i_end = national_key.loc[KEY_YEARS[-1], 'I']
    print(f"全国层面：从{KEY_YEARS[0]}年到{KEY_YEARS[-1]}年，")
    print(f"  信息熵H由{h_start:.3f}变化至{h_end:.3f}，{ '上升' if h_end>h_start else '下降' }了{abs(h_end-h_start):.3f}；")
    print(f"  均衡度J由{j_start:.3f}变化至{j_end:.3f}，{ '上升' if j_end>j_start else '下降' }了{abs(j_end-j_start):.3f}；")
    print(f"  优势度I由{i_start:.3f}变化至{i_end:.3f}，{ '上升' if i_end>i_start else '下降' }了{abs(i_end-i_start):.3f}。")

print("\n各板块关键年份指标对比：")
for year in KEY_YEARS:
    year_data = key_data[key_data['年份']==year]
    if not year_data.empty:
        print(f"\n{year}年：")
        for _, row in year_data.iterrows():
            print(f"  {row['板块']}: H={row['H']:.3f}, J={row['J']:.3f}, I={row['I']:.3f}")

print("\n注：以上分析基于关键年份数据，详细趋势请参考折线图。")
print("="*50)