import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from ast import literal_eval
import json
from collections import Counter
import time
from tqdm import tqdm

start = time.perf_counter()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

age = []
gender = []
income = []
categories = []
price = []
country = []
active = []

lens = 0
dup_count=0
missing_values=0
age_anomalies = 0
income_anomalies = 0
gender_anomalies = 0


# 异常值检测函数
def detect_anomalies(series, min_val=None, max_val=None, allowed_values=None):
    anomalies = 0
    if min_val is not None or max_val is not None:
        anomalies += series[(series < min_val) | (series > max_val)].count()
    if allowed_values is not None:
        anomalies += series[~series.isin(allowed_values)].count()
    return anomalies

# 数据读取
for i in tqdm(range(0,8)):
    df = pd.read_parquet("./10G_data_new/part-0000"+str(i)+".parquet")
    lens += len(df)
    # 删除重复值
    dup_count += df.duplicated().sum()
    df = df.drop_duplicates()

    # 缺失值统计
    missing_values += df.isnull().sum()

    # 异常值统计
    age_anomalies += detect_anomalies(df['age'], min_val=0, max_val=100)
    income_anomalies += detect_anomalies(df['income'], min_val=0)
    gender_anomalies += detect_anomalies(df['gender'], allowed_values=['男', '女', '其他', '未指定'])

    age.extend(df['age'].tolist())
    gender.extend(df['gender'].tolist())
    income.extend(df['income'].tolist())
    country.extend(df['country'].tolist())
    df['is_active_numeric'] = df['is_active'].map({'TRUE': 1, 'FALSE': 0})
    active.extend(df['is_active_numeric'].tolist())
    for hist in df['purchase_history'].dropna():
        try:
            data = literal_eval(hist) if isinstance(hist, str) else hist
            categories.append(data['categories'])
            price.append(data['avg_price'])
        except:
            continue

print("数据读取完成")


# print(age[0],type(age[0]))

# 输出统计结果
print(f"\n数据数量: {lens}")
print(f"\n重复值数量: {dup_count}")
print("\n缺失值统计:")
print(missing_values)
print(f"\n年龄异常值数量: {age_anomalies}")
print(f"收入异常值数量: {income_anomalies}")
print(f"性别异常值数量: {gender_anomalies}")

# 性别分布可视化
plt.figure(figsize=(10, 6))
# 统计词频
gender_counts = Counter(gender)

# 提取横纵坐标数据
c = list(gender_counts.keys())
f = list(gender_counts.values())
sns.barplot(x=c, y=f)
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('./10G_data_new/gender.png')

# 年龄分布可视化
plt.figure(figsize=(12, 6))
age_bins = [0, 10, 20, 30, 40, 50, 60, 70 ,80 ,90, 100]
age_group = pd.cut(age, bins=age_bins)
age_dist = age_group.value_counts().sort_index()
sns.barplot(x=age_dist.index.astype(str), y=age_dist.values)
plt.title('Age Distribution')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('./10G_data_new/age.png')

# 消费类别

# 统计前10个常见类别
top_categories = Counter(categories).most_common(10)
labels = [x[0] for x in top_categories]
counts = [x[1] for x in top_categories]

# 绘制饼状图
plt.figure(figsize=(12, 8))
plt.pie(counts, 
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 0.5},
        textprops={'fontsize': 9})
plt.title('Top 10 Purchase Categories Distribution', fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.savefig('./10G_data_new/purchase.png')

# 收入分布可视化

plt.figure(figsize=(10, 6))
sns.boxplot(x=income, 
            orient='h',
            palette='Blues',
            showfliers=True)  # 是否显示异常值

plt.title('Income Distribution Boxplot', fontsize=14)
plt.xlabel('Income (unit)', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)

# 设置科学计数法显示
plt.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plt.savefig('./10G_data_new/income.png')

# 用户画像分析
print("---------------------------------------------------------")
print("用户画像分析：")
print("年龄中位数：", np.median(age))
print("年龄平均数：", np.mean(age))
print("性别分布比例：")
for i in range(len(c)):
    print(c[i],":",f[i]/lens)
print("频率前5的国家及其频次：")
top_country = Counter(country).most_common(5)
for x in top_country:
    print(x[0],":",x[1])
print("收入中位数：", np.median(income))
print("收入平均数：", np.mean(income))
print("平均价格：", np.mean(price))
sum_active = 0
for i in active:
    sum_active+=i
print("平均活跃度：", sum_active/lens)
print("---------------------------------------------------------")

end = time.perf_counter()
runTime = end - start

print("运行时间：",runTime,"s")


