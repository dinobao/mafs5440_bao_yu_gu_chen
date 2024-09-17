# 1. EDA
# In this section, we mainly focus on application_train|test.csv files.
# The report will cover our research in missing values in these datasets, and some hypothesis,
# questions and solutions regarding features in the estimation in this part.
# We will explore the case of missing values, analyze numerical and categorical features separately,
# and use this as a basis to continue the feature engineering and model building that follows.
# %%
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
# %%
# 1.1 Import Data
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == object:
            continue

        if str(col_type)[:3] == 'int':
            col_min = df[col].min()
            col_max = df[col].max()
            col_range = col_max - col_min

            if col_range <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif col_range <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif col_range <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)

        elif str(col_type)[:5] == 'float':
            col_min = df[col].min()
            col_max = df[col].max()
            col_range = col_max - col_min

            if col_range <= np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif col_range <= np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df
app_train = import_data('application_train.csv')
# %%
# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns


# %%
# through the bar chart of the distribution of missing value, we select 40% as the thereshold for us to
# determine whether to drop the variable
mis_count = missing_values_table(app_train)
# %%
# histogram of misvalue
plt.hist(mis_count['% of Total Values'], bins=10, color='skyblue', edgecolor='black')

# 添加标题和标签
plt.title('% of Total Values Distribution')
plt.xlabel('% of Total Values')
plt.ylabel('Frequency')

# 显示图表
plt.tight_layout()
plt.show()
# columns that need drop
columns_to_drop = mis_count[mis_count['% of Total Values'] > 40].index.tolist()
# delete variable which has more than 40% na value
app_train_nona = app_train.drop(columns=columns_to_drop)

# %%
# 1.3 Feature desrciption
def info(table):
    pd.set_option('display.max_rows', 30)
    print(f'\n The shape of the table is: {table.shape}\n')
    print('-' * 80)
    print('\n Data types:')
    print(table.dtypes.value_counts().sort_values())
    print('-' * 80)
    missing_data = table.isna().mean().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    print('\n Number of features with missing data:', len(missing_data))
    print('-' * 80)
    print('\n Missing data in the table:\n')
    print(missing_data)
    print('-' * 80)
    print('\n Missing data exceeding 40%:\n')
    print(missing_data[missing_data > 0.4])
    print('-' * 80)
    print('\n Number of features with missing data over 40%:', len(missing_data[missing_data > 0.4]))
# %%
info(app_train)

# %%
# Number of each type of column
print(app_train.dtypes.value_counts())
# Number of unique classes in each object column
print(app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0))
# %%
app_obj= app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
obj_list = app_obj.index

# %%
# the Point-Biserial Correlation
def plot_obj(data, column):
    # 计算每个分类变量的总数量
    total_count = data.groupby(column).size().reset_index(name='Total_Count')

    # 计算每个分类变量中 Target=1 的数量
    target_1_count = data[data['TARGET'] == 1].groupby(column).size().reset_index(name='Target_1_Count')

    # 合并总数量和 Target=1 的数量
    merged = pd.merge(total_count, target_1_count, on=column, how='left').fillna(0)

    # 计算在每个分类变量中 Target=1 所占的百分比
    merged['Target_1_Percentage'] = merged['Target_1_Count'] / merged['Total_Count'] * 100

    # 开始绘图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 使用条形图显示每个分类变量的总数量
    sns.barplot(x=column, y='Total_Count', data=merged, ax=ax1, color='skyblue')
    ax1.set_ylabel('Total Count')
    ax1.set_title(f'{column} - Total Count and Target=1 Percentage')
    ax1.tick_params(axis='x', rotation=90)

    # 创建第二个 y 轴，用来显示 Target=1 的占比
    ax2 = ax1.twinx()
    sns.lineplot(x=column, y='Target_1_Percentage', data=merged, ax=ax2, color='red', marker="o")
    ax2.set_ylabel('Target=1 Percentage (%)')

    # 调整布局
    plt.tight_layout()
    plt.show()
for i in obj_list:
    plot_obj(app_train,i)
# cause i use the pycharm to plot, only 11 charts can be showed in one time, therefore, i do this as follows
obj_list2 = ['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']
for i in obj_list2:
    plot_obj(app_train, i)

# %%
# generate dataframe that focus on the continuous variable
obj_list1 = obj_list
for i in ['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE']:
    obj_list1 = obj_list1.drop(i)
app_train_nonaob = app_train_nona.drop(columns = obj_list1)


# %%
def calculate_correlations(data, target_column='TARGET'):
    # calculate the correlation between TARGET and others
    correlations = data.corr()[target_column].drop(target_column)
    sorted_correlations = correlations.abs().sort_values(ascending=False)
    return sorted_correlations
# %%
# boxplot , but we do not show it in our post
def plot_top_correlations_boxplot(data, target_column='TARGET', top_n=10):

    correlations = calculate_correlations(data, target_column)
    top_columns = correlations.head(top_n).index.tolist()

    top_data = data[top_columns]

    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(top_data), columns=top_columns)

    melted_data = pd.melt(scaled_data, var_name='Feature', value_name='Value')

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Feature', y='Value', data=melted_data, palette='coolwarm')

    plt.title(f'Top {top_n} Features - Normalized Distribution', fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=12)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Normalized Value', fontsize=14)

    plt.tight_layout()
    plt.show()
# %%
calculate_correlations(app_train_nonaob)
plot_top_correlations_boxplot(app_train_nonaob)


# %%
# 1.4.2 catgorical statistics
def categorical_stats(df, FEATURES):
    sns.set(style="whitegrid")

    for feature in FEATURES:
        temp = df[feature].value_counts()
        df1 = pd.DataFrame({feature: temp.index, 'value': temp.values})

        cat_perc_0 = df[df['TARGET'] == 0].groupby(feature).size().reset_index(name='Count_Target_0')
        cat_perc_1 = df[df['TARGET'] == 1].groupby(feature).size().reset_index(name='Count_Target_1')

        cat_perc = cat_perc_0.merge(cat_perc_1, how='left', on=feature).fillna(0)

        cat_perc['Percentage_Target_0'] = cat_perc['Count_Target_0'] / (cat_perc['Count_Target_0'] + cat_perc['Count_Target_1']) * 100
        cat_perc['Percentage_Target_1'] = cat_perc['Count_Target_1'] / (cat_perc['Count_Target_0'] + cat_perc['Count_Target_1']) * 100

        cat_perc.sort_values(by=feature, inplace=True)

        # 生成与类别数量相同的调色板
        num_categories = cat_perc.shape[0]
        palette = sns.color_palette("tab20", num_categories)  # 使用20色调色板

        plt.figure(figsize=(14, 7))

        # 绘制目标为0的类别百分比图，并应用不同的颜色
        ax1 = plt.subplot(121)
        sns.barplot(x=feature, y="Percentage_Target_0", data=cat_perc, palette=palette, ax=ax1)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
        ax1.set_title(f'Percentage Target 0 for {feature}')

        # 绘制目标为1的类别百分比图，并应用不同的颜色
        ax2 = plt.subplot(122)
        sns.barplot(x=feature, y='Percentage_Target_1', data=cat_perc, palette=palette, ax=ax2)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
        ax2.set_title(f'Percentage Target 1 for {feature}')

        plt.tight_layout()  # 调整布局，确保标签和内容显示完整
        plt.show()
# %%
plt.figure(figsize = (10, 8))

# KDE plot of loans that were repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')

# Labeling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages')
plt.show()

# %%

def calculate_correlations(data, target_column='TARGET'):
    # 计算所有列和 TARGET 列的相关系数
    correlations = data.corr()[target_column].drop(target_column)  # 计算相关系数并删除自己和自己的相关性

    # 按相关性从高到低排序
    sorted_correlations = correlations.abs().sort_values(ascending=False)

    return sorted_correlations


def plot_top_correlations_boxplot(data, target_column='TARGET', top_n=10):
    # 计算与TARGET的相关性并选择前top_n的列
    correlations = calculate_correlations(data, target_column)
    top_columns = correlations.head(top_n).index.tolist()  # 获取前top_n列名

    # 输出特征名称
    print("Top 10 Features:", top_columns)

    # 选择前top_n列的子集
    top_data = data[top_columns]

    # 数据标准化 (将每个特征缩放至0~1之间)
    scaler = StandardScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(top_data), columns=top_columns)

    # 将数据转换为长格式，以便用seaborn绘图
    melted_data = pd.melt(scaled_data, var_name='Feature', value_name='Value')

    # 绘制箱线图
    plt.figure(figsize=(12, 8))  # 设置图表大小
    sns.boxplot(x='Feature', y='Value', data=melted_data, palette='coolwarm')

    # 美化图表
    plt.title(f'Top {top_n} Features - Normalized Distribution', fontsize=16, fontweight='bold')
    plt.xticks(rotation=90, fontsize=12)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Normalized Value', fontsize=14)

    # 显示图表
    plt.tight_layout()
    plt.show()

    return top_columns  # 返回特征名称列表


# 使用示例
# top_10_features = plot_top_correlations_boxplot(your_dataframe, 'TARGET', top_n=10)
# print("Top 10 correlated features:", top_10_features)

# %%
plot_top_correlations_boxplot(app_train_nonaob)
# %%
# %%
var_highcorr = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'DAYS_BIRTH', 'REGION_RATING_CLIENT_W_CITY', 'REGION_RATING_CLIENT', 'DAYS_LAST_PHONE_CHANGE', 'DAYS_ID_PUBLISH', 'REG_CITY_NOT_WORK_CITY', 'FLAG_EMP_PHONE', 'DAYS_EMPLOYED']
plt.figure(figsize = (8, 6))
# Heatmap of correlations
sns.heatmap(var_highcorr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap')
plt.show()

# %%
# heat map
def plot_correlation_heatmap(data, target_column='TARGET', top_n=10):

    correlations = data.corr()[target_column].drop(target_column)
    top_columns = correlations.abs().sort_values(ascending=False).head(top_n).index.tolist()

    top_columns = [target_column] + top_columns
    top_data = data[top_columns]
    corr_matrix = top_data.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True,
                cbar_kws={"shrink": .75}, linewidths=0.5)

    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.title(f'Top {top_n} Features Correlation Heatmap', fontsize=16, fontweight='bold')

    plt.tight_layout()

    plt.show()

# %%
plot_correlation_heatmap(app_train_nonaob, 'TARGET', top_n=10)
