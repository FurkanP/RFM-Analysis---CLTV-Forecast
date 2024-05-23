import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes import BetaGeoFitter, GammaGammaFitter
import warnings

warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Load the dataset
df = pd.read_csv("/kaggle/input/flo-rfm-cltv/flo_data_20k.csv")

# Display first few rows
df.head()

# Overview of the dataset
df.info()

# Summary statistics
df.describe().T

# Check for missing values
df.isnull().sum()

# Visualize the distribution of order channels
sns.countplot(x=df["order_channel"], data=df)
plt.show()

# Boxplot of customer value for online purchases
sns.boxplot(x=df["customer_value_total_ever_online"])
plt.show()

# Create new features for total orders and total customer value
df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# Convert date columns to datetime
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

# Overview of the modified dataset
df.info()

# Aggregation of orders and customer value by order channel
df.groupby("order_channel").agg({"order_num_total": ["count", "sum"], "customer_value_total": ["sum", "mean"]})

# Top 10 customers by total customer value
df.sort_values("customer_value_total", ascending=False).head(10)

# RFM Analysis
today_date = dt.datetime(2021, 6, 1)

# Create RFM dataframe
rfm = pd.DataFrame()
rfm["customer_id"] = df["master_id"]
rfm["recency"] = (today_date - df["last_order_date"]).dt.days
rfm["frequency"] = df["order_num_total"]
rfm["monetary"] = df["customer_value_total"]

# Assign RFM scores
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5])

# Create RF score
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)

# Map RF scores to segments
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalist',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map, regex=True)

# Segment averages
rfm.groupby("segment").agg(
    {"recency": ["mean", "count"], "frequency": ["mean", "count"], "monetary": ["mean", "count"]})

# Example 1: Target customers for a new high-end women's shoe brand
target_segments_customer_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])]["customer_id"]
cust_ids = \
df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))][
    "master_id"]
cust_ids.head()

# Example 2: Target customers for a discount campaign on men's and children's products
target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose", "hibernating", "new_customers"])]["customer_id"]
cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (
            (df["interested_in_categories_12"].str.contains("ERKEK")) | (
        df["interested_in_categories_12"].str.contains("COCUK")))]["master_id"]
cust_ids.head()


# CLTV Prediction
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
           "customer_value_total_ever_online"]
for col in columns:
    replace_with_thresholds(df, col)

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

analysis_date = dt.datetime(2021, 6, 1)

cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).dt.days) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).dt.days) / 7
cltv_df["frequency"] = df["order_num_total"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]

# BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'])

# Predict expected sales for 3 and 6 months
cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3, cltv_df['frequency'], cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])
cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6, cltv_df['frequency'], cltv_df['recency_cltv_weekly'],
                                           cltv_df['T_weekly'])

# Top 10 customers by expected sales
cltv_df.sort_values("exp_sales_3_month", ascending=False).head(10)
cltv_df.sort_values("exp_sales_6_month", ascending=False).head(10)

# Gamma-Gamma model
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                       cltv_df['monetary_cltv_avg'])

# Calculate 6-month CLTV
cltv = ggf.customer_lifetime_value(bgf, cltv_df['frequency'], cltv_df['recency_cltv_weekly'], cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'], time=6, freq="W", discount_rate=0.01)
cltv_df["cltv"] = cltv

# Top 10 customers by CLTV
cltv_df.sort_values("cltv", ascending=False).head(10)

# Segment customers by CLTV
cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

# Summary statistics for each CLTV segment
cltv_df.groupby("cltv_segment").agg({"cltv": ["count", "mean", "std", "median"]})
