##############################################################
# 1.Data Preperation
##############################################################

##########################
# Required Libraries and Functions
##########################

# pip install lifetimes
# pip install sqlalchemy
# pip install mysql-connector-python==8.0.27

from sqlalchemy import create_engine
import datetime as dt
import  pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#########################
# Reading Data From Database
#########################

# credentials
creds = {'user': 'testuser',
         'passwd': 'testpasswd',
         'host': 'testhost',
         'port': 3306,
         'db' : 'testdb'
         }

# MySQL connection string
connstr = 'mysql+mysqlconnector://{user}:{passwd}@{host}:{port}/{db}'

# sqlalchemy engine for MySQL connection
# it acts as a bridge to the remote server
conn = create_engine(connstr.format(**creds))

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)

pd.read_sql_query("select * from group_7.online_retail_2010_2011 limit 10", conn)

retail_mysql_df = pd.read_sql_query("select * from group_7.online_retail_2010_2011", conn)

retail_mysql_df.shape
retail_mysql_df.head()
retail_mysql_df.info()
df = retail_mysql_df.copy()

#########################
# Data Preprocessing
#########################

# Before preprocessing
df.describe().T
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# After preprocessing
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

#########################
# Preparation of Lifetime Data Structure
#########################
# recency: Time since last purchase. Weekly.
# T: Tenure. Weekly. (how long before the analysis date the first purchase was made)
# frequency: total number of repeat purchases (frequency>1)
# monetary_value: average earning per purchase

cltv_df = df.loc[df["Country"] == "United Kingdom"].groupby("CustomerID").agg({'InvoiceDate' : [lambda date: (date.max() -date.min()).days,  # recency
                                                                                                lambda  date: (today_date - date.min()).days], # T
                                                                               'Invoice': lambda num: num.nunique(), # frequency
                                                                               'TotalPrice': lambda TotalPrice: TotalPrice.sum()}) # monetary
# drop level 0 in columns
cltv_df.columns = cltv_df.columns.droplevel(0)

cltv_df.columns = ["recency", "T", "frequency", "monetary"]

# expressing monetary value as average earning per purchase
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

# frequency must be greater than 1.In this case, we can talk about the formation of pattern.
cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df = cltv_df[cltv_df["monetary"] > 0]

# expression of recency and T in weekly terms for the BGNBD model
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

##############################################################
# 2. Application of BG-NBD Model
##############################################################
bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df["frequency"],
         cltv_df["recency"],
         cltv_df["T"])

# Who are the 10 customers we expect to buy the most in a week?
cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T']).sort_values(ascending=False).head(10)

# Who are the 10 customers we expect to buy the most in a month?
cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T']).sort_values(ascending=False).head(10)

##############################################################
# 3. Application of GAMMA-GAMMA Model
##############################################################
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df["frequency"], cltv_df["monetary"])
cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

##############################################################
# 4. Calculation of CLTV with BG-NBD ve GG Model
##############################################################
# Make a 6-month CLTV prediction for 2010-2011 UK customers.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=6, # 6 aylık
                                   freq="W", # T'nin frekans bilgisi
                                   discount_rate=0.01)
cltv.head()
cltv.shape

cltv = cltv.reset_index()

cltv.sort_values("clv", ascending=False).head(10)

cltv_final = cltv_df.merge(cltv, on="CustomerID", how="left")
cltv_final.sort_values("clv", ascending=False).head(10)

# standartization of CLTV
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

cltv_final.sort_values("scaled_clv", ascending=False).head()
###################################################################
# Calculate 1-month and 12-months CLTV for 2010-2011 UK customers
# Analyze the top 10 people at 1-month CLTV and the top 10 people at 12 months
cltv_2 = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=1, # 1 month
                                   freq="W", # frequency information of T
                                   discount_rate=0.01)

cltv_3 = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time=12, # 12 months
                                   freq="W", # frequency information of T
                                   discount_rate=0.01)

cltv_2.head()
cltv_2.shape

cltv_2 = cltv_2.reset_index()

cltv_2.sort_values("clv", ascending=False).head(10)

cltv_final2 = cltv_df.merge(cltv_2, on="CustomerID", how="left")
cltv_final2.sort_values("clv", ascending=False).head(10)

# standartization of CLTV
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final2[["clv"]])
cltv_final2["scaled_clv"] = scaler.transform(cltv_final2[["clv"]])

cltv_final2.sort_values("scaled_clv", ascending=False).head(10)

###################################################################

cltv_3.head()
cltv_3.shape

cltv_3 = cltv_3.reset_index()

cltv_3.sort_values("clv", ascending=False).head(10)

cltv_final3 = cltv_df.merge(cltv_3, on="CustomerID", how="left")
cltv_final3.sort_values("clv", ascending=False).head(10)

# standartization of CLTV
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final3[["clv"]])
cltv_final3["scaled_clv"] = scaler.transform(cltv_final3[["clv"]])

cltv_final3.sort_values("scaled_clv", ascending=False).head(10)

################################################################
# For 2010-2011 UK customers, divide all your customers into 4 segments according to 6-month CLTV
# and add the group names to dataset

cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

cltv_final.groupby("segment").agg({"mean"})

################################################################
# Sending results to database

pd.read_sql_query("show databases;", conn)
pd.read_sql_query("show tables", conn)

cltv_final.head()

cltv_final["CustomerID"] = cltv_final["CustomerID"].astype(int)

cltv_final.to_sql(name='ocd', con=conn, if_exists='replace', index=False)
