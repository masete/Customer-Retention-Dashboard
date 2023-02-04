import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib as mpl 
from datetime import date, datetime
import streamlit as st  

st.set_page_config(
    page_title="Cohorts Dashboard",
    page_icon="📈",
    layout="wide",
)

@st.cache

# Define some functions
def purchase_rate(customer_id):
    purchase_rate = [1]
    counter = 1
    for i in range(1,len(customer_id)):
        if customer_id[i] != customer_id[i-1]:
            purchase_rate.append(1)
            counter = 1
        else:
            counter += 1
            purchase_rate.append(counter)
    return purchase_rate
def join_date(date, purchase_rate):
    join_date = list(range(len(date)))
    for i in range(len(purchase_rate)):
        if purchase_rate[i] == 1:
            join_date[i] = date[i]
        else:
            join_date[i] = join_date[i-1]
    return join_date

def freq_purchase(purchase_rate, day):  # get the frequency (days btw orders)
    freq_purchase = list(range(len(day)))
    for i in range(len(purchase_rate)):
        freq_purchase[i] = 0 if purchase_rate[i] == 1 else (day[i] - day[i-1]).days
        
    return freq_purchase
# or we can just use .diff()/np.timedelta64(1, 'D')


def age_by_month(purchase_rate, month, year, join_month, join_year):
    age_by_month = list(range(len(year)))
    for i in range(len(purchase_rate)):
        if purchase_rate[i] == 1: 
            age_by_month[i] = 0
        else:
            if year[i] == join_year[i]:
                age_by_month[i] = month[i] - join_month[i]
            else:
                age_by_month[i] = month[i] - join_month[i] + 12*(year[i]-join_year[i])
    return age_by_month

def age_by_quarter(purchase_rate, quarter, year, join_quarter, join_year):
    age_by_quarter = list(range(len(year)))
    for i in range(len(purchase_rate)):
        if purchase_rate[i] == 1:
            age_by_quarter[i] = 0
        else:
            if year[i] == join_year[i]:
                age_by_quarter[i] = quarter[i] - join_quarter[i]
            else:
                age_by_quarter[i] = quarter[i] - join_quarter[i] + 4*(year[i]-join_year[i])
    return age_by_quarter

def age_by_year(year, join_year):
    age_by_year = list(range(len(year)))
    for i in range(len(year)):
        age_by_year[i] = year[i] - join_year[i]
    return age_by_year

# get the data
@st.experimental_memo
def get_data() -> pd.DataFrame:
    return pd.read_csv('/Users/joetran/OneDrive/Python/Streamlit/sales_2018-01-01_2019-12-31.csv')

# df = get_data()
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)


# Process df to get cohorts
@st.experimental_memo
def process_df(df):
    df['month'] = pd.to_datetime(df['day']).dt.month
    df = df[~df['customer_type'].isnull()]
    first_time = df.loc[df['customer_type'] == 'First-time',]
    final = df.loc[df['customer_id'].isin(first_time['customer_id'].values)]
    final = final.drop(columns = ['customer_type'])
    final['day']= pd.to_datetime(final['day'], dayfirst=True)
    sorted_final = final.sort_values(['customer_id','day'])
    sorted_final.reset_index(inplace = True, drop = True)
    april=sorted_final.copy()
    first_time = df.loc[df['customer_type'] == 'First-time',]
    final = df.loc[df['customer_id'].isin(first_time['customer_id'].values)]
    final = final.drop(columns = ['customer_type'])
    final['day']= pd.to_datetime(final['day'], dayfirst=True)
    sorted_final = final.sort_values(['customer_id','day'])
    sorted_final.reset_index(inplace = True, drop = True)
    april=sorted_final.copy()

    april['month'] =pd.to_datetime(april['day']).dt.month
    april['Purchase Rate'] = purchase_rate(april['customer_id'])
    april['Join Date'] = join_date(april['day'], april['Purchase Rate'])
    april['Join Date'] = pd.to_datetime(april['Join Date'], dayfirst=True)
    april['cohort'] = pd.to_datetime(april['Join Date']).dt.strftime('%Y-%m')
    april['year'] = pd.to_datetime(april['day']).dt.year
    april['Join Date Month'] = pd.to_datetime(april['Join Date']).dt.month
    april['Join Date Year'] = pd.to_datetime(april['Join Date']).dt.year
    april['Age by month'] = age_by_month(april['Purchase Rate'], april['month'],april['year'], april['Join Date Month'], april['Join Date Year'])
    return april

april=process_df(df)

# # calculate # NCs per month 
# nc_per_month = np.mean(df.loc[df['customer_type']=='First-time'].groupby(['month']).count()['customer_id']) 
# # calculate RCs per month
# rc_per_month=np.mean(df.loc[df['customer_type']=='Returning'].groupby(['month']).count()['customer_id'])


# cohort by exact numbers
@st.experimental_memo
def cohort_numbers(april):
    april_cohorts = april.groupby(['cohort','Age by month']).nunique()
    april_cohorts = april_cohorts.customer_id.to_frame().reset_index()   # convert series to frame
    april_cohorts = pd.pivot_table(april_cohorts, values = 'customer_id',index = 'cohort', columns= 'Age by month')
    return april_cohorts
april_cohorts = cohort_numbers(april)

def draw_cohorts_table_exact_num(april_cohorts):
    april_cohorts = april_cohorts.astype(str)
    april_cohorts=april_cohorts.replace('nan', '',regex=True)
    return april_cohorts

# cohort by percentage
@st.experimental_memo
def cohort_percent(april_cohorts):
    cohorts = april_cohorts.copy()
    #cohorts = cohorts.replace(np.nan,0,regex=True)
    for i in range(len(cohorts.columns)-1):
        cohorts[i+1] = round(cohorts[i+1]/cohorts[0]*100,2)
    cohorts[0] = cohorts[0]/cohorts[0]
    cohorts['average'] = cohorts.iloc[:,1:-1].mean(axis = 1)   # get the average across all columns
    return cohorts
cohorts = cohort_percent(april_cohorts)
@st.experimental_memo
def draw_cohorts_table_percentage(cohorts):
    for i in range(len(cohorts.columns)-2):
        cohorts[i+1]=cohorts[i+1].apply(lambda x:f"{x}%")
    cohorts = cohorts.astype(str)
    cohorts[0] = "100%"
    cohorts=cohorts.replace('nan%', '',regex=True)
    return cohorts

# cohort by AOV
@st.experimental_memo
def cohort_aov(april):
    april_aov = april.groupby(['cohort','Age by month']).mean().total_sales
    april_aov = april_aov.to_frame().reset_index()
    april_aov['total_sales'] = april_aov['total_sales'].apply(lambda x: round(x,2))
    april_aov =  pd.pivot_table(april_aov,values = 'total_sales', index = 'cohort', columns = 'Age by month')    
    return april_aov

april_aov = cohort_aov(april)
def draw_cohorts_aov(april_aov):
    april_aov = april_aov.astype(str)
    april_aov = april_aov.replace('nan', '',regex=True)
    return april_aov

