import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
import copy
import math

# FINANCIAL INDICATORS
# profitability: gross profit margin, operating profit margin, net profit margin, EBITDA margin
# efficiency: working capital, inventory dats, days receivable outstanding, days payable outstanding
#               account receivable turnover, accounts payable turnover, total asset turnover
# liquidity: current ratio, quick ratio, interest coverage ratio
# return on capital: ROA, ROE
# share as investment: EPS, PE ratio, dividend yield

##############################################################################

# TOP 4 FINANCIAL INDICATORS ??? 
# Earnings per share: indicates company's profitability as a unit of share
# Gross Margin: 
# Revenue growth rate (sequential growth): 
# PE ratio: relationship between the share price and company's earnings


# TOP 4 MARKET INDICATORS
# Long and short moving averages
# Relative Strength Index
# Commodity Channel Index
# Stochastic Oscillator

##############################################################################
# follow naming convention of file
# need the exact naming in the file to extract the values
fin_ind = {'Income Statement':['Basic EPS'], 'Profitability Ratio':['Gross Margin'], 
            'Growth Ratio':['Revenue'], 'Working Capital Ratio':['Inventory to Cash Days'], 
            'Liquidity Ratio':['Cash Ratio']}

# recompile all financial indicators and output a single csv file for easier use
def compile_financial(path):
    compiled_df = None
    for f in os.listdir(path):
        if f.startswith('~') or f.startswith('.'):
            continue
        print(f)
        df = pd.read_excel(os.path.join(path,f), index_col=0, header=0)
        df = df.iloc[:,1:] # remove hidden column
        header_names = df.iloc[1,:]
        
        # create the dataframe if not created
        if compiled_df is None:
            compiled_df = pd.DataFrame(columns=header_names)
            compiled_df.columns.name = 'Quarters'
            temp_df = df.iloc[[2],:]
            temp_df.columns = header_names
            compiled_df = compiled_df.append(temp_df)

        _,ratio_type = f.split('-')
        ratio_type,_ = os.path.splitext(ratio_type.strip())

        # find the exact ratio name and insert into new dataframe
        ratio_list = fin_ind.get(ratio_type,-1)
        if ratio_list==-1:
            continue
        for ratio in ratio_list:
            # special case for repeated words
            if ratio_type == 'Growth Ratio':
                df = df.iloc[df.index.get_loc(key='Sequential Growth'):,:] # only extract the sequential growth portion
            if ratio_type == 'Income Statement':
                df = df.iloc[:df.index.get_loc(key='Net Income'),:]
            # create a new series with indexes as the column names
            #new_series = pd.Series(df.loc[ratio,:].values, index=compiled_df.columns.values)
            #new_series = new_series.rename(ratio)
            #compiled_df = compiled_df.append(new_series)
            temp_df = df.loc[[ratio],:]
            temp_df.columns = header_names
            compiled_df = compiled_df.append(temp_df)
    
    # rename indexes
    compiled_df.rename({'3 Months Ending':'Date'},inplace=True)
    compiled_df.index.name = 'Ratio'
    compiled_df.columns.name = 'Quarter'

    
    # transposing dataframe
    compiled_df = compiled_df.T
    print(compiled_df)
    compiled_df.to_csv(path.replace('/','')+' - key_financial.csv')

# function for interpolating data to business daily
def interpolate_data(df, method='zero'):
    df.set_index('Date',inplace=True)
    # shift index to 1 day back from sat to fri
    for index, row in df.iterrows():
        # if sat, minus one day
        if index.weekday()==5:
            df = df.rename({index:(index-pd.Timedelta(days=1))})
        # if sun, add one day
        elif index.weekday()==6:
            df = df.rename({index:(index+pd.Timedelta(days=1))})
    
    plt.plot(df.loc[:,'Cash Ratio'],'r+')
    if method=='zero':
        df = df.asfreq(freq='b', fill_value=0) # set to business daily
    elif method=='linear':
        df = df.asfreq(freq='b')
        df = df.interpolate(method='time')
        plt.plot(df.loc[:,'Cash Ratio'],'b',label='Linear')
    elif method=='cubic':
        df = df.asfreq(freq='b')
        #df.reset_index(inplace=True)
        df = df.interpolate(method='spline', order=3)
        plt.plot(df.loc[:,'Cash Ratio'],'g',label='Cubic Spline')
    plt.legend()
    #plt.show()

    return df

# function to find overlapping date across indexes of all datasets
def find_earliest(**kwargs):
    min_date = pd.Timestamp(year=1990,month=1,day=1) # arbitrary large date
    max_date = pd.Timestamp(year=2019,month=1,day=1) # arbitrary small date
    for item in kwargs:
        if kwargs[item][0]>min_date:
            min_date = kwargs[item][0]
        if kwargs[item][-1]<max_date:
            max_date = kwargs[item][-1]
        #print(kwargs[item][0])
        #print(kwargs[item][-1])
        #print(type(kwargs[item][0]))
    print('Min date: {}, Max date: {}'.format(min_date, max_date))
    return min_date, max_date

# Purpose:
# 1. join datasets from different sources by overlapping dates
# 2. fill NA values after joining
def combine_datasets(**kwargs):
    # create full dataframe and forward fill NaN days due to holiday
    full_df = pd.DataFrame()
    for item in kwargs:
        full_df = full_df.join(kwargs[item], how='outer')
    full_df = full_df.fillna(method='ffill')
    # drop unnecessary columns
    full_df = full_df.drop(['Quarter', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)

    # get min and max dates from all data sources
    all_index = {item:kwargs[item].index for item in kwargs}
    min_date, max_date = find_earliest(**all_index)

    final = full_df.loc[min_date:max_date,:] 

    # return dataset with overlapping dates
    return final

def preprocess_dataset(dataset):
    # (1) log price to focus on percentage change instead of absolute change
    dataset.loc[:,'Adj Close'] = np.log(dataset.loc[:,'Adj Close'])
    # (2) normalize all columns except for price
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())

    # save a local copy of the dataset
    dataset.to_csv('AAPL - main.csv')

    return dataset

def train_test_split(dataset, spl=0.9):
    cut_index = math.ceil(len(dataset.index)*spl)
    return dataset.iloc[:cut_index,:], dataset.iloc[cut_index:]

def get_dates(pointer, train_days, next):
    x_date = (pointer - train_days + (BDay() * 1) , pointer) 
    if next == 'day':
        y_date = pointer + (BDay() * 1 ) # next day
    elif next == 'week':
        y_date = pointer + pd.Timedelta(days=7) # next week
    elif next == 'month':
        y_date = pointer + pd.DateOffset(weeks=4) # next month (4 weeks)
    return x_date, y_date

# need to be scalable across different stock tickers
# might need to normalize values for easy learning
def data_generator(dataset, train_days=60, next='day'):
    # constants
    min_date = dataset.index[0]
    max_date = dataset.index[-1]
    train_days = BDay() * train_days
    one_day = BDay() * 1 
    
    pointer = min_date + train_days - one_day # fix end of x_date as pointer and use pointer as reference
    x_date, y_date = get_dates(pointer, train_days, next)

    count = 0
    while y_date<max_date:
        x = dataset.loc[x_date[0]:x_date[1],:]
        assert not (x.isnull().values.any()), 'Error: there are NaN values'
        # extract y_train (in case there is mising adj close values, so we keep going back to previous price)
        temp_y_date = copy.deepcopy(y_date)
        while True:
            try:
                adjclose = dataset.loc[temp_y_date,'Adj Close']
                lastprice = x.iloc[-1]['Adj Close']
                direction = 1
                if float(adjclose)<float(lastprice):
                    direction = 0
                y = (y_date.strftime('%Y-%m-%d'), adjclose, direction)
                break
            except KeyError:
                temp_y_date = temp_y_date - one_day
                continue

        #print("**********************************************************")
        #print("Iteration {}".format(count))
        #print("Generating {} rows of training data".format(x.shape))
        yield x, y
        
        pointer = pointer + one_day
        x_date, y_date = get_dates(pointer, train_days, next)
        count+=1

if __name__=='__main__':

    # STEP 0: Check if combined dataset exists. If exists, GO TO STEP 4
    filename = 'AAPL - main.csv'
    if os.path.isfile(filename):
        dataset = pd.read_csv(filename, parse_dates=[0])
        dataset.set_index('Date', inplace=True)
    else:
        # STEP 1:: Aggregate financial data from discrete csv files
        #compile_financial('AAPL/')

        # STEP 2: Extract csv files into pandas dataframe
        # (1) financial dataframe
        financial = pd.read_csv('AAPL - financial.csv', parse_dates=[1])
        financial = interpolate_data(financial, method='zero')
        # (2) price dataframe
        price = pd.read_csv('AAPL - price.csv', parse_dates=[0]) # path to price 
        price.set_index('Date', inplace=True)
        # (3) technical dataframe
        technical = pd.read_csv('AAPL - technical.csv', parse_dates=[0])
        technical.set_index('Date', inplace=True)
        # camel case column names for technical
        for old_column in technical:
            new_column = ' '.join([word.title() for word in old_column.split('_')])
            technical.rename(columns={old_column:new_column}, inplace=True)

        # STEP 3: Join different datasets based on overlapping dates
        dataset = combine_datasets(financial=financial, price=price, technical=technical)
        # STEP 4: Preprocessing of dataset
        dataset = preprocess_dataset(dataset)

    # STEP 5: Split the dataset into train and test
    train, test = train_test_split(dataset, spl=0.9)
    print(train.index[0])
    print(train.index[-1])
    print(test.index[0])
    print(test.index[-1])
    exit()

    # STEP 6: Parse the train/test dataframe into a data generator
    data_gen = data_generator(train, train_days=60, next='day')
    i=0
    for x_train,y_train in data_gen:
        print(x_train.to_string())
        print(y_train)
        #print(x_train.columns)

        i+=1
        #if i>10:
        #   break
