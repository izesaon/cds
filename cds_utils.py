import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay
import copy

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

# generates a 
# need to be scalable across different stock tickers
# might need to normalize values for easy learning
# accepts a list of dataframes with first column as dates strictly
def DataGenerator(train_points=60, test_freq='daily', **kwargs):
    # create full dataframe and forward fill NaN days due to holiday
    full_df = pd.DataFrame()
    for item in kwargs:
        full_df = full_df.join(kwargs[item], how='outer')
    full_df = full_df.fillna(method='ffill')

    all_items = {item:kwargs[item].index for item in kwargs}
    min_date, max_date = find_earliest(**all_items)
    
    #one_day = pd.Timedelta(days=1)
    one_day = BDay()*1
    
    test_date = min_date + BDay()*train_points
    #test_date = min_date + pd.Timedelta(days=train_points)
    train_date = (min_date, min_date+(BDay()*(train_points-1)))
    #train_date = (min_date, test_date - one_day)
    
    if test_freq == 'weekly':
        test_date = test_date + pd.Timedelta(days=7) # weekly basis
    elif test_freq == 'monthly':
        #test_date = test_date + pd.Timedelta(months=1) # monthly basis
        #test_date = test_date + pd.DateOffset(months=1) # monthly basis
        test_date = test_date + pd.DateOffset(weeks=4) # monthly basis (4 weeks)
    
    count = 0
    while test_date<max_date:
        # extract x_train
        x_train = full_df.loc[train_date[0]:train_date[1],:]
        #x_train = pd.DataFrame()
        #for item in kwargs:
        #    temp_traindf = kwargs[item].loc[train_date[0]:train_date[1],:]
        #    x_train = x_train.join(temp_traindf,how='outer')
        #x_train = x_train.fillna(method='ffill')
        assert not (x_train.isnull().values.any()), 'Error: there are NaN values.'
        # extract y_train (in case there is mising adj close values, so we keep going back to previous price)
        temp_test_date = copy.deepcopy(test_date)
        while True:
            try:
                adjclose = kwargs['price'].loc[temp_test_date,'Adj Close']
                y_train = (test_date.strftime('%Y-%m-%d'), adjclose)
                break
            except KeyError:
                temp_test_date = temp_test_date - one_day
                continue

        print("**********************************************************")
        print("Iteration {}".format(count))
        print("Generating {} rows of training data".format(x_train.shape))
        yield x_train, y_train
        

        test_date = test_date+one_day
        train_date = (train_date[0]+one_day, train_date[1]+one_day)
        count+=1



# class DataGenerator:
#     def __iter__():
#         pass
#     def __next__():
#         pass


if __name__=='__main__':
    # Step 1: Aggregate financial data from discrete csv files
    #compile_financial('AAPL/')

    # Step 2: Extract csv files into pandas dataframe
    # financial dataframe
    financial = pd.read_csv('AAPL - key_financial.csv', parse_dates=[1])
    financial = interpolate_data(financial, method='zero')
    # price dataframe
    price = pd.read_csv('cds/AAPL(011005-011018).csv', parse_dates=[0]) # path to price 
    price.set_index('Date', inplace=True)
    # technical dataframe
    technical = pd.read_csv('cds/technical_indicators.csv', parse_dates=[0])
    technical.set_index('Date', inplace=True)
    # camel case column names 
    for old_column in technical:
        new_column = ' '.join([word.title() for word in old_column.split('_')])
        technical.rename(columns={old_column:new_column}, inplace=True)

    # Parse the dataframes into a DataGenerator
    data_generator = DataGenerator(test_freq='daily', financial=financial, price=price, technical=technical)
    i=0
    for x_train,y_train in data_generator:
        print(x_train)
        print(y_train)
        print(x_train.columns)
        exit()
        i+=1
        if i>100:
            break
        
    
    #print(financial.head())
    #print(price.head())
