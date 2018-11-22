# 50.038 Computational Data Science

Group Project on predicting direction of stock prices

## How to use

These are a list of utility functions in cds_utils.py to be used for data pre-processing

```
compile_financial(path)
```

Takes in a path to file directory containing financial statements downloaded from Bloomberg. Note that the file must be named in a specific convention (i.e. [stock ticker name] - [ratio type]). For example, AAPL - Profitability Ratio. It returns a compiled csv containing the financial indicators required 

```
interpolate_data(dataset,method)
```

Takes in a csv file as outputted from the previous method. A method of interpolating data can be specified. Returns the interpolated dataframe. 

```
find_earliest(**kwargs)
```

Takes in datetime indexes of dataframes to find the overlapping min and max date. Returns the min and max dates

```
data_generator(train_points, test_freq, **kwargs)
```

A Python generator for generating x_train and y_train. train_points specifies the number of days to train for per batch. test_freq specifies the frequency basis of output (e.g. 'daily' means predicting the next day price). **kwargs are dataframes from the various data sources. Returns a batch of x_train and y_train, where x_train is a dataframe while y_train is of the form (date, actual price)