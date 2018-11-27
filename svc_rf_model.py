import pandas as pd
import os
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import pickle

import cds_utils as util


def rf():
    pass

class SVC:
    def __init__(self, C, kernel, gamma):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.filename = "model_"+str(C)+"_"+str(kernel)+"_"+str(gamma)+".pkl"
        self.model = None
    
    # initializes the model 
    def init_model(self):
        self.model = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=True) 

    def train(self, train_gen):
        if os.path.isfile(self.filename):
            print("Model exists in local dir and loaded")
            with open(self.filename, 'rb') as f:
                self.model = pickle.load(f)
                return
        
        self.init_model()
        print("Preparing dataset...")
        x_batch = []
        y_batch = []
        for x,y in tqdm(train_gen):
            #x = x.drop(['Quarter', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
            x_flat = x.values.flatten()
            x_batch.append(x_flat)
            y_batch.append(y[-1])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        print("Fitting...might take a while")
        self.model.fit(x_batch, y_batch)

        # save the model
        with open(self.filename, 'wb') as f:
            pickle.dump(self.model, f)

    def predict(self, test_gen):
        print("Predicting...")
        y_pred = []
        y_true = []
        for x,y in test_gen:
            #x = x.drop(['Quarter', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
            x_flat = x.values.flatten()
            x_np = np.array([x_flat])
            y_pred.append(self.model.predict(x_np).tolist()[0])
            y_true.append(y[-1])
        return y_true, y_pred

    # generates a visualization
    def plot():
        pass

if __name__=='__main__':

    # STEP 0: Check if combined dataset exists. If exists, GO TO STEP 4
    filename = 'AAPL - main.csv'
    if os.path.isfile(filename):
        dataset = pd.read_csv(filename, parse_dates=[0])
        dataset.set_index('Date', inplace=True)
    else:
        # STEP 1:: Aggregate financial data from discrete csv files
        #util.compile_financial('AAPL/')

        # STEP 2: Extract csv files into pandas dataframe
        # (1) financial dataframe
        financial = pd.read_csv('AAPL - financial.csv', parse_dates=[1])
        financial = util.interpolate_data(financial, method='zero')
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
        dataset = util.combine_datasets(financial=financial, price=price, technical=technical)

        # STEP 4: Preprocessing of dataset
        dataset = util.preprocess_dataset(dataset)

    # STEP 5: Split the dataset into train and test
    train, test = util.train_test_split(dataset, spl=0.5)

    # STEP 6: Parse the train/test dataframe into a data generator
    train_gen = util.data_generator(train, train_days=10, next='day')
    test_gen = util.data_generator(test, train_days=10, next='day')

    # STEP 7: Build model
    svc = SVC(C=1, kernel='linear', gamma=1)
    svc.train(train_gen)
    print(svc.model)

    true, pred = svc.predict(test_gen)
    #pred_label = [i.tolist()[0]>0.5 for i in pred]
    accuracy = accuracy_score(true, pred)
    confusion = confusion_matrix(true, pred)
    print("Test Accuracy: {:.2f}%".format(accuracy*100))
    print("Confusion Matrix")
    print(confusion)

    # next step: grid search

