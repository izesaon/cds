import pandas as pd
import os
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, make_scorer, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import pickle
import copy

import cds_utils as util


def rf():
    pass

class SVC:
    def __init__(self, C, kernel, gamma, days, probability=False):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.probability = probability
        self.filename = "model_C{"+str(C)+"}_K{"+str(kernel)+"}_G{"+str(gamma)+"}_prob{"+str(probability)+"}_day{"+str(days)+"}.pkl"
        self.model = None
    
    # initializes the model 
    def init_model(self):
        self.model = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=self.probability)
        #self.model = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=self.probability, class_weight='balanced')
        #self.model = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=self.probability, class_weight={0:1.1,1:0.90})

    def prepare_dataset(self,train_gen):
        x_batch = []
        y_batch = []
        for x,y in tqdm(train_gen):
            #x = x.drop(['Quarter', 'Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
            x_flat = x.values.flatten()
            x_batch.append(x_flat)
            y_batch.append(y[-1])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        return x_batch, y_batch

    def train(self, train_gen):
        if os.path.isfile(self.filename):
            print("Model exists in local dir and loaded")
            with open(self.filename, 'rb') as f:
                self.model = pickle.load(f)
                return
        
        self.init_model()
        print("Preparing dataset...")
        x_batch, y_batch = self.prepare_dataset(train_gen)
        
        # computing class weights
        print(compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=y_batch))
        #exit()

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
            if self.probability:
                pred = self.model.predict_proba(x_np).tolist()[0][1] # get the prediction for 1
            else:
                pred = self.model.predict(x_np).tolist()[0]
            y_pred.append(pred)
            y_true.append(y[-1])
        return y_true, y_pred

    def param_tuning(self, train_gen, test_gen, nfolds=5):
        Cs = [0.1, 1.0, 10.0, 100.0, 1000.0]
        gammas = [0.0001, 0.001, 0.01, 0.1]
        param_grid = [{'C':Cs, 'gamma':gammas, 'kernel':['rbf']},
                        {'C':Cs, 'kernel':['linear']}]

        x_train, y_train = self.prepare_dataset(train_gen)
        x_test, y_test = self.prepare_dataset(test_gen)
        scores = ['precision_macro', 'recall_macro', 'accuracy', 'roc_auc'] # various scoring functions 
        mat_coef = make_scorer(matthews_corrcoef)
        scores = {'matthew':mat_coef, 'precision_macro':'precision_macro',
                    'recall_macro':'recall_macro','accuracy':'accuracy','roc_auc':'roc_auc'}
        print('################################################################################', file=open('output.txt','a'))
        for score in scores:
            print("# Tuning hyper-parameters for %s\n\n" % score, file=open('output.txt','a'))
            clf = GridSearchCV(svm.SVC(), param_grid, cv=nfolds, scoring=scores[score])
            #clf = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid, cv=nfolds, scoring='%s'%score)
            #clf = GridSearchCV(svm.SVC(class_weight={0:1.05,1:0.95}), param_grid, cv=nfolds, scoring=scores[score])
            print(clf, file=open('output.txt','a'))
            clf.fit(x_train, y_train)
            print("Best parameters set found on train set:\n\n",file=open('output.txt','a'))
            print(str(clf.best_params_)+"\n\n",file=open('output.txt','a'))
            print("Grid scores on train set:\n\n",file=open('output.txt','a'))
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                print("%0.3f (+/-%0.03f) for %r"
                      % (mean, std * 2, params), file=open('output.txt','a'))
            print("\n\n", file=open('output.txt','a'))
            print("Detailed classification report:\n\n",file=open('output.txt','a'))
            print("The model is trained on the full train set.",file=open('output.txt','a'))
            print("The scores are computed on the full test set.\n\n",file=open('output.txt','a'))
            y_true, y_pred = y_test, clf.predict(x_test)
            print(classification_report(y_true, y_pred),file=open('output.txt','a'))
            print("Accuracy on test set: {}\n\n".format(accuracy_score(y_true, y_pred)), file=open('output.txt','a'))
            print("Confusion matrix:", file=open('output.txt','a'))
            print(confusion_matrix(y_true, y_pred), file=open('output.txt','a'))
            print("\n\n",  file=open('output.txt','a'))
        print("---- END OF TEST ----", file=open('output.txt','a'))

        #grid_search.fit(x_batch, y_batch)

    # TODO: generates a visualization
    def plot():
        pass

if __name__=='__main__':

    # STEP 0: Check if combined dataset exists. If exists, GO TO STEP 4
    company = 'AAPL'
    filename = ' - main.csv'
    if os.path.isfile(filename):
        dataset = pd.read_csv(filename, parse_dates=[0])
        dataset.set_index('Date', inplace=True)
    else:
        # STEP 1:: Aggregate financial data from discrete csv files
        #util.compile_financial('AAPL/')

        # STEP 2: Extract csv files into pandas dataframe
        # (1) financial dataframe
        financial = pd.read_csv(company+' - financial.csv', parse_dates=[1])
        financial = util.interpolate_data(financial, method='zero')
        # (2) price dataframe
        price = pd.read_csv(company+' - price.csv', parse_dates=[0]) # path to price 
        price.set_index('Date', inplace=True)
        # (3) technical dataframe
        technical = pd.read_csv(company+' - technical.csv', parse_dates=[0])
        technical.set_index('Date', inplace=True)
        # camel case column names for technical
        for old_column in technical:
            new_column = ' '.join([word.title() for word in old_column.split('_')])
            technical.rename(columns={old_column:new_column}, inplace=True)

        # STEP 3: Join different datasets based on overlapping dates
        dataset = util.combine_datasets(financial=financial, price=price, technical=technical)

        # STEP 4: Preprocessing of dataset
        dataset = util.preprocess_dataset(dataset)
        # save a local copy of the dataset
        dataset.to_csv(company+' - main.csv')

    # STEP 5: Split the dataset into train and test
    train, test = util.train_test_split(dataset, spl=0.5)

    # STEP 6: Parse the train/test dataframe into a data generator
    train_days = 10
    train_gen = util.data_generator(train, train_days=train_days, next='day')
    test_gen = util.data_generator(test, train_days=train_days, next='day')

    # STEP 7: Build model
    probability = True
    svc = SVC(C=1, kernel='linear', gamma=1, days=train_days, probability=probability)
    svc.train(train_gen)
    print(svc.model)

    true, pred = svc.predict(test_gen)
    if probability:
        # convert probability to labels
        pred = [int(i>0.5) for i in pred]
    accuracy = accuracy_score(true, pred)
    confusion = confusion_matrix(true, pred)
    print("Test Accuracy: {:.2f}%".format(accuracy*100))
    print("Confusion Matrix")
    print(confusion)

    # STEP 8: Parameter tuning using grid search
    train_gen = util.data_generator(train, train_days=10, next='day')
    test_gen = util.data_generator(test, train_days=10, next='day')
    svc.param_tuning(train_gen, test_gen, nfolds=5)


