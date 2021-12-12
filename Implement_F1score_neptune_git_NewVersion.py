'''
Implementing F1 score in Keras
Neptune model: https://app.neptune.ai/katyl/KerasMetricNeptune/experiments?split=bth&dash=charts&viewId=standard-view 
'''

### New version
import neptune.new as neptune
import os

# Connect your script to Neptune new version  
myProject = "'YourUserName/YourProjectName'" ## !! 'YourUserName/YourProjectName'
project = neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                       project=myProject) 
project.stop()


### Implementing the Macro F1 Score in Keras
# Create an experiment and log hyperparameters
## How to track the weights and predictions in Neptune (new version)
npt_exp = neptune.init(    
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        project=myProject, 
        name='step-by-step-implement-fscores',         
        tags=['keras', 'classification', 'macro f-scores','neptune']) 

#### Load in the packages 
import pandas as pd
import numpy as np
from random import sample, seed
from collections import defaultdict

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve

#### if use tensorflow=2.0.0, then import tensorflow.keras.model_selection 
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten, BatchNormalization, Dropout, Reshape, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint 

pd.options.display.max_columns = 100
np.set_printoptions(suppress=True) 


credit_dat = pd.read_csv(r'YourDataPath\creditcard.csv')


counts = credit_dat.Class.value_counts()
class0, class1 = round(counts[0]/sum(counts)*100, 2), round(counts[1]/sum(counts)*100, 2)
print(f'Class 0 = {class0}% and Class 1 = {class1}%')

#### Plot the Distribution
sns.set(style="whitegrid")
ax = sns.countplot(x="Class", data=credit_dat)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(p.get_height()/len(credit_dat)*100), (p.get_x()+0.15, p.get_height()+1000))
ax.set(ylabel='Count', 
       title='Credit Card Fraud Class Distribution')

## Neptune new version 
npt_exp['Distribution'].upload(neptune.types.File.as_image(ax.get_figure()))

dat = credit_dat

##### comparing the variable means:
def myformat(value, decimal=4):
    return str(round(value, decimal))

### Preprocess the training and testing data 
### save 20% for final testing 
def Pre_proc(dat, current_test_size=0.2, current_seed=42):    
    x_train, x_test, y_train, y_test = train_test_split(dat.iloc[:, 0:dat.shape[1]-1], 
                                                        dat['Class'], 
                                                        test_size=current_test_size, 
                                                        random_state=current_seed)
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    y_train, y_test = np.array(y_train), np.array(y_test)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = Pre_proc(dat)


### Defining the custom metric function F1
def custom_f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall 
    
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
        precision = TP / (Pred_Positives+K.epsilon())
        return precision 
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



### Defining the Callback Metrics Object to track in Neptune
class NeptuneMetrics(Callback):
    def __init__(self, neptune_experiment, validation, current_fold):   
        super(NeptuneMetrics, self).__init__()
        self.exp = neptune_experiment
        self.validation = validation 
        self.curFold = current_fold
                    
    def on_train_begin(self, logs={}):        
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
     
    def on_epoch_end(self, epoch, logs={}):
        val_targ = self.validation[1]   
        val_predict = (np.asarray(self.model.predict(self.validation[0]))).round()        
    
        val_f1 = round(f1_score(val_targ, val_predict), 4)
        val_recall = round(recall_score(val_targ, val_predict), 4)     
        val_precision = round(precision_score(val_targ, val_predict), 4)
        
        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)
        
        print(f' — val_f1: {val_f1} — val_precision: {val_precision}, — val_recall: {val_recall}')
        
        ### Send the performance metrics to Neptune for tracking (new version) ###        
        self.exp['Epoch End Loss'] = logs['loss']
        self.exp['Epoch End F1-score'] = val_f1
        self.exp['Epoch End Precision'] = val_precision
        self.exp['Epoch End Recall'] = val_recall

        # self.exp.send_metric('Epoch End Loss', logs['loss'])
        # self.exp.send_metric('Epoch End F1-score', val_f1)
        # self.exp.send_metric('Epoch End Precision', val_precision)
        # self.exp.send_metric('Epoch End Recall', val_recall)
                
        if self.curFold == 4:            
            ### Log Epoch End metrics values for each step in the last CV fold ###
            msg = f' End of epoch {epoch} val_f1: {val_f1} — val_precision: {val_precision}, — val_recall: {val_recall}'
            # self.exp.send_text('Epoch End Metrics (each step) for fold {self.curFold}', x=epoch, y=msg) 
            
            ### Neptune new version
            self.exp[f'Epoch End Metrics (each step) for fold {self.curFold}'] = msg


               
### Building a neural nets 
def runModel(x_tr, y_tr, x_val, y_val, epos=20, my_batch_size=112):  
    inp = Input(shape = (x_tr.shape[1],))
    
    x = Dense(1024, activation='relu')(inp)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
        
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inp, out)
    
    return model 
 
### CV for the model training
models = []
f1_cv, precision_cv, recall_cv = [], [], []

current_folds = 5
current_epochs = 20
current_batch_size = 112

## macro_f1 = True for Callback 
macro_f1 = True

kfold = StratifiedKFold(current_folds, random_state=42, shuffle=True)

for k_fold, (tr_inds, val_inds) in enumerate(kfold.split(X=x_train, y=y_train)):

    print('---- Starting fold %d ----'%(k_fold+1))
    
    x_tr, y_tr = x_train[tr_inds], y_train[tr_inds]
    x_val, y_val = x_train[val_inds], y_train[val_inds]
    
    model = runModel(x_tr, y_tr, x_val, y_val, epos=current_epochs)
    
    if macro_f1:        
        model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=[])  
        model.fit(x_tr, 
                  y_tr, 
                  callbacks=[NeptuneMetrics(npt_exp, validation=(x_val, y_val), current_fold=k_fold)],  
                  epochs=current_epochs, 
                  batch_size=current_batch_size,   
                  verbose=1)
    else:
        model.compile(loss='binary_crossentropy', optimizer= "adam", metrics=[custom_f1, 'accuracy'])
        history = model.fit(x_tr, 
                          y_tr,                  
                          epochs=current_epochs, 
                          batch_size=current_batch_size,   
                          verbose=1)
        #### send to metric 
        for val in history.history['custom_f1']:
            ## Neptune new version
            npt_exp['Custom F1 metric'] = val
            ## npt_exp.send_metric('Custom F1 metric', val)
  
    models.append(model)
    
    y_val_pred = model.predict(x_val)
    y_val_pred_cat = (np.asarray(y_val_pred)).round() 

    ### Get performance metrics 
    f1, precision, recall = f1_score(y_val, y_val_pred_cat), precision_score(y_val, y_val_pred_cat), recall_score(y_val, y_val_pred_cat)
    
    print("the fold %d f1 score is %f"%((k_fold+1), f1))
    
    ##### Log performance measures for each Fold
    metric_text = f'Fold {k_fold+1} f1 score = '
    ## npt_exp.send_text(metric_text, myformat(f1))  ## str(round(f1, 4)) 
    ## Neptune new version 
    npt_exp[metric_text] = myformat(f1)
   
    f1_cv.append(round(f1, 4))
    precision_cv.append(round(precision, 4))
    recall_cv.append(round(recall, 4))        

##### Log performance measures after CV
print('mean f1 score = %f'% (np.mean(f1_cv)))

metric_text_final = 'Mean f1 score through CV = '
## npt_exp.send_text(metric_text_final, myformat(np.mean(f1_cv)))  
## Neptune new version 
npt_exp[metric_text_final] = myformat(np.mean(f1_cv))

 
### Predicting the hold-out testing data        
def predict(x_test):
    model_num = len(models)
    for k, m in enumerate(models):
        if k==0:
            y_pred = m.predict(x_test, batch_size=current_batch_size)
        else:
            y_pred += m.predict(x_test, batch_size=current_batch_size)
            
    y_pred = y_pred / model_num    
    
    return y_pred

y_test_pred_prob = predict(x_test)
y_test_pred_cat = predict(x_test).round()

cm = confusion_matrix(y_test, y_test_pred_cat)
f1_final = round(f1_score(y_test, y_test_pred_cat), 4)

#### Log final test F1 score (new version)
npt_exp['TestSet F1 score'] = myformat(f1_final) 
## npt_exp.send_text('TestSet F1 score = ', myformat(f1_final))

print(cm)

from scikitplot.metrics import plot_confusion_matrix
fig_confmat, ax = plt.subplots(figsize=(12, 10))
plot_confusion_matrix(y_test, y_test_pred_cat.astype(int).flatten(), ax=ax)

# Log performance charts to Neptune (new version)
## npt_exp.log_image('Confusion Matrix', fig_confmat)
npt_exp['Confusion Matrix'].upload(neptune.types.File.as_image(fig_confmat))
npt_exp.stop()











