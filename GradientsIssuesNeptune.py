# -*- coding: utf-8 -*-
"""
Neptune-- Gradients Problems 
@author: Kat Li
"""

# Connect your script to Neptune
### guide to update this init function, https://docs.neptune.ai/migration-guide
import neptune.new as neptune
import os
myProject = 'YourUserName/YourProjectName'
project = neptune.init(api_token=os.getenv('NEPTUNE_API_TOKEN'),
                        project=myProject) 
project.stop()

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import RandomNormal, RandomUniform
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score
import tensorflow.keras.backend as K
import tensorflow.keras as keras
from collections import Counter
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, Flatten, concatenate, Dropout, Lambda, BatchNormalization
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.metrics import Accuracy 

from neptune.new.integrations.tensorflow_keras import NeptuneCallback


tf.random.set_seed(42)
np.random.seed(42) 
CUR_SEED = 42

## Input data simulation
plt.title('Two Moons with Substantial Overlap')
# X, y = make_moons(n_samples=3000, shuffle=True , noise=0.25, random_state=1234)
X, y = make_moons(n_samples=3000, shuffle=True , noise=0.25, random_state=1234)

plt.scatter(X[:, 0], X[:, 1], c=y, s=25)    
plt.show()


batch_size, n_epochs = 32, 100

## Define a function to calculate gradients 
## tf.GradientTape allows us to track TensorFlow computations and calculate gradients w.r.t. (with respect to) some given variables
def getBatchGradWgts(grads, wgts, lossVal, 
                      gradHist, lossHist, wgtsHist, 
                      recordWeight=True, npt_exp=None):
    dataGrad, dataWeight = {}, {}
    ## batch update 'weights'
    for wgt, grad in zip(wgts, grads):
        if '/kernel:' not in wgt.name:
            continue 
        layerName = wgt.name.split("/")[0]         
        dataGrad[layerName] = grad.numpy()
        dataWeight[layerName] = wgt.numpy()
        ## Log in Neptune
        if npt_exp:
            npt_exp[f'MeanGrads{layerName.upper()}'].log(np.mean(grad.numpy()))   
            npt_exp[f'MeanWgtBatch{layerName.upper()}'].log(np.mean(wgt.numpy()))         
        
    gradHist.append(dataGrad)
    lossHist.append(lossVal.numpy())
    if recordWeight:
        wgtsHist.append(dataWeight)           
                
                
##Run model based on batch, and then calculate batch gradients/weights               
def fitModel(X, y, model, optimizer, 
              n_epochs=n_epochs, curBatch_size=batch_size, 
              modelType = 'binary', ## regression
              npt_exp=None):
    
    if modelType == 'binary':
        lossFunc = tf.keras.losses.BinaryCrossentropy()
    elif modelType == 'regression':
        lossFunc = tf.keras.losses.MeanSquaredError()
        
    subData = tf.data.Dataset.from_tensor_slices((X, y))
    subData = subData.shuffle(buffer_size=42).batch(curBatch_size)
   
    gradHist, lossHist, wgtsHist = [], [], []                    
    
    for epoch in range(n_epochs):
        print(f'== Starting epoch {epoch} ==')        
        for step, (x_batch, y_batch) in enumerate(subData):
            with tf.GradientTape() as tape:
                ## Predict with the model and calculate loss
                yPred = model(x_batch, training=True)
                lossVal = lossFunc(y_batch, yPred)
                
            ## Calculate gradients using tape and update the weights
            grads = tape.gradient(lossVal, model.trainable_weights)
            wgts = model.trainable_weights
            optimizer.apply_gradients(zip(grads, model.trainable_weights))           
            
            ## Save the Interation#5 from each epoch
            if step == 5:
                getBatchGradWgts(gradHist=gradHist, lossHist=lossHist, wgtsHist=wgtsHist, 
                                  grads=grads, wgts=wgts, lossVal=lossVal, npt_exp=npt_exp) 
                if npt_exp:
                    npt_exp['BatchLoss'].log(lossVal)   
                    
    getBatchGradWgts(gradHist=gradHist, lossHist=lossHist, wgtsHist=wgtsHist, 
                      grads=grads, wgts=wgts, lossVal=lossVal, npt_exp=npt_exp)    
    return gradHist, lossHist, wgtsHist



def gradientsVis(curGradHist, curLossHist, modelName):    
    fig, ax = plt.subplots(1, 1, sharex=True, constrained_layout=True, figsize=(7,5))
    ax.set_title(f"Mean gradient {modelName}")
    for layer in curGradHist[0]:
        ax.plot(range(len(curGradHist)), [gradList[layer].mean() for gradList in curGradHist], label=f'Layer_{layer.upper()}')
    ax.legend()
    return fig


# MODELNAME = 'vanSigBaseline'
# MODELNAME = 'vanRelu'
# MODELNAME = 'vanSigSmall'
# MODELNAME = 'vanBN' ## sometimes work; also scale your input data
# MODELNAME = 'vanSigWgtInit' 
MODELNAME = 'vanSigLR'


## Create an experiment in Neptune
if MODELNAME == 'vanSigBaseline':    
    npt_exp = neptune.init(    
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        project=myProject, 
        name='VanishingGradSigmoid', 
        description='Vanishing Gradients with Sigmoid Activation Function', 
        tags=['vanishingGradients', 'sigmoid', 'neptune']) 
    
    ## Define Neptune callback 
    neptune_cbk = NeptuneCallback(run=npt_exp, base_namespace='metrics')
    def binaryModel(curName, curInitializer, curActivation, x_tr=None):
        model = Sequential()
        model.add(InputLayer(input_shape=(2, ), name=curName+"0"))
        model.add(Dense(10, kernel_initializer=curInitializer, activation=curActivation, name=curName+"1"))
        model.add(Dense(10, kernel_initializer=curInitializer, activation=curActivation, name=curName+"2"))
        model.add(Dense(5, kernel_initializer=curInitializer, activation=curActivation,  name=curName+"3"))
        model.add(Dense(1, kernel_initializer=curInitializer, activation='sigmoid', name=curName+"4"))
        return model 
    
    curOptimizer = tf.keras.optimizers.RMSprop()
    optimizer = curOptimizer
    curInitializer = RandomUniform(-1, 1)
    ## Compile the model
    model = binaryModel(curName="SIGMOID", curInitializer=curInitializer, curActivation="sigmoid")  
    model.compile(optimizer=curOptimizer, loss='binary_crossentropy', metrics=['accuracy'])
    ## Train and Log in Neptune    
    curGradHist, curLossHist, curWgtHist = fitModel(X, y, model, optimizer=curOptimizer, npt_exp=npt_exp)
    ## log in the plot comparing all layers
    npt_exp['Comparing All Layers'].upload(neptune.types.File.as_image(gradientsVis(curGradHist, curLossHist, 
                                                                                    modelName='Sigmoid_Raw')))
    npt_exp.stop()  

elif MODELNAME == 'vanRelu':
    npt_exp = neptune.init(    
    api_token=os.getenv('NEPTUNE_API_TOKEN'),
    project=myProject, 
    name='VanishingGradRelu', 
    # description='Vanishing Gradients with Sigmoid Activation Function', 
    tags=['vanishingGradients', 'relu', 'neptune']) 

    def binaryModel(curName, curInitializer, curActivation, x_tr=None):
        model = Sequential()
        model.add(InputLayer(input_shape=(2, ), name=curName+"0"))
        model.add(Dense(10, kernel_initializer=curInitializer, activation=curActivation, name=curName+"1"))
        model.add(Dense(10, kernel_initializer=curInitializer, activation=curActivation, name=curName+"2"))
        model.add(Dense(5, kernel_initializer=curInitializer, activation=curActivation,  name=curName+"3"))
        model.add(Dense(1, kernel_initializer=curInitializer, activation='sigmoid', name=curName+"4"))
        return model 
    
    curOptimizer = tf.keras.optimizers.RMSprop()
    optimizer = curOptimizer
    curInitializer = RandomUniform(-1, 1)
    ## Compile the model
    model = binaryModel(curName="Relu", curInitializer=curInitializer, curActivation="relu")  
    model.compile(optimizer=curOptimizer, loss='binary_crossentropy', metrics=['accuracy'])
    ## Train and Log in Neptune    
    curGradHist, curLossHist, curWgtHist = fitModel(X, y, model, optimizer=curOptimizer, npt_exp=npt_exp)
    ## log in the plot comparing all layers
    npt_exp['Comparing All Layers'].upload(neptune.types.File.as_image(gradientsVis(curGradHist, curLossHist, modelName='Relu')))
    npt_exp.stop() 
    
elif MODELNAME == 'vanSigSmall':
    npt_exp = neptune.init(    
    api_token=os.getenv('NEPTUNE_API_TOKEN'),
    project=myProject, 
    name='VanishingGradSigmoidSmall', 
    tags=['vanishingGradients', 'sigmoid', 'smallStructure', 'neptune']) 
    
    def binaryModel(curName, curInitializer, curActivation, x_tr=None):
        model = Sequential()      
        model.add(InputLayer(input_shape=(2, ), name=curName+"0"))
        model.add(Dense(3, kernel_initializer=curInitializer, activation=curActivation,  name=curName+"3"))
        model.add(Dense(1, kernel_initializer=curInitializer, activation='sigmoid', name=curName+"4"))
        return model 

    curOptimizer = tf.keras.optimizers.RMSprop()
    optimizer = curOptimizer
    curInitializer = RandomUniform(-1, 1)
    ## Compile the model
    model = binaryModel(curName="SIGMOID", curInitializer=curInitializer, curActivation="sigmoid")  
    model.compile(optimizer=curOptimizer, loss='binary_crossentropy', metrics=['accuracy'])
    ## Train and Log in Neptune    
    curGradHist, curLossHist, curWgtHist = fitModel(X, y, model, optimizer=curOptimizer, npt_exp=npt_exp)
    ## log in the plot comparing all layers
    npt_exp['Comparing All Layers'].upload(neptune.types.File.as_image(gradientsVis(curGradHist, curLossHist, modelName='Sigmoid_Small')))
    npt_exp.stop()  

elif MODELNAME == 'vanBN':
    npt_exp = neptune.init(    
    api_token=os.getenv('NEPTUNE_API_TOKEN'),
    project=myProject, 
    name='VanishingGradBatchNorm', 
    tags=['vanishingGradients', 'relu', 'batchNormalization', 'neptune'])
    
    def binaryModel(curName, curInitializer, curActivation, x_tr=None):
        model = Sequential()      
        model.add(InputLayer(input_shape=(2, ), name=curName+"0"))
        model.add(BatchNormalization())
        model.add(Dense(10, kernel_initializer=curInitializer, activation=curActivation, name=curName+"1"))  
        model.add(Dense(10, kernel_initializer=curInitializer, activation=curActivation, name=curName+"2"))        
        # model.add(BatchNormalization())
        model.add(Dense(5, kernel_initializer=curInitializer, activation=curActivation, name=curName+"3"))
        model.add(Dense(1, kernel_initializer=curInitializer, activation=curActivation, name=curName+"4"))
        return model
    curOptimizer = tf.keras.optimizers.RMSprop()
    optimizer = curOptimizer
    curInitializer = RandomUniform(-1, 1)
    model = binaryModel(curName="SIGMOID", curInitializer=curInitializer, curActivation="sigmoid")  
    model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    ## Train and Log in Neptune    
    curGradHist, curLossHist, curWgtHist = fitModel(X, y, model, optimizer=curOptimizer, npt_exp=npt_exp)
    ## log in the plot comparing all layers
    npt_exp['Comparing All Layers'].upload(neptune.types.File.as_image(gradientsVis(curGradHist, curLossHist, 
                                                                                    modelName='Sigmoid_BN')))
    npt_exp.stop()  
    
elif MODELNAME == 'vanSigWgtInit':
    npt_exp = neptune.init(    
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        project=myProject, 
        name='VanishingGradSigmoidWgtInit2', 
        tags=['vanishingGradients', 'sigmoid', 'weightInit2', 'neptune']) 
    
    ## Define Neptune callback 
    neptune_cbk = NeptuneCallback(run=npt_exp, base_namespace='metrics')
    def binaryModel(curName, curInitializer, curActivation, x_tr=None):
        model = Sequential()
        model.add(InputLayer(input_shape=(2, ), name=curName+"0"))
        model.add(Dense(10, kernel_initializer=curInitializer, activation=curActivation, name=curName+"1"))
        model.add(Dense(10, kernel_initializer=curInitializer, activation=curActivation, name=curName+"2"))
        model.add(Dense(5, kernel_initializer=curInitializer, activation=curActivation,  name=curName+"3"))
        model.add(Dense(1, kernel_initializer=curInitializer, activation='sigmoid', name=curName+"4"))
        return model 
    
    curOptimizer = tf.keras.optimizers.RMSprop()
    optimizer = curOptimizer
    ### Weight needs variance 
    # curInitializer = RandomNormal(mean=0, stddev=9) ## works better than the glorot for this data
    curInitializer = 'glorot_uniform'  ## glorot_uniform + Relu good!
    ## Compile the model
    model = binaryModel(curName="SIGMOID", curInitializer=curInitializer, curActivation="sigmoid") ## sigmoid  
    model.compile(optimizer=curOptimizer, loss='binary_crossentropy', metrics=['accuracy'])
    ## Train and Log in Neptune    
    curGradHist, curLossHist, curWgtHist = fitModel(X, y, model, optimizer=curOptimizer, npt_exp=npt_exp)
    print("Final Accuracy", accuracy_score(y, (model(X) > 0.5)))
    ## log in the plot comparing all layers
    npt_exp['Comparing All Layers'].upload(neptune.types.File.as_image(gradientsVis(curGradHist, curLossHist, 
                                                                                    modelName='Sigmoid_NormalWeightInit')))
    npt_exp.stop()  

elif MODELNAME == 'vanSigLR':
    npt_exp = neptune.init(    
        api_token=os.getenv('NEPTUNE_API_TOKEN'),
        project=myProject, 
        name='VanishingGradSigmoidLR2', 
        tags=['vanishingGradients', 'sigmoid', 'lr2', 'neptune'])
    
    def binaryModel(curName, curInitializer, curActivation, x_tr=None):
        model = Sequential()
        model.add(InputLayer(input_shape=(2, ), name=curName+"0"))
        model.add(Dense(10, kernel_initializer=curInitializer, activation=curActivation, name=curName+"1"))
        model.add(Dense(10, kernel_initializer=curInitializer, activation=curActivation, name=curName+"2"))
        model.add(Dense(5, kernel_initializer=curInitializer, activation=curActivation,  name=curName+"3"))
        model.add(Dense(1, kernel_initializer=curInitializer, activation='sigmoid', name=curName+"4"))
        return model 
    
    curOptimizer = keras.optimizers.Adam(learning_rate=0.008) ## reduce the learning rate!
    curInitializer = RandomUniform(-1, 1)
    ## Compile the model
    model = binaryModel(curName="SIGMOID", curInitializer=curInitializer, curActivation="sigmoid")  
    model.compile(optimizer=curOptimizer, loss='binary_crossentropy', metrics=['accuracy'])
    ## Train and Log in Neptune    
    curGradHist, curLossHist, curWgtHist = fitModel(X, y, model, optimizer=curOptimizer, npt_exp=npt_exp)
    ## log in the plot comparing all layers
    npt_exp['Comparing All Layers'].upload(neptune.types.File.as_image(gradientsVis(curGradHist, curLossHist, modelName='Sigmoid_Raw')))
    npt_exp.stop()  
    
   
############==========================================#############
############====== Exploding Gradients ===============#############
############==========================================#############
from sklearn.datasets import make_regression
from keras import regularizers
from tensorflow.keras.models import Model

# MODELNAME = 'expReluBaseline'
# MODELNAME = 'expGlorotInit'
MODELNAME = 'expL2Reg'
# MODELNAME = 'expGradClip'

if MODELNAME == 'expReluBaseline':    
    ### (1) change weight initi; 2) L2 norm; 3) gradients clipping
    npt_exp = neptune.init(    
            api_token=os.getenv('NEPTUNE_API_TOKEN'),
            project=myProject, 
            name='ExplodingGradRelu', 
            tags=['explodingGradients', 'relu', 'neptune'])
    
elif MODELNAME == 'expGlorotInit':
    npt_exp = neptune.init(    
            api_token=os.getenv('NEPTUNE_API_TOKEN'),
            project=myProject, 
            name='ExplodingGradGlorot', 
            tags=['explodingGradients', 'glorot', 'neptune'])
    
elif MODELNAME == 'expL2Reg':    
    npt_exp = neptune.init(    
            api_token=os.getenv('NEPTUNE_API_TOKEN'),
            project=myProject, 
            name='ExplodingGradL2Reg', 
            tags=['explodingGradients', 'l2 reg', 'neptune'])
    
elif MODELNAME == 'expGradClip':
    npt_exp = neptune.init(    
            api_token=os.getenv('NEPTUNE_API_TOKEN'),
            project=myProject, 
            name='ExplodingGradClip', 
            tags=['explodingGradients', 'gradients', 'clipping', 'neptune'])

# Generate regression dataset
nfeatures = 15
X, y = make_regression(n_samples=1500, n_features=nfeatures, noise=0.2, random_state=42)

# Define the regression model 
def regressionModel(X, y, curInitializer, USE_L2REG, secondLayerAct='relu'):  
    ## Construct the neural nets
    inp = Input(shape = (X.shape[1],)) 
    if USE_L2REG:
        ## need to change activation function as well
        x = Dense(35, activation='tanh', kernel_initializer=curInitializer, 
                  kernel_regularizer=regularizers.l2(0.01),
                  activity_regularizer=regularizers.l2(0.01))(inp)  
    else:
        x = Dense(35, activation=secondLayerAct, kernel_initializer=curInitializer)(inp)  
        
    out = Dense(1, activation='linear')(x)
    model = Model(inp, out)
    return model      
    
if MODELNAME == 'expReluBaseline': 
    sgd = tf.keras.optimizers.SGD()
    curOptimizer = sgd
    
    #### ! Uniform init 
    curInitializer = RandomUniform(4,5) 
    
    model = regressionModel(X, y, curInitializer, USE_L2REG=False)
    model.compile(loss='mean_squared_error', optimizer=curOptimizer, metrics=['mse'])
    
    curModelName = 'Relu_Raw'
    
elif MODELNAME == 'expGlorotInit':
    sgd = tf.keras.optimizers.SGD()
    curOptimizer = sgd
    
    #### ! Glorot init 
    curInitializer = 'glorot_normal'
    # curInitializer = 'glorot_uniform'
    
    model = regressionModel(X, y, curInitializer, USE_L2REG=False, secondLayerAct='tanh')
    model.compile(loss='mean_squared_error', optimizer=curOptimizer, metrics=['mse'])
    
    curModelName = 'GlorotInit'
    
elif MODELNAME == 'expL2Reg':
    sgd = tf.keras.optimizers.SGD()
    curOptimizer = sgd
    
    #### ! glorot init + L2 Reg
    # curInitializer = RandomUniform(4,5) 
    curInitializer = 'glorot_normal'
    model = regressionModel(X, y, curInitializer, USE_L2REG=True)
    model.compile(loss='mean_squared_error', optimizer=curOptimizer, metrics=['mse'])
    
    curModelName = 'L2Reg'
    
elif MODELNAME == 'expGradClip':    
    #### !Gradients clipping
    # sgd = tf.keras.optimizers.SGD(clipnorm=1.0)
    sgd = tf.keras.optimizers.SGD(clipvalue=50)

    curOptimizer = sgd
    
    #### ! Glorot init + L2 Reg
    # curInitializer = RandomUniform(4,5) 
    curInitializer = 'glorot_normal'

    model = regressionModel(X, y, curInitializer, USE_L2REG=False)
    model.compile(loss='mean_squared_error', optimizer=curOptimizer, metrics=['mse'])
    curModelName = 'GradClipping'    
    
## Train and Log in Neptune    
curGradHist, curLossHist, curWgtHist = fitModel(X, y, model, optimizer=curOptimizer,
                                                modelType = 'regression',
                                                npt_exp=npt_exp)

npt_exp['Comparing All Layers'].upload(neptune.types.File.as_image(gradientsVis(curGradHist, curLossHist, 
                                                                                modelName=curModelName)))
npt_exp.stop()  








