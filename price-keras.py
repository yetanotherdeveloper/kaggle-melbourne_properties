#!/usr/bin/python

import pandas as pd
import numpy as np
from tensorflow.contrib import keras
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import argparse

def load_and_preprocess_data(melbourne_file_path):
    melbourne_data = pd.read_csv(melbourne_file_path)
    melbourne_data = melbourne_data.dropna(axis=0)

    # Replace string values of CouncilArea with onehott representation
    council = melbourne_data['CouncilArea']
    integer_encoding = LabelEncoder().fit_transform(council)
    melbourne_data['CouncilArea'] = integer_encoding
    # Replace suburb name with integer encoding value
    suburb =  melbourne_data['Suburb']
    integer_encoding = LabelEncoder().fit_transform(suburb)
    melbourne_data['Suburb'] = integer_encoding
    # Type of property
    typedata =  melbourne_data['Type']
    integer_encoding = LabelEncoder().fit_transform(typedata)
    melbourne_data['Type'] = integer_encoding
    # enocode regionname 
    region =  melbourne_data['Regionname']
    integer_encoding = LabelEncoder().fit_transform(region)
    melbourne_data['Regionname'] = integer_encoding
    # enocode SellerG 
    seller =  melbourne_data['SellerG']
    integer_encoding = LabelEncoder().fit_transform(seller)
    melbourne_data['SellerG'] = integer_encoding
    # enocode Method 
    method =  melbourne_data['Method']
    integer_encoding = LabelEncoder().fit_transform(method)
    melbourne_data['Method'] = integer_encoding
    # preprocess adress, remove apartmetn number , leaving just street
    address =  melbourne_data['Address']
    for i in range(0,len(address.values)):
        address.values[i] = address.values[i][address.values[i].find(" ")+1:]
    integer_encoding = LabelEncoder().fit_transform(address)
    melbourne_data['Address'] = integer_encoding

    # Day + month*32 + 32*12*year
    datas = melbourne_data['Date']
    for i in range(0,len(datas.values)):
        splits = datas.values[i].split('/')
        datas.values[i] = int(splits[0]) + int(splits[1])*32 + int(splits[2])*32*12
    return melbourne_data

def make_model(model_name,num_features):
    # Building model: 
    adam = keras.optimizers.Adam() # Does converge slowly 
    melbourne_model = keras.models.Sequential(name="-"+model_name + "-Adam")
    melbourne_model.add(keras.layers.Dense(20, activation='tanh',input_dim=num_features))
    melbourne_model.add(keras.layers.Dense(20, activation='tanh'))
    melbourne_model.add(keras.layers.Dense(1, activation='relu'))   # MAE: 922165

    #sgd = keras.optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9) # MAE: 460326
 #   melbourne_model.compile(loss="mean_squared_error", optimizer='rmsprop') # Does converge slowly
    melbourne_model.compile(loss="mean_squared_error", optimizer=adam) # Does converge slowly
    return melbourne_model

# TODO: figure out how to initialize bias for SNN
def make_selu_model(model_name,num_features):
    melbourne_model = keras.models.Sequential(name="-"+model_name+"-SGD")
    melbourne_model.add(keras.layers.Dense(20, kernel_initializer='lecun_normal',bias_initializer='lecun_normal',
        activation='selu',input_dim=num_features))
    melbourne_model.add(keras.layers.Dense(20, kernel_initializer='lecun_normal',bias_initializer='lecun_normal',
        activation='selu'))
    melbourne_model.add(keras.layers.Dense(1, kernel_initializer='lecun_normal', bias_initializer='lecun_normal',
        activation='selu'))   # MAE: 

    #adam = keras.optimizers.Adam() # Does converge slowly 
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9) # MAE: 
    melbourne_model.compile(loss="mean_squared_error", optimizer=sgd) # Does converge slowly
    return melbourne_model

def normalize_input(data,features):
    """ Modify data so it is zero meaned """
    # Get all data from selected columns across samples
    # TODO: remove warnings
    for col in features:
        scaler = StandardScaler().fit(data[col])
        data[col] = scaler.transform(data[col])

def train(model_name, num_epochs):
    melbourne_data = load_and_preprocess_data('./melb_data.csv')

    input_features = ['Address','Bathroom','Bedroom2','BuildingArea','Car','CouncilArea', 'Date', 'Distance', 'Landsize', 'Lattitude', 'Longtitude', 'Method', 'Postcode', 'Price', 'Propertycount', 'Regionname', 'Rooms', 'SellerG', 'Suburb', 'Type', 'YearBuilt']
    #input_features = ['Bathroom','Bedroom2','BuildingArea','Car','Distance', 'Landsize', 'Lattitude', 'Longtitude', 'Postcode', 'Price', 'Propertycount', 'Rooms' ]
    X = melbourne_data[input_features]
    y = melbourne_data.Price

    initial_epoch = 1
    if model_name == "" or model_name == "FFN": 
        model_name = "FFN"
        melbourne_model = make_model(model_name,len(input_features))
    elif model_name == "SNN":
        normalize_input(X,input_features)
        melbourne_model = make_selu_model(model_name,len(input_features))
    else:
        tmpstr = model_name[:model_name.find("-val_loss")]
        initial_epoch = int(tmpstr[tmpstr.rfind("-")+1:])
        melbourne_model = keras.models.load_model(model_name)
    num_epochs += initial_epoch - 1

    # TODO: Get name of model from compiled model and make
    # name of file to save to based on that
    # TODO: normalize input 
    show_stopper = keras.callbacks.EarlyStopping(monitor='val_loss',patience=num_epochs-10, verbose=1)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath="saved_models/melbourne_model"+melbourne_model.name+".epoch-{epoch:02d}-val_loss-{val_loss:.4f}.hdf5",monitor='val_loss',save_best_only=True,verbose=1)

    history = melbourne_model.fit(X.values, y.values, validation_split=0.2, epochs=num_epochs, initial_epoch=initial_epoch, batch_size=1,callbacks=[show_stopper,checkpoint])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Loss/cost chart")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend(["train","val"],loc="upper right")
    plt.show()
    print("Making predictions for following houses")
    predictions = melbourne_model.predict(X.values)

    print("MAE:",mean_absolute_error(predictions,y.values))

#import pdb; pdb.set_trace()
# The Melbourne data has somemissing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the predictors you use. 
# So we will take the simplest option for now, and drop those houses from our data. 
#Don't worry about this much for now, though the code is:

def infer(model_name):
    if model_name == "":
        print("Error: Inference mode require model given with --model option")
        exit(-1)
    melbourne_data = load_and_preprocess_data('./melb_data.csv')

    input_features = ['Address','Bathroom','Bedroom2','BuildingArea','Car','CouncilArea', 'Date', 'Distance', 'Landsize', 'Lattitude', 'Longtitude', 'Method', 'Postcode', 'Price', 'Propertycount', 'Regionname', 'Rooms', 'SellerG', 'Suburb', 'Type', 'YearBuilt']
    #input_features = ['Bathroom','Bedroom2','BuildingArea','Car','Distance', 'Landsize', 'Lattitude', 'Longtitude', 'Postcode', 'Price', 'Propertycount', 'Rooms' ]
    X = melbourne_data[input_features]
    y = melbourne_data.Price

    #melbourne_model.add(keras.layers.Dense(1, activation='relu',input_dim=len(input_features))) # MAE: 1072223
    melbourne_model = make_model(len(input_features))
    melbourne_model.load_weights(model_name)
    predictions = melbourne_model.predict(X.values)
    print("MAE:",mean_absolute_error(predictions,y.values))
        
    scores = melbourne_model.evaluate(X.values,y.values,verbose=0)
#    import pdb;pdb.set_trace()
    print("%s: %.2f" % (melbourne_model.metrics_names[0], scores))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Perform training", action="store_true")
    parser.add_argument("--infer", help="Perform evaluation", action="store_true")
    parser.add_argument("--model", help="Model to be used for training/inference", type=str, default="")
    parser.add_argument("--num_epochs", help="Number of epochs to perform", type=int, default=10)
    args = parser.parse_args()
    if args.train == True:    
        train(args.model,args.num_epochs)
    elif args.infer == True:
        infer(args.model)
    else:
        print("Error: Please specify either train of infer commandline option")
        exit(1)




