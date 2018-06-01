#!/usr/bin/python

import pandas as pd
import numpy as np
from tensorflow.contrib import keras
from sklearn.metrics import mean_absolute_error
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import argparse


def train(model_name):
    melbourne_file_path = './melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    melbourne_data = melbourne_data.dropna(axis=0)
    y = melbourne_data.Price
    input_features = ['Address','Bathroom','Bedroom2','BuildingArea','Car','CouncilArea', 'Date', 'Distance', 'Landsize', 'Lattitude', 'Longtitude', 'Method', 'Postcode', 'Price', 'Propertycount', 'Regionname', 'Rooms', 'SellerG', 'Suburb', 'Type', 'YearBuilt']
    #input_features = ['Bathroom','Bedroom2','BuildingArea','Car','Distance', 'Landsize', 'Lattitude', 'Longtitude', 'Postcode', 'Price', 'Propertycount', 'Rooms' ]

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

    X = melbourne_data[input_features]
    trainX, ValX, trainY, ValY = train_test_split(X,y)

    # Building model: 
    # TODO: Add Selu + Alpha dropout
    melbourne_model = keras.models.Sequential()
    melbourne_model.add(keras.layers.Dense(20, activation='tanh',input_dim=len(input_features)))
    melbourne_model.add(keras.layers.Dense(20, activation='tanh'))
    melbourne_model.add(keras.layers.Dense(1, activation='relu'))   # MAE: 922165

    #melbourne_model.add(keras.layers.Dense(1, activation='relu',input_dim=len(input_features))) # MAE: 1072223

    # TODO: Add some metric
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True) # Does not train, flat loss chart
    #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    #sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9)
    melbourne_model.compile(loss="mean_squared_error", optimizer=sgd)
    #melbourne_model.compile(loss="mean_squared_error", optimizer="adagrad")
    #melbourne_model.compile(loss="mse", optimizer="rmsprop")

    show_stopper = keras.callbacks.EarlyStopping(monitor='val_loss',patience=10, verbose=1)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath="saved_models/melbourne_model.epoch-{epoch:02d}-val_loss-{val_loss:.4f}.hdf5",monitor='val_loss',save_best_only=True,verbose=1)

    history = melbourne_model.fit(trainX.values, trainY.values, validation_split=0.2, epochs=30, batch_size=1,callbacks=[show_stopper,checkpoint])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Loss/cost chart")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend(["train","val"],loc="upper right")
    plt.show()
    print("Making predictions for following houses")
    predictions = melbourne_model.predict(ValX.values)

    print("MAE:",mean_absolute_error(predictions,ValY.values))

#import pdb; pdb.set_trace()
# The Melbourne data has somemissing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the predictors you use. 
# So we will take the simplest option for now, and drop those houses from our data. 
#Don't worry about this much for now, though the code is:

#7 dropna drops missing values (think of na as "not available")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="Perform training", action="store_true")
    parser.add_argument("--infer", help="Perform evaluation", action="store_true")
    parser.add_argument("--model", help="Model to be used for training/inference", type=str, default="")
    args = parser.parse_args()
    if args.train == True:    
        train(args.model)
    elif args.infer == True:
        infer(args.model)
    else:
        print("Error: Please specify either train of infer commandline option")
        exit(1)




