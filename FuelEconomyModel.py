# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#importing pacakages
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

#Function to do label encoding
def LabelEncode(dfData,szColName):
    dfEncodeData=dfData[[szColName]]
    dfEncodeData=dfEncodeData.drop_duplicates()
    dfEncodeData=dfEncodeData.reset_index(drop=True).reset_index()
    dfEncodeData=dfEncodeData.rename(columns={'index':'ID'})
    dictEncodeData=dfEncodeData.set_index(szColName)['ID'].to_dict()
    return dictEncodeData

#Taking a look at the coefficents
def coef_weights(coefficients, X_train):
    '''
    INPUT:
    coefficients - the coefficients of the linear model 
    X_train - the training data, so the column names can be used
    OUTPUT:
    coefs_df - a dataframe holding the coefficient, estimate, and abs(estimate)
    
    Provides a dataframe that can be used to understand the most influential coefficients
    in a linear model by providing the coefficient estimates along with the name of the 
    variable attached to the coefficient.
    '''
    coefs_df = pd.DataFrame()
    coefs_df['est_int'] = X_train.columns
    coefs_df['coefs'] = lm_model.coef_
    coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
    coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
    return coefs_df

def GenerateModelInput(dictInput):
    dfX=pd.DataFrame(dictInput)
    dfX['FUEL_TYPE_ID'] = dfX['FUEL_TYPE'].map(dictFuelType)
    dfX['CLASS_TYPE_ID'] = dfX['CLASS_TYPE'].map(dictClassType)
    dfX['TRANSMISSION_TYPE_ID'] = dfX['TRANSMISSION_TYPE'].map(dictTransType)
    dfX=dfX.drop(['FUEL_TYPE','CLASS_TYPE','TRANSMISSION_TYPE'], axis = 1)
    return dfX.values


#Reading the Dataset
dfData=pd.read_csv('Data\database.csv',low_memory=False)

#DATA CLEANSING
#Only selecting columns that are needed for this analysis
liSelectedColumns=['Year','Make','Class','Transmission','Fuel Type 1','City MPG (FT1)','Highway MPG (FT1)','Gasoline/Electricity Blended (CD)']
dfSelData=dfData[liSelectedColumns].copy()

#Since we are only interested in gasoline vehicles, let's filter for that
dfSelData=dfSelData[dfSelData['Gasoline/Electricity Blended (CD)']==False]
del dfSelData['Gasoline/Electricity Blended (CD)']
dfSelData=dfSelData[dfSelData['Fuel Type 1'].str.contains("Gasoline")]

#Checking for missing data
dfMissingData=dfSelData[dfSelData.Transmission.isnull()]
len(dfMissingData)
#Only 2 missing data points so can eliminate those rows
dfSelData=dfSelData[~dfSelData.Transmission.isnull()]
len(dfSelData)

#Multiple transmission types
#Encoding it down to the two major types to keep model simple
dfSelData['Transmission_Type']=dfSelData['Transmission'].apply(lambda x: 'Manual' if 'Manual' in x else 'Automatic')

#The transmission encoding key
dfTransmissionCheck=dfSelData[['Transmission','Transmission_Type']]
dfTransmissionCheck=dfTransmissionCheck.drop_duplicates()

#Cleaning Up column class to a manageable level
#The class type encoding key
dictClassMap={'Minicompact Cars':'Car',
'Two Seaters':'Car',
'Special Purpose Vehicle 2WD':'SPV',
'Special Purpose Vehicle 4WD':'SPV',
'Subcompact Cars':'Car',
'Midsize Cars':'Car',
'Midsize Station Wagons':'Car',
'Compact Cars':'Car',
'Midsize-Large Station Wagons':'Car',
'Large Cars':'Car',
'Small Station Wagons':'Car',
'Standard Pickup Trucks 2WD':'Truck',
'Vans, Passenger Type':'Van',
'Vans, Cargo Type':'Van',
'Standard Pickup Trucks 4WD':'Truck',
'Special Purpose Vehicles':'SPV',
'Small Pickup Trucks 2WD':'Truck',
'Small Pickup Trucks 4WD':'Truck',
'Vans':'Van',
'Standard Pickup Trucks':'Truck',
'Small Pickup Trucks':'Truck',
'Vans Passenger':'Van',
'Standard Pickup Trucks/2wd':'Truck',
'Special Purpose Vehicles/2wd':'SPV',
'Special Purpose Vehicles/4wd':'SPV',
'Sport Utility Vehicle - 4WD':'SUV',
'Sport Utility Vehicle - 2WD':'SUV',
'Minivan - 2WD':'Van',
'Minivan - 4WD':'Van',
'Small Sport Utility Vehicle 4WD':'SUV',
'Small Sport Utility Vehicle 2WD':'SUV',
'Standard Sport Utility Vehicle 4WD':'SUV',
'Standard Sport Utility Vehicle 2WD':'SUV'}

dfSelData['Class_Type'] = dfSelData['Class'].map(dictClassMap)

#Dont need the transmission and class columns anymore
dfSelData = dfSelData.drop(['Transmission','Class'], 1)

#Data cleaning up done
dfCleansedData=dfSelData.copy()

#Creating a model to predict fuel economy of a vehicle
dfCleansedData['AVG_ECONOMY_MPG']=(dfCleansedData['City MPG (FT1)']+dfCleansedData['Highway MPG (FT1)'])/2
dfModelData=dfCleansedData.copy()
dfModelData=dfModelData.groupby(['Year','Fuel Type 1','Class_Type','Transmission_Type'])['AVG_ECONOMY_MPG'].mean().reset_index()

#Three categorical columns
#Need to convert them to numeric to predict
#Going to use label encoding
dictFuelType=LabelEncode(dfModelData,'Fuel Type 1')
dictClassType=LabelEncode(dfModelData,'Class_Type')
dictTransType=LabelEncode(dfModelData,'Transmission_Type')

dfModelData['FUEL_TYPE_ID'] = dfModelData['Fuel Type 1'].map(dictFuelType)
dfModelData['CLASS_TYPE_ID'] = dfModelData['Class_Type'].map(dictClassType)
dfModelData['TRANSMISSION_TYPE_ID'] = dfModelData['Transmission_Type'].map(dictTransType)
dfModelData=dfModelData.drop(['Fuel Type 1','Class_Type','Transmission_Type'], axis = 1)

response_col='AVG_ECONOMY_MPG'

#Split into explanatory and response variables
X = dfModelData.drop(response_col, axis=1)
y = dfModelData[response_col]

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=11)

lm_model = LinearRegression(normalize=True) # Instantiate
lm_model.fit(X_train, y_train) #Fit

#Predict using your model
y_test_preds = lm_model.predict(X_test)
y_train_preds = lm_model.predict(X_train)

#Score using your model
test_score = r2_score(y_test, y_test_preds)
train_score = r2_score(y_train, y_train_preds)
print("The R Square for Training data is:"+str(train_score))
print("The R Square for Test data is:"+str(test_score))



#Use the function to look at the coefficients of the model
coef_df = coef_weights(lm_model.coef_, X_train)
print("\nThe coefficients are:\n")
print(coef_df)
print("\n")


#Predicting the fuel economy of an user input vehicle
dictInput=[{'Year':2000,'FUEL_TYPE':'Regular Gasoline','CLASS_TYPE':'Car','TRANSMISSION_TYPE':'Automatic'},
           {'Year':2000,'FUEL_TYPE':'Regular Gasoline','CLASS_TYPE':'Car','TRANSMISSION_TYPE':'Manual'}
          ]

predMPG=lm_model.predict(GenerateModelInput(dictInput))

for i in range(0,len(predMPG)):
    print("The fuel economy for vehicle"+str(i+1)+" is:"+str(round(predMPG[i],0))+" mpg")

