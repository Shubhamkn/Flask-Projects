# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# fetching the csv file to a pandas dataframe
Bangalore_dataset = pd.read_csv('Bangalore.csv')

# checking the dataset shape, parameters
Bangalore_dataset.shape

# Bangalore_dataset.info()

# other than price,Area,No.of bedrooms
# 0 - Absent, 1 - Present, 9 - Not mentioned
Bangalore_dataset.describe()

"""After checking the description of the dataset, it can be said that majority of the parameters don't 
have appropraite values so including them may bring about misleading results, so  in this case we won't be taking 
that into considerations"""

Bangalore_dataset.head()

# observing the Price vs Area plot, this will help in determinine non important data
Price = np.log(Bangalore_dataset['Price'])
sns.regplot(x = "Area", y = Price, data = Bangalore_dataset, fit_reg = False)

Bangalore_dataset.drop(Bangalore_dataset[Bangalore_dataset['Area']>=4000].index,inplace = True)
Price = np.log(Bangalore_dataset['Price'])
sns.regplot(x = "Area", y = Price, data = Bangalore_dataset, fit_reg = False)

sns.countplot(x = Bangalore_dataset['JoggingTrack'],data = Bangalore_dataset)

sns.countplot(x = Bangalore_dataset['JoggingTrack'],data = Bangalore_dataset)

sns.countplot(x = Bangalore_dataset['Gymnasium'],data = Bangalore_dataset)

sns.countplot(x = Bangalore_dataset['IndoorGames'],data = Bangalore_dataset)

sns.countplot(x = Bangalore_dataset['SwimmingPool'],data = Bangalore_dataset)

"""From the rest of the dataset also it can be said that a  lot of data is missing so we may need to delete a majority of data"""

Bangalore_dataset.replace(9,np.nan,inplace=True)
Bangalore_dataset.dropna(axis = 0,how="any",inplace=True)

# correlation = Bangalore_dataset.corr() # checking the correlation
# plt.figure(figsize = (60,60))
# we need heatmap from seaborn
# sns.heatmap(correlation, cbar = True,square=True, fmt='.3f' ,annot=True, annot_kws={'size':9},cmap = 'Blues')

"""this is how we can get the correlation between various parameters."""

# print(correlation['Price'] * 100)

# selected = correlation[(correlation.Price * 100 > 25.0) | (correlation.Price * 100 < -25.0)]
# print(len(selected))
# selected['Price'] * 100

# counting duplicates and removing them later
Bangalore_dataset.duplicated().sum()

# getting rid of the duplicates and verify
Bangalore_dataset.drop_duplicates(inplace=True)
Bangalore_dataset.shape

Bangalore_dataset.drop(["MaintenanceStaff","Gymnasium","LandscapedGardens","RainWaterHarvesting","IndoorGames",
                        "Intercom","PowerBackup","Cafeteria","MultipurposeRoom","Wifi","Children'splayarea","LiftAvailable",
                        "VaastuCompliant","GolfCourse","Wardrobe","ShoppingMall","ATM","School","24X7Security"],axis=1,inplace=True)

Bangalore_dataset

Bangalore_dataset.describe()

# label encoding the locations before moving on to the splitting
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Bangalore_dataset["Location"] = le.fit_transform(Bangalore_dataset["Location"])

Bangalore_dataset.describe()

# now the dependent variables and independent variables matrices
X = Bangalore_dataset.iloc[:,1:].values
y = Bangalore_dataset.iloc[:,0].values
# y = y/100000   # converting prices into lakhs
# print(X)
# print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

LR = LinearRegression()
KNR = KNeighborsRegressor()
DTR = DecisionTreeRegressor(random_state = 0)
RFR = RandomForestRegressor(n_estimators = 100,random_state = 0)
XGBR = XGBRegressor(max_depth = 100, learning_rate = 0.01, n_estimators = 500)
LR.fit(X_train,y_train)
KNR.fit(X_train,y_train)
DTR.fit(X_train, y_train)
RFR.fit(X_train, y_train)
XGBR.fit(X_train, y_train)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

models = [LR, KNR, DTR, RFR, XGBR]
models_name = ['LR','KNR','DTR','RFR','XGBR']
# for i in range (5):
#   print(models_name[i])
#   print('Training')
#   y_pred = models[i].predict(X_train)
#   print('r2_score: ', r2_score(y_train, y_pred))
#   print('root_mean_square_error: ', mean_squared_error(y_train, y_pred, squared = False))
#   print('mean_absolute_error: ', mean_absolute_error(y_train, y_pred))
#   print('mean_absolute_percentage_error: ', mean_absolute_percentage_error(y_train, y_pred))
#   print()
#   print('Testing')
#   y_pred = models[i].predict(X_test)
#   print('r2_score: ', r2_score(y_test, y_pred))
#   print('root_mean_square_error: ', mean_squared_error(y_test, y_pred, squared = False))
#   print('mean_absolute_error: ', mean_absolute_error(y_test, y_pred))
#   print('mean_absolute_percentage_error: ', mean_absolute_percentage_error(y_test, y_pred))
#   print()

# print(le.classes_)

# print(le.transform(['Amruthahalli', 'Anagalapura Near Hennur Main Road']))

# print(le.inverse_transform([20,33,40,50,104]))