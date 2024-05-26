

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

Delhi_dataset = pd.read_csv('Delhi.csv')

# checking the dataset shape, parameters
Delhi_dataset.shape

# Delhi_dataset.info()

# other than price,Area,No.of bedrooms, and resale
# 0 - Absent, 1 - Present, 9 - Not mentioned
# Delhi_dataset.describe()

# Delhi_dataset.head()

# observing the Price vs Area plot, this will help in determinine non important data
Price = np.log(Delhi_dataset['Price'])
sns.regplot(x = "Area", y = Price, data = Delhi_dataset, fit_reg = False)

Delhi_dataset.drop(Delhi_dataset[Delhi_dataset['Area']>=6000].index, inplace = True)

Price = np.log(Delhi_dataset['Price'])
sns.regplot(x = "Area", y = Price, data = Delhi_dataset, fit_reg = False)

sns.countplot(x = Delhi_dataset['Cafeteria'],data = Delhi_dataset)

sns.countplot(x = Delhi_dataset['IndoorGames'],data = Delhi_dataset)

sns.countplot(x = Delhi_dataset['MaintenanceStaff'],data = Delhi_dataset)

sns.countplot(x = Delhi_dataset['Gymnasium'],data = Delhi_dataset)

sns.countplot(x = Delhi_dataset['SwimmingPool'],data = Delhi_dataset)

"""the number of not mentioned data is almost 60% so it won't help in prediction so we will remove the rows with not mentioned data"""

Delhi_dataset.replace(9,np.nan,inplace=True)
Delhi_dataset.dropna(axis=0,how="any",inplace=True)

# correlation = Delhi_dataset.corr() # checking the correlation
# # plt.figure(figsize = (60,60))
# # we need heatmap from seaborn
# # sns.heatmap(correlation, cbar = True,square=True, fmt='.3f' ,annot=True, annot_kws={'size':9},cmap = 'Blues')

# print(correlation['Price'])

"""this correlation gives theimportant parameters that will be needed for the prediction"""

# counting duplicates and removing them later
Delhi_dataset.duplicated().sum()

# getting rid of the duplicates and verify
Delhi_dataset.drop_duplicates(inplace=True)
Delhi_dataset.shape

# label encoding the locations before moving on to the splitting
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Delhi_dataset["Location"] = le.fit_transform(Delhi_dataset["Location"])

Delhi_dataset.drop(["MaintenanceStaff", "ShoppingMall","ATM","School","StaffQuarter","Cafeteria","MultipurposeRoom",
                   "WashingMachine","Wifi","GolfCourse","Wardrobe","TV","DiningTable","Sofa","BED"], axis = 1,inplace=True)

Delhi_dataset.describe()

# now the dependent variables and independent variables matrices
X = Delhi_dataset.iloc[:,1:].values
y = Delhi_dataset.iloc[:,0].values
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
XGBR = XGBRegressor(max_depth = 80, learning_rate = 0.01, n_estimators = 1000)
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