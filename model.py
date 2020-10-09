import numpy as np
import pandas as pd
import streamlit as st

import pickle

train_data = pd.read_csv('train_modified.csv')
test_data = pd.read_csv('test_modified.csv')

X_train=train_data.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'],axis=1)
y_train=train_data['Item_Outlet_Sales']
X_test=test_data.drop(['Item_Identifier','Outlet_Identifier'],axis=1).copy()

from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor(max_depth=15,min_samples_leaf=100)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)


pickle_out=open("model.pkl","wb")
pickle.dump(model, pickle_out)
pickle_out.close()

