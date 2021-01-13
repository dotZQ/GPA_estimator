import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

df = pd.read_csv('/Users/zq314159/Downloads/GPA.csv')

train , test = train_test_split(df,test_size=0.2,random_state=0)
x = train['SAT'].to_frame()
y = train['GPA'].to_frame()
x_test = test['SAT'].to_frame()
y_test = test['GPA'].to_frame()
model = LinearRegression()
model.fit(x,y)
score = model.score(x_test,y_test)
score
SAT_input = float(input("input SAT score :"))
sat = [[SAT_input]]
result = model.predict(sat)
print("GPA : ",result)
