import pandas as pd

data = pd.read_csv('D:/Anaconda/datasets/boston/Boston.csv')

colnames = data.columns.values.tolist()

predictors = colnames[:13]
target = colnames[13]

x=data[predictors]
y=data[target]

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(min_impurity_split=1,min_samples_leaf=1,random_state=0)

tree.fit(x,y)

prediccion = tree.predict(x)

data['prediccion'] = prediccion

#COMPARO LOS VALORES REALES CON LA PREDICCION

print(data[['prediccion','medv']])

print(tree.score(x, y))

