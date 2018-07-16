

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas
import numpy as np
import matplotlib.pyplot as plt


#data retrive from dataset
data = pandas.read_csv('creditcard.csv')

#normalizando a coluna amount e retirando a coluna de tempo
data['normAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)

#Deixando dados 50% não fraudulentos e 50% não fraudulentos
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

normal_indices = data[data.Class == 0].index
fraud_indices = np.array(data[data.Class == 1].index)

random_normal_indices = np.array(np.random.choice(normal_indices, number_records_fraud, replace = False))

newDataIndices = np.concatenate([fraud_indices,random_normal_indices])
data = data.iloc[newDataIndices,:]

X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']

#Separando os dados de treinamento e de teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#realizando o treinamento do modelo

clf = MLPClassifier(solver='sgd')


clf.fit(X_train.values,y_train.values.ravel())

y_pred = clf.predict(X_test)
y_true = y_test.values.ravel()

accuracy_score(y_true, y_pred)
mean_squared_error(y_true, y_pred)




