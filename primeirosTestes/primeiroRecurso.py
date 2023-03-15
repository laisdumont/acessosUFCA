import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC  
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# Extraindo os dados tratados usando apenas o primeiro recurso
df = pd.read_csv('primeirosTestes/dadostratados.csv')

# Etiquetando os dados com duas classes a partir do Kmeans
kmeans = KMeans(n_clusters=2).fit(df)
pred = kmeans.predict(df)

# Adicionando as classes ao dataframe
df.insert(12,'classe',pred,True)

# Separando os atributos das classes
x = df.iloc[:,:-1].values
y = df.iloc[:,-1:].values.ravel()

# Normalizando os dados
min_max_scaler = MinMaxScaler()
x = min_max_scaler.fit_transform(x)

# Dividindo os dados em 5 partes iguais para treino e teste
folds = StratifiedKFold(n_splits=5, shuffle=True)

fold_accuracy = []
fold_matrix = []

# Treino e predição de dados
for train_index, test_index in folds.split(x, y):
    clf = SVC( )    # Classificador usado foi o SVM, com parâmetros padrão da funcão
    clf.fit(x[train_index],y[train_index])  # Classificação com os índices de treino do StratrifiedKFold
    predicted = clf.predict(x[test_index])  # Predição com os ídices de teste do StratrifiedKFold
    fold_matrix.append(metrics.confusion_matrix(y[test_index],predicted))   # Matrizes de confusão do resultadoXpredição
    fold_accuracy.append(metrics.accuracy_score(y[test_index],predicted))   # Acurácia de cada predição

print('Support Vector Machine')

print('Acurácias:',fold_accuracy)
print('Média das acurácias:',np.mean(fold_accuracy))
print('Desvio padrão das acurácias:',np.std(fold_accuracy))

print('Matrizes de Confusão') 
for i in fold_matrix:
    plt.figure(figsize=(5,3))
    sn.set(font_scale=1)
    sn.heatmap(i, annot=True, fmt=".1f")

    plt.show()