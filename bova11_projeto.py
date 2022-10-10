import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import seaborn as sns
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM 
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
yf.pdr_override()

# Importando dados
base_vale = web.get_data_yahoo('BOVA11.SA', start='2015-01-01')
# Removendo valores faltantes
base_vale = base_vale.dropna()
# Criando base de preços
base_vale = base_vale.iloc[:, 0:1]
# Criando base de treinamento
base_treino = base_vale.iloc[:1911].values

# Normalizando dados
normalizador = MinMaxScaler(feature_range=(0,1))
treino_normalizado = normalizador.fit_transform(base_treino)

# Vamos utilizar os últimos três meses como período para realizar as previsões para o próximo dia
previsores = []
preço_real_1d = []
for i in range(90, 1911):
    previsores.append(treino_normalizado[i-90:i, 0])
    preço_real_1d.append(treino_normalizado[i, 0])
    
    
# Formatando os dados para serem consumidores pela rede
previsores, preço_real_1d = np.array(previsores), np.array(preço_real_1d)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

# Criando e configurando a rede neural
regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = 'linear'))

regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['mean_absolute_error'])
regressor.fit(previsores, preço_real_1d, epochs = 100, batch_size = 32)

# Criando base de testes
preços_reais = base_vale.iloc[1911:].values
base_entradas = base_vale.iloc[:1911]
base_completa = base_vale.values
entradas_1d = base_completa[len(base_completa) - len(preços_reais) - 90:]
entradas_1d = entradas_1d.reshape(-1, 1)
entradas_1d = normalizador.transform(entradas_1d)
X_test_1d = []
for i in range(90, 112):
    X_test_1d.append(entradas_1d[i-90:i, 0])

# Prevendo os dados    
X_test_1d = np.array(X_test_1d)
X_test_1d = np.reshape(X_test_1d, (X_test_1d.shape[0], X_test_1d.shape[1], 1))
prediction_1d = regressor.predict(X_test_1d)
prediction_1d = normalizador.inverse_transform(prediction_1d)


# Vendo diferenças nas médias
print(prediction_1d.mean())
print(preços_reais.mean())

# Plotando
sns.set()
plt.figure(figsize=(20,10))
plt.plot(preços_reais, color = 'green', label = 'Preço real BOVA11')
plt.plot(prediction_1d, color = 'red', label = 'Previsão BOVA11')
plt.title('Previsão de preços BOVA11')
plt.xlabel('Tempo')
plt.ylabel('Preço BOVA11')
plt.legend()
plt.show()