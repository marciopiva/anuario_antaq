#base
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#tensorflow
import tensorflow as tf
from tensorflow.keras import layers

#trabalhando com 4 classes alvo
def seta_classe(df, coluna_ordem, coluna_classe):
    quartil_1 = df[coluna_ordem].quantile(q=0.25, interpolation='linear')
    quartil_2 = df[coluna_ordem].quantile(q=0.50, interpolation='linear')
    quartil_3 = df[coluna_ordem].quantile(q=0.75, interpolation='linear')
    quartil_4 = df[coluna_ordem].quantile(q=0.85, interpolation='linear') 

    for _, row in df.iterrows():
        if row[coluna_ordem] >= 1 and row[coluna_ordem] <= quartil_1:
            row[coluna_classe] = 0
        elif row[coluna_ordem] > quartil_1 and row[coluna_ordem] <= quartil_2:
            row[coluna_classe] = 1
        elif row[coluna_ordem] > quartil_2 and row[coluna_ordem] <= quartil_3:
            row[coluna_classe] = 2
        elif row[coluna_ordem] > quartil_3 and row[coluna_ordem] <= quartil_4:
            row[coluna_classe] = 3

def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)

    plt.legend()
    plt.show()


def create_model(my_learning_rate, atributos_entrada, atributos):

    # Modelo
    model = tf.keras.models.Sequential()

    # Camada de entrada - estímulos
    #camada_entrada = tf.keras.layers.DenseFeatures(atributos_entrada)(atributos)
    #model.add(camada_entrada)
    model.add(tf.keras.Input(shape=(5,)))

    # Camadas escondidas
    model.add(tf.keras.layers.Dense(units=512, activation='relu', name='Primeira_Camada'))
    model.add(tf.keras.layers.Dense(units=256, activation='relu', name='Segunda_Camada'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu', name='Terceira_Camada'))

    # Regularização
    #model.add(tf.keras.layers.Dropout(rate=0.1))

    # quatro classes de saída
    model.add(tf.keras.layers.Dense(units=4, activation='softmax', name='Camada_Saida'))     
                            
    # Construct the layers into a model that TensorFlow can execute.  
    # Notice that the loss function for multi-class classification
    # is different than the loss function for binary classification.  
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    return model    


def train_model(model, train_features, train_label, epochs,
                batch_size, validation_split):

    history = model.fit(x=train_features, y=train_label, batch_size=batch_size,
                        epochs=epochs, shuffle=True, 
                        validation_split=validation_split)

    # To track the progression of training, gather a snapshot
    # of the model's metrics at each epoch. 
    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist        


# miscilêneas
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format
np.set_printoptions(linewidth = 200)

# percentual para split treino, teste e validação
split_treino_percentual = 0.8 
split_validacao_percentual = 0.0

# carregando dataset completo - gerado pelas views do sqlite - vw_dataset_cg + vw_dataset_gs + vw_dataset_gl
df_atraca = pd.read_csv('../csv/atraca.csv', low_memory=False)
df_atraca.set_index('idatracacao')

# filtrando Santos / Movimentação de Carga / Longo Curso 
df_atraca = df_atraca.loc[df_atraca['cd_portoatracacao'] == 11]      #Santos
df_atraca = df_atraca.loc[df_atraca['cd_tipooperacao'] == 2]         #Movimentação da Carga
df_atraca = df_atraca.loc[df_atraca['cd_tiponavegacaoatracao'] == 3] #Longo Curso

# removendo tempos "inválidos"
df_atraca = df_atraca.loc[df_atraca['toperacao'] > 0]                #Remover tempos zerados
df_atraca = df_atraca.loc[df_atraca['testadia'] > 0]                 #Remover tempos zerados
df_atraca.set_index('idatracacao')
df_atraca = df_atraca.reindex(np.random.permutation(df_atraca.index))

# separando datasets por tipo de carga devido aos comportamentos diferentes nas atracações
df_cg = df_atraca.loc[df_atraca['cd_tipocarga'] == 1] # carga geral
df_gs = df_atraca.loc[df_atraca['cd_tipocarga'] == 2] # granel sólido
df_gl = df_atraca.loc[df_atraca['cd_tipocarga'] == 3] # granel líquido e gasoso
df_cg_cs = df_cg.loc[df_cg['teu'] == 0]               # somente carga solta
df_cg_cntr = df_cg.loc[df_cg['teu'] > 0]              # somente container

# setando df para trabalho
df_base = df_cg_cs

# eliminando outliers e criando 4 classes 
df_base['classe_toperacao'] = 5
seta_classe(df_base, 'toperacao', 'classe_toperacao')
df_base = df_base.loc[df_base['classe_toperacao'] != 5]

# atributos e rotulo para treino
atributos = ['cd_berco', 'cd_ano', 'cd_mes','cd_terminal', 'pesocargabruta']
rotulo = ['classe_toperacao']

# separar os atributos e rotulo 
X_df = df_base[atributos]
y_df = df_base[rotulo]

# normalizando do dataset com z-score
X_df_media = X_df.mean()
X_df_desvio = X_df.std()
X_df_norm = (X_df - X_df_media) / X_df_desvio

split_treino = round(len(X_df_norm) * split_treino_percentual)
X_df_treino = X_df_norm[:split_treino]
y_df_treino = y_df[:split_treino]

X_df_teste = X_df_norm[split_treino:]
y_df_teste = y_df[split_treino:]

# Hiperparametros
taxa_aprendizado = 0.01
epocas = 400
tamanho_batch = 1000

# criar camada de atributos de entrada 
atributos = ['cd_berco', 'cd_ano', 'cd_mes','cd_terminal','nacionalidadearmador', 'teu', 'pesocargabruta']
atributos_entrada = []
cd_berco = tf.feature_column.numeric_column("cd_berco")
atributos_entrada.append(cd_berco)
cd_ano = tf.feature_column.numeric_column("cd_ano")
atributos_entrada.append(cd_ano)
cd_mes = tf.feature_column.numeric_column("cd_mes")
atributos_entrada.append(cd_mes)
cd_terminal = tf.feature_column.numeric_column("cd_terminal")
atributos_entrada.append(cd_terminal)
nacionalidadearmador = tf.feature_column.numeric_column("nacionalidadearmador")
atributos_entrada.append(nacionalidadearmador)
teu = tf.feature_column.numeric_column("teu")
atributos_entrada.append(teu)
pesocargabruta = tf.feature_column.numeric_column("pesocargabruta")
atributos_entrada.append(pesocargabruta)

# Establish the model's topography.
my_model = create_model(taxa_aprendizado, atributos_entrada, atributos)

# Train the model on the normalized training set.
epochs, hist = train_model(my_model, X_df_treino, y_df_treino, 
                           epocas, tamanho_batch, split_validacao_percentual)


# Plot a graph of the metric vs. epochs.
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate against the test set.
print("\n Evaluate the new model against the test set:")
my_model.evaluate(x=X_df_teste, y=y_df_teste, batch_size=tamanho_batch)
