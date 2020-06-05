#base
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#tensorflow
import tensorflow as tf


def criar_modelo(taxa_aprendizado):

    modelo_rl = tf.keras.models.Sequential()
    modelo_rl.add(tf.keras.layers.Dense(units=1,input_shape=(1,)))
    modelo_rl.compile(optimizer=tf.keras.optimizers.RMSprop(lr=taxa_aprendizado),
                      loss="mean_squared_error", metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return modelo_rl


def treinar_modelo(modelo_rl, df, feature, label, epochs, batch_size, validation_split):
    history = modelo_rl.fit(x=df[feature], y=df[label], batch_size=batch_size,
                            epochs=epochs, validation_split=validation_split, 
                            verbose=0, use_multiprocessing=True)

    trained_weight = modelo_rl.get_weights()[0]
    trained_bias = modelo_rl.get_weights()[1]

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


def plot_the_model(df, trained_weight, trained_bias, feature, label):
    plt.figure()
    plt.xlabel(feature)
    plt.ylabel(label)
    plt.suptitle('TRABALHO COM CONTAINERS - TEUs vs TEMPO DE OPERACAO')

    random_examples = df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    x0 = 0
    y0 = trained_bias
    x1 = 6000
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')

    plt.show()


def plot_the_loss_curve(epochs, rmse):
    plt.figure()
    plt.suptitle('TRABALHO COM CONTAINERS - TEUs vs TEMPO DE OPERACAO')
    plt.xlabel("Epoca")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()  


#trabalhando com 4 classes alvo
def seta_classe(df, coluna_ordem, coluna_classe):
    quartil_1 = df[coluna_ordem].quantile(q=0.25, interpolation='linear')
    quartil_2 = df[coluna_ordem].quantile(q=0.50, interpolation='linear')
    quartil_3 = df[coluna_ordem].quantile(q=0.75, interpolation='linear')
    quartil_4 = df[coluna_ordem].quantile(q=0.85, interpolation='linear') 

    for _, row in df.iterrows():
        if row[coluna_ordem] >= 1 and row[coluna_ordem] <= quartil_1:
            row[coluna_classe] = 1
        elif row[coluna_ordem] > quartil_1 and row[coluna_ordem] <= quartil_2:
            row[coluna_classe] = 2
        elif row[coluna_ordem] > quartil_2 and row[coluna_ordem] <= quartil_3:
            row[coluna_classe] = 3
        elif row[coluna_ordem] > quartil_3 and row[coluna_ordem] <= quartil_4:
            row[coluna_classe] = 4

# miscilêneas
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# percentual para split treino, teste e validação
split_treino_percentual = 0.8 
split_validacao_percentual = 0.2

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

# separando datasets por tipo de carga devido aos comportamentos diferentes nas atracações
df_cg = df_atraca.loc[df_atraca['cd_tipocarga'] == 1] #carga geral
df_cg.set_index('idatracacao')
split_treino = round(len(df_cg) * split_treino_percentual)
df_cg_treino = df_cg[:split_treino]
df_cg_teste = df_cg[split_treino:]

df_gs = df_atraca.loc[df_atraca['cd_tipocarga'] == 2] #granel sólido
df_gs.set_index('idatracacao')
split_treino = round(len(df_gs) * split_treino_percentual)
df_gs_treino = df_gs[:split_treino]
df_gs_teste = df_gs[split_treino:]

df_gl = df_atraca.loc[df_atraca['cd_tipocarga'] == 3] #granel líquido e gasoso
df_gl.set_index('idatracacao')
split_treino = round(len(df_gl) * split_treino_percentual)
df_gl_treino = df_gl[:split_treino]
df_gl_teste = df_gl[split_treino:]

df_cg_cs = df_cg.loc[df_cg['teu'] == 0]               #somente carga solta
df_cg_cs.set_index('idatracacao')
split_treino = round(len(df_cg_cs) * split_treino_percentual)
df_cg_cs_treino = df_cg_cs[:split_treino]
df_cg_cs_teste = df_cg_cs[split_treino:]

df_cg_cntr = df_cg.loc[df_cg['teu'] > 0]              #somente container
df_cg_cntr.set_index('idatracacao')

#eliminando outliers e criando 4 classes
df_cg_cntr['classe_toperacao'] = 5
seta_classe(df_cg_cntr, 'toperacao', 'classe_toperacao')
df_cg_cntr = df_cg_cntr.loc[df_cg_cntr['classe_toperacao'] != 5]

split_treino = round(len(df_cg_cntr) * split_treino_percentual)
df_cg_cntr_treino = df_cg_cntr[:split_treino]
df_cg_cntr_teste = df_cg_cntr[split_treino:]

# Hiperparametros
taxa_aprendizado = 0.001
epocas = 240
tamanho_batch = 50

# atributo e rotulo para predição
atributo = 'teu'     # TEUs
rotulo = 'toperacao' # Tempo de Operação

modelo_rl = None
modelo_rl = criar_modelo(taxa_aprendizado)
weight, bias, epochs, rmse = treinar_modelo(modelo_rl, df_cg_cntr_treino, 
                                            atributo, rotulo,
                                            epocas, tamanho_batch, 
                                            split_validacao_percentual)

print("\nPeso:  %.4f" % weight)
print('Inclinação/tendência: %.4f\n' % bias )

plot_the_model(df_cg_cntr_treino, weight, bias, atributo, rotulo)
plot_the_loss_curve(epochs, rmse)

modelo_rl.evaluate(df_cg_cntr_teste[atributo], df_cg_cntr_teste[rotulo], batch_size=tamanho_batch)

# separando 10 amostras para predição
df_cg_cntr_predicao = df_cg_cntr_teste[1:10]
predicao = modelo_rl.predict_on_batch(x=df_cg_cntr_predicao[atributo])

print("teu     toperacao    predicao")
print("-----------------------------")

i = 0
for _, row in df_cg_cntr_predicao.iterrows():
    print("%5.0f %8.0f %8.0f" % (row[atributo], row[rotulo], predicao[i][0]))
    i = i + 1    
  
