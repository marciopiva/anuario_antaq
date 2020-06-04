# base
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# scikit learn
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


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
df_base = df_cg_cntr

# eliminando outliers e criando 4 classes 
df_base['classe_toperacao'] = 5
seta_classe(df_base, 'toperacao', 'classe_toperacao')
df_base = df_base.loc[df_base['classe_toperacao'] != 5]

# atributos e rotulo para treino
atributos = ['cd_berco', 'cd_ano', 'cd_mes','cd_terminal', 'teu', 'pesocargabruta']
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
taxa_aprendizado = 0.005
epocas = 400
tamanho_batch = 1000

clf = MLPClassifier(hidden_layer_sizes=(256, 128, 64), batch_size=tamanho_batch,
                    learning_rate_init=taxa_aprendizado, max_iter=epocas, verbose=True, 
                    shuffle=True, n_iter_no_change=100)

clf.fit(X_df_treino, y_df_treino)

# rever o treino
predictions = clf.predict(X_df_treino)

print(confusion_matrix(y_df_treino, predictions))
print(classification_report(y_df_treino, predictions))

# fazer o teste
predictions = clf.predict(X_df_teste)

print(confusion_matrix(y_df_teste, predictions))
print(classification_report(y_df_teste, predictions))
