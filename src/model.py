import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#scikit learn
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.utils import shuffle

#classes para toperacao para o contexto de Santos
def seta_classe(df, coluna_ordem, coluna_classe):
    quartil_1 = df[coluna_ordem].quantile(q=0.25, interpolation='linear')
    quartil_2 = df[coluna_ordem].quantile(q=0.50, interpolation='linear')
    quartil_3 = df[coluna_ordem].quantile(q=0.75, interpolation='linear')
    quartil_4 = (quartil_3 - quartil_1) * 1.5 #base 1.5 * FIQ

    for _, row in df.iterrows():
        if row[coluna_ordem] >= 1 and row[coluna_ordem] <= quartil_1:
            row[coluna_classe] = 1
        elif row[coluna_ordem] > quartil_1 and row[coluna_ordem] <= quartil_2:
            row[coluna_classe] = 2
        elif row[coluna_ordem] > quartil_2 and row[coluna_ordem] <= quartil_3:
            row[coluna_classe] = 3
        elif row[coluna_ordem] > quartil_3 and row[coluna_ordem] <= quartil_4:
            row[coluna_classe] = 4

    
#carregando dataset completo - gerado pelas views do sqlite - vw_dataset_cg + vw_dataset_gs + vw_dataset_gl
df_atraca = pd.read_csv('../csv/atraca.csv', low_memory=False)
df_atraca.set_index('idatracacao')

#filtrando Santos / Movimentação de Carga / Longo Curso 
df_atraca = df_atraca.loc[df_atraca['cd_portoatracacao'] == 11]      #Santos
df_atraca = df_atraca.loc[df_atraca['cd_tipooperacao'] == 2]         #Movimentação da Carga
df_atraca = df_atraca.loc[df_atraca['cd_tiponavegacaoatracao'] == 3] #Longo Curso

#removendo tempos "inválidos"
df_atraca = df_atraca.loc[df_atraca['toperacao'] > 0]                #Remover tempos zerados
df_atraca = df_atraca.loc[df_atraca['testadia'] > 0]                 #Remover tempos zerados

#separando datasets por tipo de carga devido aos comportamentos diferentes nas atracações
df_cg = df_atraca.loc[df_atraca['cd_tipocarga'] == 1] #carga geral
df_gs = df_atraca.loc[df_atraca['cd_tipocarga'] == 2] #granel sólido
df_gl = df_atraca.loc[df_atraca['cd_tipocarga'] == 3] #granel líquido e gasoso
df_cg_cs = df_cg.loc[df_cg['teu'] == 0]               #somente carga solta
df_cg_cntr = df_cg.loc[df_cg['teu'] > 0]              #somente container


#
#primeiras análises com containers
#
df_top_ano = df_cg_cntr.groupby('cd_ano').agg({'toperacao': [np.mean]})
df_top_mes = df_cg_cntr.groupby('cd_mes').agg({'toperacao': [np.mean]})
df_pes_ano = df_cg_cntr.groupby('cd_ano').agg({'pesocargabruta': [np.sum]})
df_pes_mes = df_cg_cntr.groupby('cd_mes').agg({'pesocargabruta': [np.sum]})
df_teu_ano = df_cg_cntr.groupby('cd_ano').agg({'teu': [np.sum]})
df_teu_mes = df_cg_cntr.groupby('cd_mes').agg({'teu': [np.sum]})

fig, axes = plt.subplots(nrows=3, ncols=2)
axes[0,0].set_title('Tempo vs Ano')
axes[0,1].set_title('Tempo vs Mes')
axes[1,0].set_title('Peso vs Ano')
axes[1,1].set_title('Peso vs Mes')
axes[2,0].set_title('TEU vs Ano')
axes[2,1].set_title('TEU vs Mes')

df_top_ano.plot(kind='bar', ax=axes[0,0], legend=False, color='cyan')
df_top_mes.plot(kind='line', ax=axes[0,1], legend=False, color='blue')
df_pes_ano.plot(kind='bar', ax=axes[1,0], legend=False, color='red')
df_pes_mes.plot(kind='line', ax=axes[1,1], legend=False, color='green')
df_teu_ano.plot(kind='bar', ax=axes[2,0], legend=False, color='gray')
df_teu_mes.plot(kind='line', ax=axes[2,1], legend=False, color='black')
#plt.show()

fig, axes = plt.subplots(nrows=3, ncols=1)

#estatistica descritiva
print('>>> carga geral - antes')
print(df_cg_cntr[['toperacao', 'tesperaatracacao', 'testadia']].describe())

#olhando os histogramas
df_cg_cntr.hist(column='toperacao', ax=axes[0])
df_cg_cntr.hist(column='tesperaatracacao', ax=axes[1])
df_cg_cntr.hist(column='testadia', ax=axes[2])
#plt.show()

#eliminando outliers e criando 4 classes para trabalho com árvores de decisão - label: toperacao
df_cg_cntr['classe_toperacao'] = 5
seta_classe(df_cg_cntr, 'toperacao', 'classe_toperacao')
df_cg_cntr = df_cg_cntr.loc[df_cg_cntr['classe_toperacao'] != 5]

#eliminando outliers e criando 4 classes para trabalho com árvores de decisão - label: tesperaatracacao
df_cg_cntr['classe_tesperaatracacao'] = 5
seta_classe(df_cg_cntr, 'tesperaatracacao', 'classe_tesperaatracacao')
df_cg_cntr = df_cg_cntr.loc[df_cg_cntr['classe_tesperaatracacao'] != 5]

#eliminando outliers e criando 4 classes para trabalho com árvores de decisão - label: tesperaatracacao
df_cg_cntr['classe_testadia'] = 5
seta_classe(df_cg_cntr, 'testadia', 'classe_testadia')
df_cg_cntr = df_cg_cntr.loc[df_cg_cntr['classe_testadia'] != 5]

print('>>> carga geral - depois')
print(df_cg_cntr[['toperacao', 'tesperaatracacao', 'testadia']].describe())

#revendo os histogramas
fig, axes = plt.subplots(nrows=3, ncols=1)

df_cg_cntr.hist(column='toperacao', ax=axes[0])
df_cg_cntr.hist(column='tesperaatracacao', ax=axes[1])
df_cg_cntr.hist(column='testadia', ax=axes[2])
#plt.show()

#arvore de decisão com scikit learn
df_cg_shuffle = shuffle(df_cg_cntr, random_state=1)
X = df_cg_shuffle[['cd_berco', 'cd_ano', 'cd_mes','cd_terminal','nacionalidadearmador', 'teu', 'pesocargabruta']] # atributos
y = df_cg_shuffle.classe_toperacao                                                                                # classe resultado
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.1, random_state=1)                      #80% para treina e 20% para teste

#parâmetros básicos
clf_arvore = DecisionTreeClassifier(criterion='entropy', max_depth=5)                                             #entropia como métrica, com profundidade 5

#fase de treinamento
clf = clf_arvore.fit(X_treino, y_treino)
y_predicao_treino = clf_arvore.predict(X_treino)
print("Acuracidade no treino com DT:", accuracy_score(y_treino, y_predicao_treino))

#fase de testes
y_predicao = clf_arvore.predict(X_teste)

#resultado nos testes
print("Acuracidade no teste com DT:", accuracy_score(y_teste, y_predicao))

clf_rf = RandomForestClassifier(n_estimators=50, random_state=1)
clf_rf.fit(X_treino, y_treino)
y_predicao = clf_rf.predict(X_teste)

#resultado nos testes
print("Acuracidade no teste com RF:", accuracy_score(y_teste, y_predicao))


#
#primeiras análises com granel sólido
#
#print('df_gs.describe')
#print(df_gs[['toperacao', 'tesperaatracacao', 'testadia']].describe())



#
#primeiras análises com granel líquido
#
#print('df_gl.describe')
#print(df_gl[['toperacao', 'tesperaatracacao', 'testadia']].describe())



