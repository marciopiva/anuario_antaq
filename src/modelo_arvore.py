#bases
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import graphviz

#scikit learn
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report
from sklearn.utils import shuffle


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

#trabalhando com 10 faixas para atributos contínuos
def seta_faixa(df, coluna_ordem, coluna_faixa):
    quartil_1 = df[coluna_ordem].quantile(q=0.1, interpolation='linear')
    quartil_2 = df[coluna_ordem].quantile(q=0.2, interpolation='linear')
    quartil_3 = df[coluna_ordem].quantile(q=0.3, interpolation='linear')
    quartil_4 = df[coluna_ordem].quantile(q=0.4, interpolation='linear') 
    quartil_5 = df[coluna_ordem].quantile(q=0.5, interpolation='linear')
    quartil_6 = df[coluna_ordem].quantile(q=0.6, interpolation='linear')
    quartil_7 = df[coluna_ordem].quantile(q=0.7, interpolation='linear')
    quartil_8 = df[coluna_ordem].quantile(q=0.8, interpolation='linear') 
    quartil_9 = df[coluna_ordem].quantile(q=0.9, interpolation='linear') 

    for _, row in df.iterrows():
        if row[coluna_ordem] <= quartil_1:
            row[coluna_faixa] = 1
        elif row[coluna_ordem] > quartil_1 and row[coluna_ordem] <= quartil_2:
            row[coluna_faixa] = 2
        elif row[coluna_ordem] > quartil_2 and row[coluna_ordem] <= quartil_3:
            row[coluna_faixa] = 3
        elif row[coluna_ordem] > quartil_3 and row[coluna_ordem] <= quartil_4:
            row[coluna_faixa] = 4
        elif row[coluna_ordem] > quartil_4 and row[coluna_ordem] <= quartil_5:
            row[coluna_faixa] = 5
        elif row[coluna_ordem] > quartil_5 and row[coluna_ordem] <= quartil_6:
            row[coluna_faixa] = 6
        elif row[coluna_ordem] > quartil_6 and row[coluna_ordem] <= quartil_7:
            row[coluna_faixa] = 7
        elif row[coluna_ordem] > quartil_7 and row[coluna_ordem] <= quartil_8:
            row[coluna_faixa] = 8
        elif row[coluna_ordem] > quartil_8 and row[coluna_ordem] <= quartil_9:
            row[coluna_faixa] = 9
        elif row[coluna_ordem] > quartil_9:
            row[coluna_faixa] = 10


def analise_geral(df, rotulo, titulo):
    #alguns agrupamentos para visualização
    df_top_ano = df.groupby('cd_ano').agg({rotulo: [np.mean]})
    df_top_mes = df.groupby('cd_mes').agg({rotulo: [np.mean]})
    df_pes_ano = df.groupby('cd_ano').agg({'pesocargabruta': [np.sum]})
    df_pes_mes = df.groupby('cd_mes').agg({'pesocargabruta': [np.sum]})
    df_teu_ano = df.groupby('cd_ano').agg({'teu': [np.sum]})
    df_teu_mes = df.groupby('cd_mes').agg({'teu': [np.sum]})

    #graficos gerais
    fig, axes = plt.subplots(nrows=3, ncols=2)
    fig.suptitle(titulo + ' - Agrupamentos', fontsize=16)

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

    plt.show()

    #estatistica descritiva
    print('>>> antes')
    print(df[['toperacao', 'tesperaatracacao', 'testadia']].describe())

    #olhando os histogramas e boxplot
    fig, axes = plt.subplots(nrows=3, ncols=2)
    fig.suptitle(titulo + ' - Histogramas e BoxPlot >> antes', fontsize=16)

    df.hist(column='toperacao', ax=axes[0, 0])
    df.boxplot(column='toperacao', ax=axes[0, 1])

    df.hist(column='tesperaatracacao', ax=axes[1, 0])
    df.boxplot(column='tesperaatracacao', ax=axes[1, 1])

    df.hist(column='testadia', ax=axes[2, 0])
    df.boxplot(column='testadia', ax=axes[2, 1])
    plt.show()

    #eliminando outliers e criando 4 classes para trabalho com árvores de decisão - label: toperacao
    df['classe_toperacao'] = 5
    seta_classe(df, 'toperacao', 'classe_toperacao')
    df = df.loc[df['classe_toperacao'] != 5]

    #eliminando outliers e criando 4 classes para trabalho com árvores de decisão - label: tesperaatracacao
    df['classe_tesperaatracacao'] = 5
    seta_classe(df, 'tesperaatracacao', 'classe_tesperaatracacao')
    df = df.loc[df['classe_tesperaatracacao'] != 5]

    #eliminando outliers e criando 4 classes para trabalho com árvores de decisão - label: tesperaatracacao
    df['classe_testadia'] = 5
    seta_classe(df, 'testadia', 'classe_testadia')
    df = df.loc[df['classe_testadia'] != 5]

    print('>>> depois')
    print(df[['toperacao', 'tesperaatracacao', 'testadia']].describe())

    #olhando os histogramas e boxplot
    fig, axes = plt.subplots(nrows=3, ncols=2)
    fig.suptitle(titulo + ' - Histogramas e BoxPlot >> depois', fontsize=16)

    df.hist(column='toperacao', ax=axes[0, 0])
    df.boxplot(column='toperacao', ax=axes[0, 1])

    df.hist(column='tesperaatracacao', ax=axes[1, 0])
    df.boxplot(column='tesperaatracacao', ax=axes[1, 1])

    df.hist(column='testadia', ax=axes[2, 0])
    df.boxplot(column='testadia', ax=axes[2, 1])
    plt.show()

    return df


def classifica_arvore(df, rotulo):
    #arvore de decisão com scikit learn
    classes = ['1', '2', '3', '4']
    atributos = ['cd_berco', 'cd_ano', 'cd_mes','cd_terminal','nacionalidadearmador', 'teu', 'pesocargabruta']
    
    #df_shuffle = shuffle(df, random_state=1)
    df_shuffle = df.reindex(np.random.permutation(df.index))
    X = df_shuffle[atributos]
    y = df_shuffle[rotulo]                                                                                         # classe resultado
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=1)                   # 80% para treina e 20% para teste

    #parâmetros básicos
    clf_arvore = DecisionTreeClassifier(criterion='entropy', max_depth=8, random_state=1)                         # entropia como métrica

    #fase de treinamento
    clf_arvore = clf_arvore.fit(X_treino, y_treino)
    y_predicao_treino = clf_arvore.predict(X_treino)
    print("Acuracidade no treino com DT:", accuracy_score(y_treino, y_predicao_treino))
    print("Precisão no treino com DT:", precision_score(y_treino, y_predicao_treino, average='weighted'))

    #matriz de confusão - para treino
    print(confusion_matrix(y_treino, y_predicao_treino))
    print(classification_report(y_treino, y_predicao_treino))

    #fase de testes
    y_predicao_teste = clf_arvore.predict(X_teste)

    #resultado nos testes
    print("Acuracidade no teste com DT:", accuracy_score(y_teste, y_predicao_teste))
    print("Precisão no teste com DT:", precision_score(y_teste, y_predicao_teste, average='weighted'))

    #matriz de confusão - para treino
    print(confusion_matrix(y_teste, y_predicao_teste))
    print(classification_report(y_teste, y_predicao_teste))

    clf_rf = RandomForestClassifier(n_estimators=20, random_state=1)
    clf_rf.fit(X_treino, y_treino)
    y_predicao_treino = clf_rf.predict(X_treino)
    print("Acuracidade no treino com RF:", accuracy_score(y_treino, y_predicao_treino))

    #fase de testes
    y_predicao = clf_rf.predict(X_teste)

    #resultado nos testes
    print("Acuracidade no teste com RF:", accuracy_score(y_teste, y_predicao))

    #visualizando a arvore
    dot_data = export_graphviz(clf_arvore, out_file=None, filled=True, rounded = True, feature_names=atributos, class_names=classes)
    graph = graphviz.Source(dot_data)
    graph.render('arvore', view=True)


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
#CONTAINERS
#
print('**********************************')
print('*** TRABALHANDO COM CONTAINERS ***')
print('**********************************')

#análise geral
df_cg_cntr = analise_geral(df_cg_cntr, 'toperacao', 'TRABALHANDO COM CONTAINERS')

#classificação com árvores
classifica_arvore(df_cg_cntr, 'classe_toperacao')

#
#CARGA SOLTA
#
print('***********************************')
print('*** TRABALHANDO COM CARGA SOLTA ***')
print('***********************************')

#análise geral
df_cg_cs = analise_geral(df_cg_cs, 'toperacao', 'TRABALHANDO COM CARGA SOLTA')

#classificação com árvores
classifica_arvore(df_cg_cs, 'classe_toperacao')


#
#GRANEL SÓLIDO
#
print('*************************************')
print('*** TRABALHANDO COM GRANEL SÓLIDO ***')
print('*************************************')

#análise geral
df_gs = analise_geral(df_gs, 'toperacao', 'TRABALHANDO COM GRANEL SÓLIDO')

#classificação com árvores
classifica_arvore(df_gs, 'classe_toperacao')


#
#GRANEL LÍQUIDO E GASOSO
#
print('***********************************************')
print('*** TRABALHANDO COM GRANEL LÍQUIDO E GASOSO ***')
print('***********************************************')

#análise geral
df_gl = analise_geral(df_gl, 'toperacao')

#classificação com árvores
classifica_arvore(df_gl, 'classe_toperacao')
