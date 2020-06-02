import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

def ret_classe_cg(registro):
    if registro['toperacao'] >= 1 and registro['toperacao'] <= 9:
        return 1
    elif registro['toperacao'] >= 10 and registro['toperacao'] <= 14:
        return 2
    elif registro['toperacao'] >= 15 and registro['toperacao'] <= 21:
        return 3
    elif registro['toperacao'] >= 22 and registro['toperacao'] <= 30:
        return 4
    else:
        return 5


df_atraca = pd.read_csv('../csv/atraca.csv', low_memory=False)
df_atraca.set_index('idatracacao')

df_atraca = df_atraca.loc[df_atraca['cd_portoatracacao'] == 11]      #Santos
df_atraca = df_atraca.loc[df_atraca['cd_tipooperacao'] == 2]         #Movimentação da Carga
df_atraca = df_atraca.loc[df_atraca['cd_tiponavegacaoatracao'] == 3] #Longo Curso
df_atraca = df_atraca.loc[df_atraca['toperacao'] > 0]                #Remover tempos zerados
df_atraca = df_atraca.loc[df_atraca['testadia'] > 0]                 #Remover tempos zerados

df_cg = df_atraca.loc[df_atraca['cd_tipocarga'] == 1] #carga geral
df_gs = df_atraca.loc[df_atraca['cd_tipocarga'] == 2] #granel sólido
df_gl = df_atraca.loc[df_atraca['cd_tipocarga'] == 3] #granel líquido e gasoso
df_cg_cs = df_cg.loc[df_cg['teu'] == 0]               #somente carga solta
df_cg_cntr = df_cg.loc[df_cg['teu'] > 0]              #somente container

fig, axes = plt.subplots(nrows=3, ncols=1)

print('df_cg.describe')
print(df_cg[['toperacao', 'tesperaatracacao', 'testadia']].describe())
df_cg.hist(column='toperacao', ax=axes[0])
df_cg.hist(column='tesperaatracacao', ax=axes[1])
df_cg.hist(column='testadia', ax=axes[2])
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=1)
df_cg['classe'] = df_cg.apply(ret_classe_cg, axis=1)
df_cg = df_cg.loc[df_cg['classe'] != 5]
print(df_cg[['toperacao', 'tesperaatracacao', 'testadia']].describe())
df_cg.hist(column='toperacao', ax=axes[0])
df_cg.hist(column='tesperaatracacao', ax=axes[1])
df_cg.hist(column='testadia', ax=axes[2])
plt.show()

print('df_gs.describe')
print(df_gs[['toperacao', 'tesperaatracacao', 'testadia']].describe())


print('df_gl.describe')
print(df_gl[['toperacao', 'tesperaatracacao', 'testadia']].describe())

df_top_ano = df_cg_cntr.groupby('cd_ano').agg({'testadia': [np.mean]})
df_top_mes = df_cg_cntr.groupby('cd_mes').agg({'testadia': [np.mean]})
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

#df2.hist(column='toperacao', ax=axes[0,0])
#df2.boxplot(column='toperacao', ax=axes[0,1])
df_top_ano.plot(kind='bar', ax=axes[0,0], legend=False, color='cyan')
df_top_mes.plot(kind='line', ax=axes[0,1], legend=False, color='blue')
df_pes_ano.plot(kind='bar', ax=axes[1,0], legend=False, color='red')
df_pes_mes.plot(kind='line', ax=axes[1,1], legend=False, color='green')
df_teu_ano.plot(kind='bar', ax=axes[2,0], legend=False, color='gray')
df_teu_mes.plot(kind='line', ax=axes[2,1], legend=False, color='black')
plt.show()

