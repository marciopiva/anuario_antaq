#base
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

#tensorflow
import tensorflow as tf


def build_model(my_learning_rate):
    """Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1, 
                                    input_shape=(1,)))

    # Compile the model topography into code that TensorFlow can efficiently
    # execute. Configure training to minimize the model's mean squared error. 
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model        


def train_model(model, df, feature, label, epochs, batch_size, validation_split):
    """Train the model by feeding it data."""

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs. 
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we're going to take a snapshot
    # of the model's root mean squared error at each epoch. 
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


def plot_the_model(df, trained_weight, trained_bias, feature, label):
    """Plot the trained model against 200 random training examples."""

    # Label the axes.
    plt.xlabel(feature)
    plt.ylabel(label)

    # Create a scatter plot from 200 random points of the dataset.
    random_examples = df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = 10000
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')

    # Render the scatter plot and the red line.
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()  

# The following lines adjust the granularity of reporting. 
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# percentual para split treino, teste e validação
split_treino_percentual = 0.8
split_validacao_percentual = 0.2

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
split_treino = round(len(df_cg_cntr) * split_treino_percentual)
df_cg_cntr_treino = df_cg_cntr[:split_treino]
df_cg_cntr_teste = df_cg_cntr[split_treino:]

# Hiperparametros
learning_rate = 0.005
epochs = 60
batch_size = 30

# Specify the feature and the label.
my_feature = 'teu'  # the total number of rooms on a specific city block.
my_label= 'toperacao' # the median value of a house on a specific city block.
# That is, you're going to create a model that predicts house value based 
# solely on total_rooms.  

# Discard any pre-existing version of the model.
my_model = None

# Invoke the functions.
my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, df_cg_cntr_treino, 
                                         my_feature, my_label,
                                         epochs, batch_size, 
                                         split_validacao_percentual)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias )

plot_the_model(df_cg_cntr_treino, weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

x_test = df_cg_cntr_teste[my_feature]
y_test = df_cg_cntr_teste[my_label]

my_model.evaluate(x_test, y_test, batch_size=batch_size)
  
