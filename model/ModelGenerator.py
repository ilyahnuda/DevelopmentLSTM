from model import StockPrediction
import tensorflow as tf


# self.single_lstm = [10, 30, 50, 100, 150, 200]
# self.two_lstm = [10, 20, 50, 100, 100, 150, 150, 200]
# self.three_lstm = [50, 50, 100, 100, 150, 150, 200, 200]
# optimizers = [tf.keras.optimizers.Adagrad, tf.keras.optimizers.Adam]
# learning_rates = [0.01, 0.001]

class ModelGenerator:
    def __init__(self):
        self.window_size = 40
        self.labels = 5
        self.input_shape = (self.window_size, self.labels)
        self.batch_size = 8
        self.validation_split = 0.1
        self.max_lstm_depth = 3
        self.max_linear_depth = 4
        self.single_lstm = [100, 150, 200]
        self.two_lstm = [100, 150, 150]
        self.three_lstm = [150, 150, 200]
        self.all_variants = (self.single_lstm, self.two_lstm, self.three_lstm)

    def generate_models(self):
        for i in range(1,self.max_lstm_depth):
            layer = self.all_variants[i]
            for var in range(len(layer) - i):
                units = layer[var:var + i + 1]

                print("New model")
                for j in range(12):
                    model = tf.keras.Sequential()
                    architecture = []
                    len_units = len(units)
                    for unit in range(len_units):
                        architecture.append(units[unit])
                        if len_units < 2:
                            model.add(tf.keras.layers.LSTM(units[unit], input_shape=(self.window_size, self.labels),
                                                           activation=tf.nn.leaky_relu, kernel_regularizer='l2'))
                        else:
                            if unit == len_units - 1:
                                model.add(
                                    tf.keras.layers.LSTM(units[unit], kernel_regularizer='l2',
                                                         activation=tf.nn.leaky_relu))
                            elif unit == 0:
                                model.add(tf.keras.layers.LSTM(units[unit], input_shape=(self.window_size, self.labels),
                                                               activation=tf.nn.leaky_relu, kernel_regularizer='l2',
                                                               return_sequences=True))
                            else:
                                model.add(tf.keras.layers.LSTM(units[unit], activation=tf.nn.leaky_relu,
                                                               kernel_regularizer='l2', return_sequences=True))

                    model.add(tf.keras.layers.Dense(self.labels, activation='linear'))
                    metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()]
                    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse',
                                  metrics=metrics)
                    res_model = StockPrediction(sequence=model, architecture=architecture,
                                                optimizer=tf.keras.optimizers.Adagrad.__name__,
                                                learning_rate=0.001)
                    yield res_model


if __name__ == '__main__':
    ModelGenerator().generate_models()
    print(tf.keras.optimizers.Adagrad.__name__)
