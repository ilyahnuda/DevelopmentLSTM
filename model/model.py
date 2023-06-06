import tensorflow as tf

"""The model is dedicated to predict stock predictions """


class StockPrediction:
    def __init__(self, sequence=None, architecture=None):
        if not sequence:
            self.model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(1, input_shape=(100, 5), activation=tf.nn.leaky_relu),
                tf.keras.layers.Dense(5, activation='linear')
            ])
            self.architecture = (150,)
            metrics = [tf.keras.metrics.RootMeanSquaredError(), tf.keras.metrics.MeanAbsolutePercentageError()]
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse',
                               metrics=metrics)
        else:
            self.model = sequence
            self.architecture = architecture

    def train_model(self, X_train, Y_train, num_epoch=10, batch_size=16):
        hist = self.model.fit(X_train, Y_train, epochs=num_epoch, batch_size=batch_size, validation_split=0.1)

        return hist

    def test_model(self, X_test, Y_test):
        Y_pred = self.model.predict(X_test)

        MAPE = tf.keras.metrics.MeanAbsolutePercentageError()
        MAPE.update_state(Y_test, Y_pred)

        RMSE = tf.keras.metrics.RootMeanSquaredError()
        RMSE.update_state(Y_test, Y_pred)

        return Y_pred, {'RMSE': RMSE.result().numpy(), 'MAPE': MAPE.result().numpy()}

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, path):
        self.model.save(path)


def load_model(path):
    return tf.keras.models.load_model(path)
