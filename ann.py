import keras.backend as K
import tensorflow as tf
from keras.layers import (
    LSTM,
    Concatenate,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    Lambda,
    Multiply,
    Subtract,
)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score


def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)


def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def auroc(y_true, y_pred):
    return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)


class SiameseNeuralNetwork:
    def __init__(self, x_train, x_validation, y_train, y_validation, embedded):
        # x_train = [train_q1_seq, train_q2_seq]
        self.x_train = x_train
        self.x_validation = x_validation
        self.y_train = y_train
        self.y_validation = y_validation
        self.model = None
        self.embedded = embedded

    def build(self):
        # Source: https://github.com/prabhnoor0212/Siamese-Network-Text-Similarity/blob/master/quora_siamese.ipynb
        input_1 = Input(shape=(self.x_train[0].shape[1],))
        input_2 = Input(shape=(self.x_train[1].shape[1],))

        common_embed = Embedding(
            name="synopsis_embedd",
            input_dim=self.embedded.get_word_index() + 1,
            output_dim=len(self.embedded.get_index()["def"]),
            weights=[self.embedded.get_matrix()],
            input_length=self.x_train[0].shape[1],
            trainable=False,
        )

        lstm_1 = common_embed(input_1)
        lstm_2 = common_embed(input_2)

        common_lstm = LSTM(64, return_sequences=True, activation="relu")
        vector_1 = common_lstm(lstm_1)
        vector_1 = Flatten()(vector_1)

        vector_2 = common_lstm(lstm_2)
        vector_2 = Flatten()(vector_2)

        x3 = Subtract()([vector_1, vector_2])
        x3 = Multiply()([x3, x3])

        x1_ = Multiply()([vector_1, vector_1])
        x2_ = Multiply()([vector_2, vector_2])
        x4 = Subtract()([x1_, x2_])

        # https://stackoverflow.com/a/51003359/10650182
        x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)(
            [vector_1, vector_2]
        )

        conc = Concatenate(axis=-1)([x5, x4, x3])

        x = Dense(100, activation="relu", name="conc_layer")(conc)
        x = Dropout(0.01)(x)
        out = Dense(1, activation="sigmoid", name="out")(x)

        self.model = Model([input_1, input_2], out)

    def compile(self):
        self.model.compile(
            loss="binary_crossentropy", metrics=["acc", auroc], optimizer=Adam(0.00001)
        )

    def summary(self):
        self.model.summary()

    def fit(self):
        self.model.fit(
            [self.x_train[0], self.x_train[1]],
            self.y_train.values.reshape(-1, 1),
            epochs=5,
            batch_size=64,
            validation_data=(
                [self.x_validation[0], self.x_validation[1]],
                self.y_validation.values.reshape(-1, 1),
            ),
        )

    def evaluate(self):
        return self.model.evaluate(
            [self.x_validation[0], self.x_validation[1]],
            self.y_validation.values.reshape(-1, 1),
        )
