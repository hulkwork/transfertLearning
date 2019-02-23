from keras.datasets import cifar10
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.applications.resnet50 import preprocess_input as resnet_preprocessor

from sklearn.utils import shuffle as sk_shuffle
import scipy
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from transfer.model import resnet50_feature_extractor

K.tensorflow_backend._get_available_gpus()


def dense_model(hidden_units=1024, dropout_rate=0.5, n_classes=10):
    model = Sequential()
    model.add(Dropout(dropout_rate, input_shape=(2048,)))
    model.add(Dense(hidden_units, activation='relu', input_shape=(2048,)))
    model.add(Dense(n_classes, activation='softmax'))
    return model


def data_generator(
        X, y, batch_size, target_size=(224, 224, 3),
        preprocessor=resnet_preprocessor, shuffle=False):
    start = 0
    end = start + batch_size
    n = X.shape[0]
    if shuffle:
        X, y = sk_shuffle(X, y)
    while True:
        X_batch = X[start: end]
        y_batch = y[start: end]
        X_resized = np.array([scipy.misc.imresize(x, target_size) for x in X_batch])
        X_preprocessed = preprocessor(X_resized)

        start += batch_size
        end += batch_size
        if start >= n:
            start = 0
            end = batch_size
            if shuffle:
                X, y = sk_shuffle(X, y)
        yield (X_preprocessed, y_batch)


class TransferResnet50(object):
    def __init__(self):
        pass

    def base_model(self, input_shape=(224, 224, 3)):
        # Extract feature from ResNet50
        return resnet50_feature_extractor(input_shape=input_shape)

    def model_create(self):
        raise NotImplementedError('Please implemented your model')

    def transfer_model(self, input_shape=(224, 224, 3)):
        self._base_model = self.base_model(input_shape)
        self.model = self.model_create()
        if self.model.layers[0].input_shape != (None, 2048):
            raise Exception("Your first layer must be (None,2048) != %s" % str(self.model.layers[0].input_shape))

    def train(self, X, y, validation_prop=0.3, batch_size=64, epochs=2, verbose=1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_prop)
        self.transfer_model(input_shape=X_train[0].shape)
        bottleneck_train = self._base_model.predict(X_train, batch_size=batch_size, verbose=verbose)
        print('bottleneck_train done')
        bottleneck_val = self._base_model.predict(X_val, batch_size=batch_size, verbose=verbose)
        print("bottleneck_val Done")
        self.history = self.model.fit(bottleneck_train, y_train, batch_size=batch_size,
                                      validation_data=(bottleneck_val, y_val),
                                      epochs=epochs, verbose=verbose)

    def predict(self, X, batch_size=64):
        bottleneck = self._base_model.predict(X, batch_size=batch_size)
        return self.model.predict(bottleneck, batch_size=batch_size)

    def test_transfer_cifar10(self):
        # add a global spatial average pooling layer
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(y_train)
        X_train = X_train
        y_train = y_train
        X_test = X_test
        y_test = y_test
        y_train = onehot_encoder.transform(y_train)
        y_test = onehot_encoder.transform(y_test)

        input_shape = X_train[0].shape
        resnet = resnet50_feature_extractor(input_shape)

        batch_size = 40
        cnn_codes_train = resnet.predict_generator(
            data_generator(
                X_train, y_train, batch_size=batch_size, target_size=input_shape),
            X_train.shape[0] / batch_size, verbose=1)

        cnn_codes_val = resnet.predict_generator(
            data_generator(
                X_test, y_test, batch_size=batch_size, target_size=input_shape),
            X_test.shape[0] / batch_size, verbose=1)

        model = dense_model(dropout_rate=0.0)
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        batch_size = 32
        history = model.fit(
            cnn_codes_train, y_train,
            batch_size=batch_size,
            validation_data=(cnn_codes_val, y_test),
            epochs=2,
        )
        return resnet, model, history
