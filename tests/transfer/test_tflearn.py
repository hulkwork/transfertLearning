from unittest import TestCase
from keras.datasets import mnist
from transfer import tflearn as module


class TfLearn(TestCase):
    def setUp(self):
        self.data = mnist.load_data()

    def test_data_load(self):
        (X_train, y_train), (X_test, y_test) = self.data
        print(X_train.shape)

    def test_transfer(self):
        m = module.TransferResnet50()
        model = m.test_transfer_cifar10()

    def test_create_object(self):
        # add a global spatial average pooling layer
        (X_train, y_train), (X_test, y_test) = module.cifar10.load_data()
        onehot_encoder = module.OneHotEncoder(sparse=False)
        onehot_encoder.fit(y_train)
        X_train = X_train[:100]
        y_train = y_train[:100]
        X_test = X_test[:100]
        y_test = y_test[:100]
        y_train = onehot_encoder.transform(y_train)
        y_test = onehot_encoder.transform(y_test)

        input_shape = X_train[0].shape

        class TfTest(module.TransferResnet50):
            def __init__(self):
                super(TfTest, self).__init__()

            def model_create(self):
                model = module.dense_model(dropout_rate=0.01)
                model.compile(
                    optimizer=module.Adam(lr=1e-3),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
                return model

        actual = TfTest()
        actual.transfer_model(input_shape)
        actual.train(X_train, y_train, batch_size=32)
