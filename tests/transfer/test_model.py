from transfer import model as module
from unittest import TestCase

# It's not to verify keras work, only see the dimension of the last layer shape
class TestModels(TestCase):
    def setUp(self):
        pass

    def test_resnet(self):
        resnet = module.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224,224,3))
        print(resnet.layers[-1].output_shape)
        print(resnet.summary())

    def test_inception_v3(self):
        inception_v3 = module.InceptionV3(include_top=False)
        print(inception_v3.summary())
        print(inception_v3.layers[-1].output_shape)

