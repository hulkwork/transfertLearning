from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Reshape
from keras.models import Model


def resnet50_feature_extractor(input_shape=(224, 224, 3), weights='imagenet'):
    model = ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape)
    # output shape is 2048
    output = Reshape((2048,))(model.output)
    return Model(model.input, output)

def inception_v3_feature_extractor(input_shape=(224, 224, 3),weights = 'imagenet'):
    model = InceptionV3(
        weights=weights,
        include_top=False,
        input_shape=input_shape)

    output = Reshape((2048,))(model.output)
    return Model(model.input, output)

