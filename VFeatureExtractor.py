import tensorflow as tf
from keras.applications.resnet import ResNet50
from keras import backend as K

class VFeatureExtractor:
    def __init__(self, height):
        super().__init__()
        self.height = height
        self.model = ResNet50(include_top=False, input_shape=(height, height, 3))
        self.featureSize = 4096

    def __call__(self, x):
        return self.getFeature2(x)

    def getFeature1(self, x):
        func = K.function(self.model.layers[0].input, self.model.layers[174].output)
        x = func(x)
        x = tf.nn.pool(x, [7, 7], 'AVG')
        x = tf.squeeze(x, [1, 2])
        return x
    
    def getFeature2(self, x):
        func = K.function(self.model.layers[0].input, [self.model.layers[174].output, self.model.layers[164].output])
        y = func(x)
        x = tf.concat([tf.nn.pool(y[0], [7, 7], 'AVG'), tf.nn.pool(y[1], [7, 7], 'AVG')], -1)
        x = tf.squeeze(x, [1, 2])
        return x