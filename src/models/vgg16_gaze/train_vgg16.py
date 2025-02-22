import ssl
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from data.dataset import INPUT_WIDTH, INPUT_HEIGHT
from training.testing import test
from utils.callbacks import StopByAccuracyCallback
from utils.data_generalisation import get_generators

ssl._create_default_https_context = ssl._create_unverified_context
vgg16 = VGG16(weights='imagenet', input_shape=self.input_shape, classes=self.classes, include_top=False)

class GazeNoGazeVgg16:
    def __init__(self, input_width, input_height):
        self.input_shape = (input_width, input_height, 2)
        self.classes = 2
        self.model = None
        self.build_model()

    def build_model(self):
        vgg16 = VGG16(weights='imagenet', input_shape=self.input_shape, classes=self.classes, include_top=False)

        for layer in vgg16.layers:
            layer.trainable = False

        x = Flatten()(vgg16.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.classes, activation='softmax')(x)

        self.model = Model(inputs=vgg16.input, outputs=predictions)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['f1_score'])
        
def train_vgg16(train_images, train_labels) -> Model:
    vgg16 = GazeNoGazeVgg16(INPUT_WIDTH, INPUT_HEIGHT)
    model = vgg16.model
    model.summary()

    train_generator, validation_generator = get_generators(train_images, train_labels)

    model.fit(
        train_generator,
        steps_per_epoch=40,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=10,
        callbacks=[StopByAccuracyCallback()]
    )

    return model

MODEL_NAME = '/home/nele_pauline_suffo/models/GazeNoGazeVgg16.h5'

def train_vgg16(train_images, train_labels) -> Model:
    vgg16 = GazeNoGazeVgg16(INPUT_WIDTH, INPUT_HEIGHT)
    model = vgg16.model
    model.summary()

    train_generator, validation_generator = get_generators(train_images, train_labels)

    model.fit(
        train_generator,
        steps_per_epoch=40,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=10,
        callbacks=[StopByAccuracyCallback()]
    )
    print(f"Save weights into {MODEL_NAME}")
    model.save_weights(MODEL_NAME)

    test(model)

    return model


def restore_vgg16(test_model=False) -> Model:
    vgg16 = GazeNoGazeVgg16(INPUT_WIDTH, INPUT_HEIGHT)
    model = vgg16.model

    model.load_weights(MODEL_NAME)

    if test_model:
        test(model)

    return model