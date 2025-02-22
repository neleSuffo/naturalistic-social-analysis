import ssl
import logging
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from data.dataset import INPUT_WIDTH, INPUT_HEIGHT
from training.testing import test
from keras.metrics import AUC
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.preprocessing.image import ImageDataGenerator

accuracy_threshold = 98e-2

class StopByAccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') >= accuracy_threshold:
            print('Accuracy has reach = %2.2f%%' % (logs['accuracy'] * 100), 'training has been stopped.')
            self.model.stop_training = True

def get_generators(train_images, train_labels, batch_size=32):
    # Define the augmentation parameters
    # Apply the augmentation to the training data
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        horizontal_flip=True,
        shear_range=0.2,
        fill_mode='wrap',
        validation_split=0.2
    )

    # Apply the augmentation to the training data
    train_generator = train_datagen.flow(
        train_images,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        subset='training'
    )

    validation_generator = train_datagen.flow(
        train_images,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        subset='validation'
    )

    return train_generator, validation_generator


ssl._create_default_https_context = ssl._create_unverified_context
vgg16 = VGG16(weights='imagenet', input_shape=self.input_shape, classes=self.classes, include_top=False)
        
class GazeNoGazeVgg16:
    def __init__(self, input_width, input_height):
        self.input_shape = (input_width, input_height, 3)
        self.classes = 2
        self.model = None
        self.build_model()

    def build_model(self):
        base_model = VGG16(weights='imagenet', 
                          input_shape=self.input_shape, 
                          include_top=False)

        # Freeze VGG16 layers
        for layer in base_model.layers:
            layer.trainable = False

        x = Flatten()(base_model.output)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)   
  
        self.model = Model(inputs=base_model.input, outputs=predictions)
        self.model.compile(optimizer='adam', 
                           loss='binary_crossentropy',
                           metrics=['accuracy', F1Score()])
        
        
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
        class_weight=class_weights,
        callbacks=[StopByAccuracyCallback()]
    )

    return model

MODEL_NAME = '/home/nele_pauline_suffo/models/GazeNoGazeVgg16.h5'

def train_vgg16(train_dir: str, val_dir: str) -> Model:
    """
    Train VGG16 model for gaze classification
    
    Args:
        train_dir: Path to training data directory
        val_dir: Path to validation data directory
    """
    vgg16 = GazeNoGazeVgg16(INPUT_WIDTH, INPUT_HEIGHT)
    model = vgg16.model
    model.summary()

    train_generator, validation_generator = get_generators(train_dir, val_dir)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
            
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[StopByAccuracyCallback()]
    )

    logging.info(f"Saving model to {MODEL_NAME}")
    model.save(MODEL_NAME)
    
    return model, history


def restore_vgg16(test_model=False) -> Model:
    vgg16 = GazeNoGazeVgg16(INPUT_WIDTH, INPUT_HEIGHT)
    model = vgg16.model

    model.load_weights(MODEL_NAME)

    if test_model:
        test(model)

    return model

def main(train_dir: str, val_dir: str) -> None:
    """
    Train VGG16 model for gaze classification   
    
    Parameters:
    ----------
    train_dir : str
        Path to training data directory
    val_dir : str
        Path to validation data directory
    """
    # Train model
    model, history = train_vgg16(train_dir, val_dir)
    
    
if __name__ == "__main__":
    train_dir = "/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/train"
    val_dir = "/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/val"
    main(train_dir, val_dir)