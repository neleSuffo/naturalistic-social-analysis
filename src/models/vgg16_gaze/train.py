import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from focal_loss import BinaryFocalLoss

def create_model(img_width, img_height):
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers
    x = Flatten()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)

    # Compile the model
    loss_func = BinaryFocalLoss(gamma=2)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss=loss_func, metrics=['accuracy'])
    
    return model

def prepare_data_generators(train_dir, validation_dir, img_width, img_height, batch_size):
    # Define a custom function for resizing and center-cropping
    def resize_and_crop(image):
        image = tf.image.resize_with_crop_or_pad(image, 256, 256)  # Resize to 256x256
        image = tf.image.resize_with_crop_or_pad(image, img_width, img_height)  # Center-crop to target size
        return image
    
    # Create data generators with custom resizing and center-cropping
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        preprocessing_function=resize_and_crop
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=resize_and_crop
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )
    
    return train_generator, validation_generator

def train_model(model, train_generator, validation_generator, epochs):
    # Define early stopping callback
    from tensorflow.keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max', verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stopping]
    )
    
    return history

def save_model(model, model_path):
    # Save the model
    model.save(model_path)

def main():
    os.environ["OMP_NUM_THREADS"] = "4"  # OpenMP threads 
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    img_width, img_height = 224, 224
    batch_size = 16
    epochs = 20
    train_dir = '/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/train'
    validation_dir = '/home/nele_pauline_suffo/ProcessedData/yolo_gaze_input/val'
    model_path = '/home/nele_pauline_suffo/models/vgg16_gaze_classification_model.h5'

    model = create_model(img_width, img_height)
    train_generator, validation_generator = prepare_data_generators(train_dir, validation_dir, img_width, img_height, batch_size)
    history = train_model(model, train_generator, validation_generator, epochs)
    save_model(model, model_path)

if __name__ == "__main__":
    main()