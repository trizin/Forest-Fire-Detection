# %%
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
# Get the dataset from https://data.mendeley.com/datasets/gjmr63rz2r/1
# %%
ds_train = tf.keras.preprocessing.image_dataset_from_directory('./Dataset/train', shuffle=True, validation_split=0.2, subset="training", seed=42, image_size=(160, 160), batch_size=32)
ds_val = tf.keras.preprocessing.image_dataset_from_directory('./Dataset/train', shuffle=True, validation_split=0.2, subset="validation", seed=42, image_size=(160, 160), batch_size=32)
ds_test = tf.keras.preprocessing.image_dataset_from_directory('./Dataset/test', image_size=(160, 160), batch_size=32)

# %%
data_augmentation = tf.keras.Sequential([  # for data augmentation
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
base_model.trainable = False
# %% Construct Model
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, output)
# %% Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
# %% Train model
model.fit(ds_train, validation_data=ds_val, epochs=20, callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
# %% Evaluate Model
model.evaluate(ds_test)
# %% Save model
model.save('model.h5')
# %%
