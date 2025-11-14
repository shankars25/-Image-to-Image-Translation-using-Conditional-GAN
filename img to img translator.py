# -*- coding: utf-8 -*-
"""
Pix2Pix Image-to-Image Translation
Pure Python Version (NO Colab code)
Compatible with TensorFlow 2.15+ / 2.19+ (2025)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import tarfile
import time


print("TensorFlow version:", tf.__version__)

# =====================================================
# STEP 1 — DOWNLOAD & EXTRACT DATASET (FACADES)
# =====================================================
url = "http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz"
dataset_path = tf.keras.utils.get_file("facades.tar.gz", origin=url)

extract_path = os.path.join(os.path.dirname(dataset_path), "facades")

# Extract once
if not os.path.exists(extract_path):
    with tarfile.open(dataset_path, "r:gz") as tar:
        tar.extractall(os.path.dirname(dataset_path))
print("Dataset extracted at:", extract_path)

print("Train samples:", len(os.listdir(os.path.join(extract_path, "train"))))
print("Test samples:", len(os.listdir(os.path.join(extract_path, "test"))))


# =====================================================
# STEP 2 — DATA LOADING & PREPROCESSING
# =====================================================
def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    # Split into Input (B) and Target (A)
    w = tf.shape(image)[1] // 2
    real = image[:, :w, :]
    inp = image[:, w:, :]

    inp = tf.image.resize(inp, [256, 256])
    real = tf.image.resize(real, [256, 256])

    inp = (tf.cast(inp, tf.float32) / 127.5) - 1
    real = (tf.cast(real, tf.float32) / 127.5) - 1

    return inp, real


train_files = tf.data.Dataset.list_files(extract_path + "/train/*.jpg", shuffle=True)
test_files = tf.data.Dataset.list_files(extract_path + "/test/*.jpg", shuffle=True)

train_ds = train_files.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(1)
test_ds = test_files.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(1)


# =====================================================
# STEP 3 — GENERATOR (U-NET)
# =====================================================
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    block = keras.Sequential()

    block.add(layers.Conv2D(filters, size, strides=2, padding='same',
                            kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        block.add(layers.BatchNormalization())

    block.add(layers.LeakyReLU())
    return block


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    block = keras.Sequential()

    block.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                     kernel_initializer=initializer, use_bias=False))
    block.add(layers.BatchNormalization())

    if apply_dropout:
        block.add(layers.Dropout(0.5))

    block.add(layers.ReLU())
    return block


def Generator():
    inputs = layers.Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, activation='tanh')

    x = inputs
    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    output = last(x)
    return keras.Model(inputs=inputs, outputs=output)


generator = Generator()


# =====================================================
# STEP 4 — DISCRIMINATOR (PATCHGAN)
# =====================================================
def Discriminator():
    inp = layers.Input(shape=[256, 256, 3], name="input_image")
    tar = layers.Input(shape=[256, 256, 3], name="target_image")

    x = layers.concatenate([inp, tar])
    initializer = tf.random_normal_initializer(0., 0.02)

    down1 = downsample(64, 4, False)(x)
    down2 = downsample(128, 4)(down1)
    down3 = downsample(256, 4)(down2)

    zero_pad1 = layers.ZeroPadding2D()(down3)
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    bn = layers.BatchNormalization()(conv)
    relu = layers.LeakyReLU()(bn)
    zero_pad2 = layers.ZeroPadding2D()(relu)

    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)

    return keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()


# =====================================================
# STEP 5 — LOSSES & OPTIMIZERS
# =====================================================
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_obj(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + (100 * l1_loss)


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = loss_obj(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + fake_loss


generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# =====================================================
# STEP 6 — TRAINING LOOP (SMALL DEMO)
# =====================================================
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real = discriminator([input_image, target], training=True)
        disc_generated = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated, gen_output, target)
        disc_loss = discriminator_loss(disc_real, disc_generated)

    generator_optimizer.apply_gradients(
        zip(gen_tape.gradient(gen_loss, generator.trainable_variables),
            generator.trainable_variables)
    )

    discriminator_optimizer.apply_gradients(
        zip(disc_tape.gradient(disc_loss, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )


EPOCHS = 1
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for input_image, target in train_ds.take(50):
        train_step(input_image, target)
    print("Epoch completed.")


# =====================================================
# STEP 7 — TEST & DISPLAY RESULTS
# =====================================================
def generate_images(model, test_input, target):
    prediction = model(test_input, training=False)

    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], target[0], prediction[0]]
    titles = ["Input", "Ground Truth", "Generated"]

    for i, img in enumerate(display_list):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.imshow((img * 0.5 + 0.5))
        plt.axis("off")

    plt.show()


for example_input, example_target in test_ds.take(1):
    generate_images(generator, example_input, example_target)

print("Pix2Pix Training Completed Successfully.")
