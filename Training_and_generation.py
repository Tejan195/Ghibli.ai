import tensorflow as tf
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import datetime

# -----------------------------
# 1. Define Model Architectures
# -----------------------------

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

def Generator():
    # A simple U-Net architecture
    inputs = layers.Input(shape=[256, 256, 3])
    
    # Encoder
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 128, 128, 64)
        downsample(128, 4),                        # (bs, 64, 64, 128)
        downsample(256, 4),                        # (bs, 32, 32, 256)
        downsample(512, 4),                        # (bs, 16, 16, 512)
    ]
    
    # Decoder
    up_stack = [
        upsample(512, 4, apply_dropout=True),      # (bs, 32, 32, 512)
        upsample(256, 4, apply_dropout=True),      # (bs, 64, 64, 256)
        upsample(128, 4),                          # (bs, 128, 128, 128)
        upsample(64, 4),                           # (bs, 256, 256, 64)
    ]
    
    x = inputs
    skips = []
    # Downsampling
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    
    # Upsampling
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])
    
    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4,
                                  strides=2,
                                  padding='same',
                                  kernel_initializer=initializer,
                                  activation='tanh')  # (bs, 256, 256, 3)
    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = layers.Input(shape=[256, 256, 3], name='input_image')
    
    x = layers.Conv2D(64, 4, strides=2, padding='same',
                      kernel_initializer=initializer)(inp)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(128, 4, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.Conv2D(256, 4, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    x = layers.ZeroPadding2D()(x)
    x = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    
    # PatchGAN
    x = layers.ZeroPadding2D()(x)
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)  # (bs, 30, 30, 1)
    
    return tf.keras.Model(inputs=inp, outputs=last)

# -----------------------------
# 2. Loss Functions and Optimizers
# -----------------------------
loss_obj = tf.keras.losses.MeanSquaredError()

def discriminator_loss(real, generated):
    # Use real labels as ones and fake labels as zeros.
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = (real_loss + generated_loss) * 0.5
    return total_disc_loss

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

# Cycle-consistency loss
def calc_cycle_loss(real_image, cycled_image, lambda_cycle=10):
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return lambda_cycle * loss

# Identity loss (optional)
def identity_loss(real_image, same_image, lambda_identity=5):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return lambda_identity * loss

# Optimizers
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# -----------------------------
# 3. Build Models for Both Directions
# -----------------------------
# G: X -> Y (e.g., Real -> Ghibli)
# F: Y -> X (e.g., Ghibli -> Real)

generator_g = Generator()  # Generates Ghibli style from real images
generator_f = Generator()  # Generates real style from Ghibli images

discriminator_x = Discriminator()  # Discriminates real images in domain X
discriminator_y = Discriminator()  # Discriminates images in domain Y

# -----------------------------
# 4. Prepare the Dataset
# -----------------------------
# We'll load images from our train folders. Ensure your train folders for each domain are set.
# Folder structure:
#  └── dataset/
#       ├── train_x/  (Real-world images)
#       └── train_y/  (Ghibli-style images)

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = (tf.cast(image, tf.float32) / 127.5) - 1  # Normalize to [-1,1]
    return image

def load_image_train(image_file):
    image = load_image(image_file)
    # Data augmentation
    image = tf.image.random_flip_left_right(image)
    return image

# Change these paths to your dataset locations
train_x_path = "dataset/train_x"  # e.g., real-world images folder
train_y_path = "dataset/train_y"  # e.g., Ghibli-style images folder

train_x = tf.data.Dataset.list_files(train_x_path + '/*.jpg')
train_y = tf.data.Dataset.list_files(train_y_path + '/*.jpg')

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 1

train_x = train_x.map(load_image_train, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)
train_y = train_y.map(load_image_train, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

# -----------------------------
# 5. Training Step
# -----------------------------
@tf.function
def train_step(real_x, real_y):
    # persistent=True is used to allow multiple calls to the gradient tape.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y, F translates Y -> X.
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # Identity mapping
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # Calculate losses
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

        total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate gradients
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # Apply gradients
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    return total_gen_g_loss, total_gen_f_loss, disc_x_loss, disc_y_loss

# -----------------------------
# 6. Training Loop
# -----------------------------
EPOCHS = 10  # Increase this number for better results

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    n = 0
    for real_x, real_y in tf.data.Dataset.zip((train_x, train_y)):
        g_loss, f_loss, d_x_loss, d_y_loss = train_step(real_x, real_y)
        if n % 100 == 0:
            print(f"Step {n}: GenG_loss: {g_loss.numpy():.4f}, GenF_loss: {f_loss.numpy():.4f}, "
                  f"DiscX_loss: {d_x_loss.numpy():.4f}, DiscY_loss: {d_y_loss.numpy():.4f}")
        n += 1

    # Optionally, save the models every few epochs.
    if (epoch + 1) % 5 == 0:
        generator_g.save(f"generator_g_epoch_{epoch+1}.h5")
        generator_f.save(f"generator_f_epoch_{epoch+1}.h5")
        print("Saved models at epoch", epoch+1)

print("Training complete!")
