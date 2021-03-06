# %%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import imageio
import time

def make_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        53*53*256, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((53, 53, 256)))
    # New shape is (BATCH_SIZE, 53, 53, 256)

    model.add(tf.keras.layers.Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # New shape is (BATCH_SIZE, 53, 53, 128)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(
        64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # New shape is (BATCH_SIZE, 53, 53, 64)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(
        32, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # New shape is (BATCH_SIZE, 53, 53, 32)
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(
        2, 2), padding='same', use_bias=False, activation='tanh'))
    # New shape is (BATCH_SIZE, 106, 106, 1)

    return model


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def make_discriminator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[106, 106, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        64, (5, 5), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        128, (5, 5), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(
        256, (5, 5), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)


    for i in range(predictions.shape[0]):
        fig = plt.figure()
        # plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

        plt.savefig('gan_model/images/image_at_epoch_{:04d}_{:04d}.png'.format(epoch, i),  bbox_inches='tight',transparent=True, pad_inches=0)
    # plt.show()


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(data):
    checkpoint_dir = 'gan_model/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    seed = tf.random.normal(
        [num_examples_to_generate, noise_dim])
    
    try:
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("Successfully loaded previous checkpoint!")
    except:
        print("Could not load any checkpoints!")

    print("Starting training...")
    for epoch in range(EPOCHS):
        start = time.time()

        for image_batch in data:
            train_step(image_batch)

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start))

    # Generate after the final epoch
    generate_and_save_images(generator, EPOCHS, seed)


def load_data(folder):
    images = []
    for im_path in glob.glob(folder + "/*.png"):
        images.append(imageio.imread(im_path))
    return np.asarray(images)

if __name__ == "__main__":
    generator = make_generator()
    discriminator = make_discriminator()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    IMG_HEIGHT = 106
    IMG_WIDTH = 106
    BUFFER_SIZE = 10000
    BATCH_SIZE = 256
    EPOCHS = 5
    noise_dim = 100
    num_examples_to_generate = 16
    cross_entropy = tf.keras.losses.BinaryCrossentropy(
        from_logits=True)

    train_images = load_data("data_prepping2/midi_imgs2")
    train_images = train_images.reshape(train_images.shape[0], 106, 106, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    train(train_dataset)
