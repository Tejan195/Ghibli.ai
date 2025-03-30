import tensorflow as tf

def load_and_preprocess(image_path):
    # Ensure image_path is a string
    image_path = str(image_path)

    # Read the image file
    try:
        image = tf.io.read_file(image_path)
    except tf.errors.NotFoundError:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Decode based on file extension
    if image_path.lower().endswith('.png'):
        image = tf.image.decode_png(image, channels=3)
    else:
        image = tf.image.decode_jpeg(image, channels=3)

    # Resize (optional, remove if already resized by resize.py)
    # image = tf.image.resize(image, [256, 256])

    # Normalize to [-1, 1]
    image = (tf.cast(image, tf.float32) / 127.5) - 1

    # Optional: Add data augmentation for training
    # image = tf.image.random_flip_left_right(image)

    return image