import tensorflow as tf  # type: ignore


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
)
def random_crop_and_flip(
    x,
    y,
    image_size,
):
    """Apply random crop and flip."""
    x, y = tf.map_fn(
        lambda elems: random_crop(elems[0], elems[1], image_size),
        elems=(x, y),
        fn_output_signature=(tf.float32, tf.float32),
    )
    x, y = random_flip(x, y)
    return x, y


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
)
def random_crop_and_flip_swath_centered(
    x,
    y,
    image_size,
):
    """
    Apply random crop and flip.
    But crop is always centered around data swath
    """
    x, y = tf.map_fn(
        lambda elems: random_crop_swath_centered(
            elems[0], elems[1], image_size
        ),
        elems=(x, y),
        fn_output_signature=(tf.float32, tf.float32),
    )
    x, y = tf.map_fn(
        lambda elems: random_rotate_and_flip(elems[0], elems[1]),
        elems=(x, y),
        fn_output_signature=(tf.float32, tf.float32),
    )
    return x, y


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.float32),
    )
)
def random_flip(
    x,
    y,
):
    """Random flip of data."""

    def apply_horizontal_flip():
        return tf.reverse(x, [2]), tf.reverse(y, [2])

    def apply_vertical_flip():
        return tf.reverse(x, [1]), tf.reverse(y, [1])

    def no_flip():
        return x, y

    x, y = tf.cond(tf.random.uniform(()) > 0.5, apply_horizontal_flip, no_flip)
    x, y = tf.cond(tf.random.uniform(()) > 0.5, apply_vertical_flip, no_flip)
    return x, y


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
)
def random_crop(
    x,
    y,
    image_size,
):
    """Random crop of data."""
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)
    n1 = tf.cast(y_shape[0] / x_shape[0], tf.int32)
    n2 = tf.cast(y_shape[1] / x_shape[1], tf.int32)
    s1 = tf.random.uniform(
        (),
        minval=0,
        maxval=x_shape[0] - image_size,
        dtype=tf.dtypes.int32,
    )
    s2 = tf.random.uniform(
        (),
        minval=0,
        maxval=x_shape[1] - image_size,
        dtype=tf.dtypes.int32,
    )
    return (
        x[s1: s1 + image_size, s2: s2 + image_size, :],
        y[
            s1 * n1: (s1 + image_size) * n1,
            s2 * n2: (s2 + image_size) * n2,
            :,
        ],
    )


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
)
def random_crop_swath_centered(x, y, image_size):
    """Random crop of data, always centered around the x-axis."""

    x_shape = tf.shape(x)
    y_shape = tf.shape(y)
    n1 = tf.cast(y_shape[0] / x_shape[0], tf.int32)
    n2 = tf.cast(y_shape[1] / x_shape[1], tf.int32)
    center_x = x_shape[0] // 2

    s1 = tf.random.uniform(
        (),
        minval=0,
        maxval=x_shape[0] - image_size,
        dtype=tf.dtypes.int32,
    )
    s2 = center_x - image_size // 2
    s2 = tf.maximum(tf.minimum(s2, x_shape[1] - image_size), 0)
    return (
        x[s1: s1 + image_size, s2: s2 + image_size, :],
        y[
            s1 * n1: (s1 + image_size) * n1,
            s2 * n2: (s2 + image_size) * n2,
            :,
        ],
    )


@tf.function
def set_missing_data(
    x: tf.Tensor,
    missing_fraction: float,
    fill_value: float,
) -> tf.Tensor:
    """Set a fraction of the data to a given fill value."""
    return tf.where(
        tf.math.greater(
            tf.random.uniform(
                shape=(tf.shape(x)[0], tf.shape(x)[1]),
                minval=0,
                maxval=1,
            ),
            missing_fraction,
        ),
        x,
        fill_value,
    )


@tf.function(
    input_signature=(
        tf.TensorSpec(shape=[None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, 1], dtype=tf.float32),
    )
)
def random_rotate_and_flip(
    x,
    y,
):
    """Random rotation and flip of data."""

    def apply_transpose():
        return tf.transpose(x, perm=[1, 0, 2]), tf.transpose(y, perm=[1, 0, 2])

    def apply_horizontal_flip():
        return tf.reverse(x, [1]), tf.reverse(y, [1])

    def apply_vertical_flip():
        return tf.reverse(x, [0]), tf.reverse(y, [0])

    def no_change():
        return x, y

    x, y = tf.cond(tf.random.uniform(()) > 0.5, apply_transpose, no_change)
    x, y = tf.cond(
        tf.random.uniform(()) > 0.5, apply_horizontal_flip, no_change
    )
    x, y = tf.cond(tf.random.uniform(()) > 0.5, apply_vertical_flip, no_change)
    return x, y
