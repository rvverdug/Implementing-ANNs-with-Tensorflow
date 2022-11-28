import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
from twin_networks import Subtask, TwinModelMnist
from utils import create_summary_writers, write_metrics
from labels import get_label_function


def preprocess_mnist(data):
    """Preprocesses the MNIST data."""
    data = data.map(lambda x, t: (tf.cast(x, float), t))
    data = data.map(lambda x, t: (tf.reshape(x, (-1,)), t))
    return data.map(lambda x, t: ((x / 128.0) - 1.0, t))


def create_new_dataset(data, batch_size, subtask: Subtask):
    """Creates a new dataset w/ labels depending on the subtask"""
    data = preprocess_mnist(data)
    zipped_ds = tf.data.Dataset.zip((data.shuffle(2000), data.shuffle(2000)))
    label_function = get_label_function(subtask=subtask)
    zipped_ds = zipped_ds.map(
        lambda x1, x2: (x1[0], x2[0], label_function(x1[1], x2[1]))
    )
    zipped_ds = zipped_ds.map(lambda x1, x2, t: (x1, x2, tf.cast(t, tf.int32)))
    if subtask == Subtask.DIFFERENCE:
        zipped_ds = zipped_ds.map(lambda x1, x2, z: (x1, x2, tf.one_hot(z, 19)))

    zipped_ds.cache()
    zipped_ds = zipped_ds.shuffle(2000)
    zipped_ds = zipped_ds.batch(batch_size)
    zipped_ds = zipped_ds.prefetch(tf.data.AUTOTUNE)
    return zipped_ds


def training_loop(
    model, train_ds, val_ds, epochs, train_summary_writer, val_summary_writer, save_path
):

    for step in range(epochs):
        for data in tqdm.tqdm(train_ds, position=0, leave=True):
            metrics = model.train_step(data)
        write_metrics(model, metrics, train_summary_writer, step)
        model.reset_metrics()

        for data in tqdm.tqdm(val_ds, position=0, leave=True):
            metrics = model.test_step(data)
        write_metrics(model, metrics, val_summary_writer, step, validation=True)
        model.reset_metrics()

    if save_path:
        model.save_weights(save_path)


def create_model_from_saved_weights(save_path, subtask, train_ds):
    fresh_model = TwinModelMnist(subtask=subtask)
    for img1, img2, label in train_ds:
        fresh_model((img1, img2))
        break
    return fresh_model.load_weights(save_path)


def run_stuff_once(
    subtask=Subtask.DIFFERENCE,
    optimizer=tf.keras.optimizers.Adam(),
    epochs=10,
    optimizer_name=None,
):
    """Loads the data, creates new datasets and trains/saves model."""
    mnist = tfds.load("mnist", split=["train", "test"], as_supervised=True)
    train_ds = mnist[0]
    val_ds = mnist[1]
    train_ds = create_new_dataset(train_ds, batch_size=32, subtask=subtask)
    val_ds = create_new_dataset(val_ds, batch_size=32, subtask=subtask)

    for img1, img2, label in train_ds:
        print(img1.shape, img2.shape, label.shape)
        break

    train_summary_writer, val_summary_writer = create_summary_writers(
        config_name=f"{subtask.name}_{optimizer_name}"
    )
    model = TwinModelMnist(subtask=subtask, optimizer=optimizer)
    save_path = f"trained_model_{subtask}"

    training_loop(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        train_summary_writer=train_summary_writer,
        val_summary_writer=val_summary_writer,
        save_path=save_path,
    )

    model = create_model_from_saved_weights(save_path, subtask, train_ds)
    return save_path


def do_experiments(learning_rate=0.001, epochs=10):
    optimizer_dict = {
        "adam_optimizer": tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "sgd_wo_momentum": tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.0
        ),
        "sgd_w_momentum": tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=0.8
        ),
        "rms_prop": tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        "ada_grad": tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
    }

    for optimizer_name, optimizer in optimizer_dict.items():
        run_stuff_once(
            subtask=Subtask.LARGER_FIVE,
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            epochs=epochs,
        )
        run_stuff_once(
            subtask=Subtask.DIFFERENCE,
            optimizer=optimizer,
            optimizer_name=optimizer_name,
            epochs=epochs,
        )


def main():
    save_path_larger_five = run_stuff_once(subtask=Subtask.LARGER_FIVE, epochs=2)
    save_path_a_minus_b = run_stuff_once(subtask=Subtask.DIFFERENCE, epochs=2)
    do_experiments(epochs=2)


if __name__ == "__main__":
    main()
