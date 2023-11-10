import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from data import tf_dataset, load_data
from model import build_model

# def iou(y_true, y_pred):
#     def f(y_true, y_pred):
#         intersection = (y_true * y_pred).sum()
#         union = y_true.sum() + y_pred.sum() - intersection
#         x = (intersection + 1e-15) / (union + 1e-15)
#         x = x.astype(tf.float32)
#         return x
#     return tf.numpy_function(f, [y_true, y_pred], tf.float32)
def accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        union = tf.reduce_sum(tf.cast(y_true + y_pred - y_true * y_pred, tf.float32))
        x = (intersection + 1e-15) / (union + 1e-15)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)
# Register the iou function for serialization
tf.keras.utils.get_custom_objects()["iou"] = iou

def train_unet(data_dir, batch_size, lr, epochs):
    # Load and preprocess data
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(data_dir)
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    # Create the U-Net model
    model = build_model()

    # Compile the model
    opt = tf.keras.optimizers.Adam(lr)
    metrics = ["accuracy", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), iou]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)

    # Define callbacks
    callbacks = [
        ModelCheckpoint("files/model.keras"),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
        CSVLogger("files/data.csv"),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    # Calculate steps per epoch
    train_steps = len(train_x) // batch_size
    valid_steps = len(valid_x) // batch_size

    if len(train_x) % batch_size != 0:
        train_steps += 1
    if len(valid_x) % batch_size != 0:
        valid_steps += 1

    # Train the model
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_steps=valid_steps,
        callbacks=callbacks
    )

if __name__ == "__main__":
    # Define hyperparameters
    data_directory = "CVC-612/"
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 20

    # Start training
    train_unet(data_directory, batch_size, learning_rate, num_epochs)
