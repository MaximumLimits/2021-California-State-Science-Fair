import tensorflow as tf
import numpy as np

from science_fair_8th_9th.images import load_images
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
X_train, y_train, X_test, y_test, X_val, y_val = load_images(shuffle_before=True, num_augments=4)  # 4


input_shape = X_train.shape[1:]

disable_debug_info = False
if disable_debug_info:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def create_and_train_TARnet(return_results=False, save_history=False, save_file='TARnet.hdf5'):

    initializer = tf.keras.initializers.VarianceScaling(2)

    layers = [
        tf.keras.layers.Conv2D(64, 3, 1, 'same', input_shape=input_shape, kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        # tf.keras.layers.MaxPooling2D(),
        # tf.keras.layers.BatchNormalization(),
        #
        # tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        # tf.keras.layers.MaxPooling2D(),
        # tf.keras.layers.BatchNormalization(),
        #
        # tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        # tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(6, 3, 1, 'same', kernel_initializer=initializer),  # revised
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Activation('softmax')
    ]

    model = tf.keras.models.Sequential(layers)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)  # val_loss
    save_best_model = tf.keras.callbacks.ModelCheckpoint(save_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)  # val_loss

    first_learning_rate = 2.26e-4
    second_learning_rate = 3.35e-5

    optimizer = tf.keras.optimizers.RMSprop()
    model.compile(optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    epoch_zero_train_loss, epoch_zero_train_acc = model.evaluate(X_train, y_train, verbose=0)
    epoch_zero_val_loss, epoch_zero_val_acc = model.evaluate(X_val, y_val, verbose=0)
    train_loss.append(epoch_zero_train_loss)
    val_loss.append(epoch_zero_val_loss)
    train_acc.append(epoch_zero_train_acc)
    val_acc.append(epoch_zero_val_acc)
    print('Before training')
    print(f'{X_train.shape[0]}/{X_train.shape[0]} - ?s - loss: {epoch_zero_train_loss:.4f} - accuracy: {epoch_zero_train_acc:.4f} - val_loss: {epoch_zero_val_loss:.4f} - val_accuracy: {epoch_zero_val_acc:.4f}')

    print(f'\nFirst Training Session | lr: {first_learning_rate:.2e}')
    optimizer.learning_rate = first_learning_rate
    history = model.fit(X_train, y_train, 26, epochs=16, validation_data=(X_val, y_val), verbose=2, callbacks=[early_stopping, save_best_model])
    train_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])
    train_acc.extend(history.history['accuracy'])
    val_acc.extend(history.history['val_accuracy'])

    print(f'\nSecond Training Session | lr: {second_learning_rate:.2e}')
    model.load_weights(save_file)
    optimizer.learning_rate = second_learning_rate
    history = model.fit(X_train, y_train, 26, epochs=14, validation_data=(X_val, y_val), verbose=2, callbacks=[early_stopping, save_best_model])
    train_loss.extend(history.history['loss'])
    val_loss.extend(history.history['val_loss'])
    train_acc.extend(history.history['accuracy'])
    val_acc.extend(history.history['val_accuracy'])

    model.load_weights(save_file)
    test_results = model.evaluate(X_test, y_test)

    if save_history:
        train_loss = np.asarray(train_loss)
        train_acc = np.asarray(train_acc)
        val_loss = np.asarray(val_loss)
        val_acc = np.asarray(val_acc)
        np.save(f'training_losses/TARnet.npy', train_loss)
        np.save(f'training_accuracies/TARnet.npy', train_acc)
        np.save(f'validation_losses/TARnet.npy', val_loss)
        np.save(f'validation_accuracies/TARnet.npy', val_acc)

    if return_results:
        return test_results[1]

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, constrained_layout=True)

    axs[0].plot(train_loss, label='Train loss')
    axs[0].plot(val_loss, label='Validation loss')
    axs[0].set_title('Train and Validation Losses of TARnet', fontsize=18)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='upper right')
    axs[0].set_xlim(0, len(train_acc))

    axs[1].plot(train_acc, label='Train accuracy')
    axs[1].plot(val_acc, label='Validation Accuracy')
    axs[1].set_title('Train and Validation Accuracies of TARnet', fontsize=18)
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].hlines(0.85, 0, len(train_acc), label='85% Accuracy')
    axs[1].legend(loc='lower right')

    plt.xlim(0, len(train_acc))

    plt.show()


create_and_train_TARnet()
