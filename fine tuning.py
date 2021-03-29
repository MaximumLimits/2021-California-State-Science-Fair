import tensorflow as tf
import numpy as np
from science_fair_8th.images import load_images

X_train, y_train, X_test, y_test, X_val, y_val = load_images(shuffle_before=True)
input_shape = X_train.shape[1:]

disable_debug_info = False
if disable_debug_info:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def new_design_va80(augment_data=False, tune_hypers=False, learning_rate=3.4e-4, batch_size=26, return_results=False):

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

        tf.keras.layers.Conv2D(256, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(8, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        # tf.keras.layers.Conv2D(6, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        # In kernel_initializer, the 2 is removed so that the initialization returns to Xavier initialization instead of
        # Kaiming initialization. Xaiver initialization should in theory work better since Kaiming initialization
        # is meant for ReLU, but the final activation is softmax. However, in practice, it does not make a difference.
        tf.keras.layers.Conv2D(6, 3, 1, 'same', kernel_initializer=tf.keras.initializers.VarianceScaling(), activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Activation('softmax')
    ]

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model = tf.keras.models.Sequential(layers)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    augmentor = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    optimizer = tf.keras.optimizers.RMSprop()

    if not tune_hypers:
        for fit in range(2):
            # validation_accuracies are the result of training 7 models independently and averaging validation_accuracies

            # 48x48 / 64x64-
            # 48x48 images - 81.305933% accuracy
            # 64x64 images - 82.466024% accuracy
            # model.compile(tf.keras.optimizers.RMSprop(learning_rate=2.5e-4 * 10**-fit), loss=tf.keras.validation_losses.sparse_categorical_crossentropy, metrics=['accuracy'])

            # 100x100
            # 100x100 images - 85.449123% accuracy
            # model.compile(tf.keras.optimizers.RMSprop(learning_rate=6e-4 * 10**-fit), loss=tf.keras.validation_losses.sparse_categorical_crossentropy, metrics=['accuracy'])

            # 128x128
            # 128x128 images - 88.067615% accuracy
            # model.compile(tf.keras.optimizers.RMSprop(learning_rate=4.5e-4 * 10**-fit), loss=tf.keras.validation_losses.sparse_categorical_crossentropy, metrics=['accuracy'])
            # model.compile(tf.keras.optimizers.RMSprop(learning_rate=2.5e-4 * 10**-fit), loss=tf.keras.validation_losses.sparse_categorical_crossentropy, metrics=['accuracy'])
            # model.compile(tf.keras.optimizers.RMSprop(learning_rate=3e-4 * 5**-fit), loss=tf.keras.validation_losses.sparse_categorical_crossentropy, metrics=['accuracy'])
            # the good one is below
            # model.compile(tf.keras.optimizers.RMSprop(learning_rate=2.5e-4 * 5**-fit), loss=tf.keras.validation_losses.sparse_categorical_crossentropy, metrics=['accuracy'])
            optimizer.learning_rate = 1e-4 * 10**-fit

            model.compile(optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

            # 156x156
            # 156x156 images - 86.54293% accuracy
            # model.compile(tf.keras.optimizers.RMSprop(learning_rate=4.5e-4 * 10**-fit), loss=tf.keras.validation_losses.sparse_categorical_crossentropy, metrics=['accuracy'])

            if not augment_data:
                # history = model.fit(X_train, y_train, 26, epochs=25, validation_data=(X_val, y_val), verbose=2, callbacks=[callback])
                history = model.fit(X_train, y_train, 26, epochs=16, validation_data=(X_val, y_val), verbose=2, callbacks=[callback])

                print()
                train_loss.extend(history.history['loss'])
                val_loss.extend(history.history['val_loss'])
                train_acc.extend(history.history['accuracy'])
                val_acc.extend(history.history['val_accuracy'])

            else:
                highest_val_loss = np.inf
                count = 0
                patience = 3
                num_epochs = 18
                model.compile(optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
                for epoch in range(num_epochs):
                    print(f'Epoch: {epoch+1}/{num_epochs}')
                    data = augmentor.flow(X_train, y_train, batch_size=20)
                    history = model.fit_generator(data, validation_data=(X_val, y_val), epochs=1, verbose=2)
                    train_loss.extend(history.history['loss'])
                    val_loss.extend(history.history['val_loss'])
                    current_val_loss = history.history['val_loss'][-1]
                    train_acc.extend(history.history['accuracy'])
                    current_val_acc = history.history['val_accuracy'][-1]
                    val_acc.append(current_val_acc)
                    count += 1
                    if highest_val_loss - current_val_loss > 0:
                        count = 0
                        highest_val_loss = current_val_loss
                    if count == patience:
                        break

                    # test = input.txt('break?')
                    # if test == 'yes':
                    #     break

                print()

        test_results = model.evaluate(X_test, y_test)
        print(input_shape)

        if return_results:
            return test_results[1]

    else:
        model.compile(learning_rate=learning_rate, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size, epochs=12, validation_data=(X_val, y_val), verbose=0)
        train_acc = model.evaluate(X_train, y_train, verbose=0)[1]
        val_acc = model.evaluate(X_val, y_val, verbose=0)[1]

        return train_acc, val_acc

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    # fig.suptitle(f'{material}, {label}', fontsize=18)

    axs[0].plot(train_loss, label='Train loss')
    axs[0].plot(val_loss, label='Validation loss')
    axs[0].set_title('Train and Validation Losses')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='upper right')

    axs[1].plot(train_acc, label='Train accuracy')
    axs[1].plot(val_acc, label='Validation Accuracy')
    axs[1].set_title('Train and Validation Accuracies')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend(loc='lower right')

    plt.show()

    # plt.plot(train_loss, label='Train loss')
    # plt.plot(val_loss, label='Validation loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend(loc='upper right')
    # plt.show()


def random_search(num_iters):
    for _ in range(num_iters):
        lr = np.random.uniform(2e-4, 6e-4)

        train_acc, val_acc = new_design_va80(False, True, lr)

        print(f'LR: {lr:e} | Train Accuracy: {train_acc:.5f} | Val Accuracy: {val_acc:.5f}')
        print()


def check_mean_accuracy(num_models=4, augment_data=False):
    accuracies = []
    for _ in range(num_models):
        global X_train, y_train, X_test, y_test, X_val, y_val, input_shape
        X_train, y_train, X_test, y_test, X_val, y_val = load_images(shuffle_before=True)
        input_shape = X_train.shape[1:]
        accuracies.append(new_design_va80(augment_data=augment_data, return_results=True))

    print(np.mean(accuracies))


new_design_va80(augment_data=True)

