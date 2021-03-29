import tensorflow as tf
import numpy as np
from images import load_images

X_train, y_train, X_test, y_test, X_val, y_val = load_images()
input_shape = X_train.shape[1:]

disable_debug_info = False
if disable_debug_info:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def alpha_test():
    initializer = tf.keras.initializers.VarianceScaling(2)
    # input_shape = (64*64*3,) ideal
    input_shape = (32 * 32 * 3,)
    layers = [
        tf.keras.layers.Dense(100, activation='relu', kernel_initializer=initializer, input_shape=input_shape,
                              kernel_regularizer=tf.keras.regularizers.l2(5.196882e-02)),
        tf.keras.layers.Dense(6, activation='softmax', kernel_initializer=initializer,
                              kernel_regularizer=tf.keras.regularizers.l2(5.196882e-02))
    ]

    model = tf.keras.Sequential(layers)

    from science_fair_8th_9th.images import past_load_images
    X_train, y_train, X_test, y_test, X_val, y_val = past_load_images()

    with tf.device('/gpu:0'):
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=7.134183e-02),
                      loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.sparse_categorical_accuracy])
        history = model.fit(X_train, y_train, 300, epochs=11)

        import matplotlib.pyplot as plt

        train_accuracy = history.history['sparse_categorical_accuracy']
        plt.plot(train_accuracy, label='Train accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

        eval = model.evaluate(X_test, y_test)
        print(eval)


def original_design():
    initializer = tf.keras.initializers.VarianceScaling(2)

    layers = [
        tf.keras.layers.Conv2D(64, 3, 1, 'same', input_shape=X_train.shape[1:], kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),  # 64x64

        tf.keras.layers.Conv2D(48, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(40, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),  # 32x32

        tf.keras.layers.Conv2D(32, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, kernel_initializer=initializer, activation='softmax')
    ]

    model = tf.keras.models.Sequential(layers)
    model.compile(tf.keras.optimizers.RMSprop(learning_rate=5e-4),
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    epoch_zero_train_loss, epoch_zero_train_acc = model.evaluate(X_train, y_train, verbose=0)
    epoch_zero_val_loss, epoch_zero_val_acc = model.evaluate(X_val, y_val, verbose=0)

    history = model.fit(X_train, y_train, 150, epochs=25, validation_data=(X_val, y_val), verbose=2)

    train_loss, train_acc, val_loss, val_acc = history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']
    train_loss.insert(0, epoch_zero_train_loss)
    train_acc.insert(0, epoch_zero_train_acc)
    val_loss.insert(0, epoch_zero_val_loss)
    val_acc.insert(0, epoch_zero_val_acc)

    # return history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']
    return train_loss, train_acc, val_loss, val_acc

    import matplotlib.pyplot as plt

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(train_accuracy, label='Train accuracy')
    plt.plot(val_accuracy, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def og_batch_before():
    initializer = tf.keras.initializers.VarianceScaling(2)

    layers = [
        tf.keras.layers.Conv2D(64, 3, 1, 'same', input_shape=X_train.shape[1:], kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.MaxPooling2D(),  # 64x64

        tf.keras.layers.Conv2D(48, 3, 1, 'same', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(40, 3, 1, 'same', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.MaxPooling2D(),  # 32x32

        tf.keras.layers.Conv2D(32, 3, 1, 'same', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(16, 3, 1, 'same', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, kernel_initializer=initializer, activation='softmax')
    ]

    # model = tf.keras.models.Sequential(layers)
    # model.compile(tf.keras.optimizers.RMSprop(learning_rate=5e-4),
    #               loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    #
    # history = model.fit(X_train, y_train, 150, epochs=12, validation_data=(X_val, y_val), verbose=2)
    #
    # return history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']

    model = tf.keras.models.Sequential(layers)
    model.compile(tf.keras.optimizers.RMSprop(learning_rate=5e-4),
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    epoch_zero_train_loss, epoch_zero_train_acc = model.evaluate(X_train, y_train, verbose=0)
    epoch_zero_val_loss, epoch_zero_val_acc = model.evaluate(X_val, y_val, verbose=0)

    history = model.fit(X_train, y_train, 150, epochs=12, validation_data=(X_val, y_val), verbose=2)

    train_loss, train_acc, val_loss, val_acc = history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']
    train_loss.insert(0, epoch_zero_train_loss)
    train_acc.insert(0, epoch_zero_train_acc)
    val_loss.insert(0, epoch_zero_val_loss)
    val_acc.insert(0, epoch_zero_val_acc)

    # return history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']
    return train_loss, train_acc, val_loss, val_acc

    import matplotlib.pyplot as plt

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(train_accuracy, label='Train accuracy')
    plt.plot(val_accuracy, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def more_layers():
    initializer = tf.keras.initializers.VarianceScaling(2)

    layers = [
        tf.keras.layers.Conv2D(64, 3, 1, 'same', input_shape=X_train.shape[1:], kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(48, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(48, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(48, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(48, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(32, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, kernel_initializer=initializer, activation='softmax')
    ]

    model = tf.keras.models.Sequential(layers)
    return model


def train_model(model):
    model.compile(tf.keras.optimizers.RMSprop(learning_rate=5e-3),
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    epoch_zero_train_loss, epoch_zero_train_acc = model.evaluate(X_train, y_train, verbose=0)
    epoch_zero_val_loss, epoch_zero_val_acc = model.evaluate(X_val, y_val, verbose=0)

    with tf.device('/gpu:0'):
        history = model.fit(X_train, y_train, 100, epochs=12, validation_data=(X_val, y_val), verbose=2)

    train_loss, train_acc, val_loss, val_acc = history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']
    train_loss.insert(0, epoch_zero_train_loss)
    train_acc.insert(0, epoch_zero_train_acc)
    val_loss.insert(0, epoch_zero_val_loss)
    val_acc.insert(0, epoch_zero_val_acc)

    # return history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']
    return train_loss, train_acc, val_loss, val_acc

    return history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']

    import matplotlib.pyplot as plt

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(train_accuracy, label='Train accuracy')
    plt.plot(val_accuracy, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def less_layers(reg):
    initializer = tf.keras.initializers.VarianceScaling(2)
    reg = tf.keras.regularizers.l2(reg)

    layers = [

        tf.keras.layers.Conv2D(32, 3, 1, 'same', kernel_initializer=initializer, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(16, 3, 1, 'same', kernel_initializer=initializer, activation='relu', kernel_regularizer=reg),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(6, kernel_initializer=initializer, activation='softmax', kernel_regularizer=reg)
    ]

    model = tf.keras.models.Sequential(layers)
    return model


def train_less_layers():

    # while True:
    # lr = 10 ** -np.random.uniform(2, 5)
    # reg = 10 ** np.random.uniform(-2, 2)
    # lr = np.random.uniform(3, 7) * 10 ** -3
    # reg = 10 ** np.random.uniform(-4, -1)
    lr = 5.793e-03
    reg = 1.112e-03

    model = less_layers(reg)

    model.compile(tf.keras.optimizers.RMSprop(learning_rate=lr),
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    epoch_zero_train_loss, epoch_zero_train_acc = model.evaluate(X_train, y_train, verbose=0)
    epoch_zero_val_loss, epoch_zero_val_acc = model.evaluate(X_val, y_val, verbose=0)

    with tf.device('/gpu:0'):
        # history = model.fit(X_train, y_train, 200, epochs=2, validation_data=(X_val, y_val), verbose=0)
        history = model.fit(X_train, y_train, 200, epochs=15, validation_data=(X_val, y_val), verbose=2)

    # print(f'LR: {lr:.3e} | Reg: {reg:.3e} | Accuracy: {history.history["val_accuracy"][-1]:.4f}')

    train_loss, train_acc, val_loss, val_acc = history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']
    train_loss.insert(0, epoch_zero_train_loss)
    train_acc.insert(0, epoch_zero_train_acc)
    val_loss.insert(0, epoch_zero_val_loss)
    val_acc.insert(0, epoch_zero_val_acc)

    # return history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']
    return train_loss, train_acc, val_loss, val_acc

    return history.history['loss'], history.history['accuracy'], history.history['val_loss'], history.history['val_accuracy']
    import matplotlib.pyplot as plt

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(train_accuracy, label='Train accuracy')
    plt.plot(val_accuracy, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def new_design_va75():
    # 1768/1768 - 22s - loss: 0.3665 - accuracy: 0.8756 - val_loss: 0.7373 - val_accuracy: 0.7500
    # 212x212 input
    initializer = tf.keras.initializers.VarianceScaling(2)

    layers = [
        tf.keras.layers.Conv2D(128, 3, 1, 'same', input_shape=X_train.shape[1:], kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(48, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(32, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(8, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(6, kernel_initializer=initializer, activation='softmax')
        # tf.keras.layers.Conv2D(6, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        # tf.keras.layers.GlobalAveragePooling2D(),
        # tf.keras.layers.Activation('softmax')
    ]

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=7)
    model = tf.keras.models.Sequential(layers)
    model.compile(tf.keras.optimizers.RMSprop(learning_rate=5e-4),
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    epoch_zero_train_loss, epoch_zero_train_acc = model.evaluate(X_train, y_train, verbose=0)
    epoch_zero_val_loss, epoch_zero_val_acc = model.evaluate(X_val, y_val, verbose=0)

    history = model.fit(X_train, y_train, 20, epochs=25, validation_data=(X_val, y_val), verbose=2, callbacks=[callback])

    model.compile(tf.keras.optimizers.RMSprop(learning_rate=1e-5),
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    new_history = model.fit(X_train, y_train, 20, epochs=25, validation_data=(X_val, y_val), verbose=2, callbacks=[callback])

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    train_loss.extend(new_history.history['loss'])
    val_loss.extend(new_history.history['val_loss'])
    train_accuracy.extend(new_history.history['accuracy'])
    val_accuracy.extend(new_history.history['val_accuracy'])

    train_loss.insert(0, epoch_zero_train_loss)
    train_accuracy.insert(0, epoch_zero_train_acc)
    val_loss.insert(0, epoch_zero_val_loss)
    val_accuracy.insert(0, epoch_zero_val_acc)

    return train_loss, train_accuracy, val_loss, val_accuracy


def new_design_va80(augment_data=False, tune_hypers=False, learning_rate=4.5e-4, batch_size=26):

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

        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.4),
        # tf.keras.layers.Dense(6, kernel_initializer=initializer, activation='softmax')
        tf.keras.layers.Conv2D(6, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Activation('softmax')
    ]

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=5)
    model = tf.keras.models.Sequential(layers)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    augmentor = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    if not tune_hypers:
        for fit in range(2):
            # 48x48 / 64x64
            # 48x48 images - 81.305933% accuracy
            # 64x64 images - 82.466024% accuracy
            # model.compile(tf.keras.optimizers.RMSprop(learning_rate=2.5e-4 * 10**-fit), loss=tf.keras.validation_losses.sparse_categorical_crossentropy, metrics=['accuracy'])

            # 100x100
            # 100x100 images - 85.449123% accuracy
            # model.compile(tf.keras.optimizers.RMSprop(learning_rate=6e-4 * 10**-fit), loss=tf.keras.validation_losses.sparse_categorical_crossentropy, metrics=['accuracy'])

            # 128x128
            # 128x128 images - 88.067615% accuracy
            model.compile(tf.keras.optimizers.RMSprop(learning_rate=4.5e-4 * 10**-fit), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

            # 156x156
            # 156x156 images - 86.54293% accuracy
            # model.compile(tf.keras.optimizers.RMSprop(learning_rate=4.5e-4 * 10**-fit), loss=tf.keras.validation_losses.sparse_categorical_crossentropy, metrics=['accuracy'])

            if not augment_data:
                history = model.fit(X_train, y_train, 26, epochs=25, validation_data=(X_val, y_val), verbose=2, callbacks=[callback])
                print()
                train_loss.extend(history.history['loss'])
                val_loss.extend(history.history['val_loss'])
                train_acc.extend(history.history['accuracy'])
                val_acc.extend(history.history['val_accuracy'])

            else:
                model.compile(tf.keras.optimizers.RMSprop(learning_rate=2e-4 * 10 ** -fit), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
                history = model.fit_generator(augmentor.flow(X_train, y_train, batch_size=26), validation_data=(X_val, y_val), epochs=25, verbose=2, callbacks=[callback])
                print()
                train_loss.extend(history.history['loss'])
                val_loss.extend(history.history['val_loss'])
                train_acc.extend(history.history['accuracy'])
                val_acc.extend(history.history['val_accuracy'])

        return train_loss, train_acc, val_loss, val_acc

        # return test_results[1]
    else:
        model.compile(learning_rate=learning_rate, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size, epochs=10, validation_data=(X_val, y_val), verbose=0)

        return history.history

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


def new_design_va80_batch_norm_before():
    # 48x48 input
    initializer = tf.keras.initializers.VarianceScaling(2)

    layers = [
        tf.keras.layers.Conv2D(64, 3, 1, 'same', input_shape=input_shape, kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        # tf.keras.layers.Activation('relu'),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(256, 3, 1, 'same', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(256, 3, 1, 'same', kernel_initializer=initializer),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Conv2D(8, 3, 1, 'same', kernel_initializer=initializer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.4),
        # tf.keras.layers.Dense(6, kernel_initializer=initializer, activation='softmax')
        tf.keras.layers.Conv2D(6, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Activation('softmax')
    ]

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=5)
    model = tf.keras.models.Sequential(layers)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for fit in range(2):
        model.compile(tf.keras.optimizers.RMSprop(learning_rate=3e-4 * 10**-fit),
                      loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

        # history = model.fit(X_train, y_train, 32, epochs=25, validation_data=(X_val, y_val), verbose=2, callbacks=[callback])
        history = model.fit(X_train, y_train, 64, epochs=16, validation_data=(X_val, y_val), verbose=2, callbacks=[callback])

        print()

        train_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        train_acc.extend(history.history['accuracy'])
        val_acc.extend(history.history['val_accuracy'])

    return train_loss, train_acc, val_loss, val_acc

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


def va80_but_with_layer_norm():
    # 48x48 input
    initializer = tf.keras.initializers.VarianceScaling(2)

    layers = [
        tf.keras.layers.Conv2D(64, 3, 1, 'same', input_shape=input_shape, kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(1),
        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(1),

        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(1),
        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(1),

        tf.keras.layers.Conv2D(256, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(1),
        tf.keras.layers.Conv2D(256, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(1),

        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(1),
        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(1),

        tf.keras.layers.Conv2D(8, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(1),

        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dropout(0.4),
        # tf.keras.layers.Dense(6, kernel_initializer=initializer, activation='softmax')
        tf.keras.layers.Conv2D(6, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Activation('softmax')
    ]

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=5)
    model = tf.keras.models.Sequential(layers)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for fit in range(2):
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=2.5e-4 * 10 ** -fit)
        # optimizer = tf.keras.optimizers.SGD(learning_rate=2e-4 * 10 ** -fit, momentum=0.9)
        model.compile(optimizer, loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

        history = model.fit(X_train, y_train, 26, epochs=25, validation_data=(X_val, y_val), verbose=2, callbacks=[callback])
        print()

        train_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        train_acc.extend(history.history['accuracy'])
        val_acc.extend(history.history['val_accuracy'])

    return train_loss, train_acc, val_loss, val_acc

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


def vgg16_transfer_learning():
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=7)
    vgg16 = tf.keras.applications.VGG16(input_shape=input_shape, include_top=False)
    vgg16.trainable = False
    initializer = tf.keras.initializers.VarianceScaling(2)
    model = tf.keras.Sequential([
        vgg16,
        tf.keras.layers.Conv2D(128, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(1),
        tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),

        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.BatchNormalization(1),

        tf.keras.layers.Conv2D(8, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.BatchNormalization(1),

        tf.keras.layers.Conv2D(6, 3, 1, 'same', kernel_initializer=initializer, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Activation('softmax')
    ])

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for fit in range(2):
        model.compile(tf.keras.optimizers.RMSprop(learning_rate=2.5e-4 * 10 ** -fit),
                      loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

        if fit == 0:
            epoch_zero_train_loss, epoch_zero_train_acc = model.evaluate(X_train, y_train, verbose=0)
            epoch_zero_val_loss, epoch_zero_val_acc = model.evaluate(X_val, y_val, verbose=0)

        history = model.fit(X_train, y_train, 26, epochs=25, validation_data=(X_val, y_val), verbose=2, callbacks=[callback])
        print()

        train_loss.extend(history.history['loss'])
        val_loss.extend(history.history['val_loss'])
        train_acc.extend(history.history['accuracy'])
        val_acc.extend(history.history['val_accuracy'])

    train_loss.insert(0, epoch_zero_train_loss)
    train_acc.insert(0, epoch_zero_train_acc)
    val_loss.insert(0, epoch_zero_val_loss)
    val_acc.insert(0, epoch_zero_val_acc)

    return train_loss, train_acc, val_loss, val_acc

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

    print(model.summary())

    plt.show()


def res_net():
    initializer = tf.keras.initializers.VarianceScaling(2)

    def res_block(inputs, filters):
        conv1 = tf.keras.layers.Conv2D(filters, 3, 1, 'same', kernel_initializer=initializer, activation='relu')(inputs)
        batch1 = tf.keras.layers.BatchNormalization()(conv1)
        conv2 = tf.keras.layers.Conv2D(filters, 3, 1, 'same', kernel_initializer=initializer)(batch1)
        batch2 = tf.keras.layers.BatchNormalization()(conv2)
        add = tf.keras.layers.Add()([batch2, inputs])
        out = tf.keras.layers.Activation('relu')(add)
        return out

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, 1, 'same', kernel_initializer=initializer, activation='relu')(x)

    num_res_blocks = 5
    for _ in range(num_res_blocks):
        x = res_block(x, 64)

    flatten = tf.keras.layers.Flatten()(x)
    scores = tf.keras.layers.Dense(6, kernel_initializer=initializer, activation='softmax')(flatten)

    model = tf.keras.models.Model(inputs=inputs, outputs=scores)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.1, patience=5)
    model.compile(tf.keras.optimizers.RMSprop(learning_rate=5e-4),
                  loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    history = model.fit(X_train, y_train, 200, epochs=25, validation_data=(X_val, y_val), verbose=2, callbacks=[callback])

    import matplotlib.pyplot as plt

    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(train_accuracy, label='Train accuracy')
    plt.plot(val_accuracy, label='Validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()


def random_search(num_iters):
    from science_fair_8th_9th.images import create_images_labels_datasets

    size = 128
    for _ in range(num_iters):
        lr = np.random.uniform(8e-5, 1e-3)
        # new_size = int(np.random.choice([48, 64, 100, 128, 156]))
        batch_size = np.random.randint(16, 33)

        global X_train, y_train, X_test, y_test, X_val, y_val
        global input_shape

        X_train, y_train, X_test, y_test, X_val, y_val = load_images()
        input_shape = X_train.shape[1:]

        print(f'LR: {lr:e} | Image Size: | {size} | Batch Size: {batch_size}')

        history = new_design_va80(False, True, lr, batch_size)
        train_loss = history['loss'][-1]
        train_acc = history['accuracy'][-1]
        val_loss = history['val_loss'][-1]
        val_acc = history['val_accuracy'][-1]

        print(f'Train Loss: {train_loss:.5f} | Train Accuracy: {train_acc:.5f}')
        print(f'Val Loss: {val_loss:.5f} | Val Accuracy: {val_acc:.5f}')
        print()


def save_results(name, train_loss, train_acc, val_loss, val_acc):
    train_loss = np.asarray(train_loss)
    train_acc = np.asarray(train_acc)
    val_loss = np.asarray(val_loss)
    val_acc = np.asarray(val_acc)
    np.save(f'training_losses/{name}.npy', train_loss)
    np.save(f'training_accuracies/{name}.npy', train_acc)
    np.save(f'validation_losses/{name}.npy', val_loss)
    np.save(f'validation_accuracies/{name}.npy', val_acc)


def create_results():
    from science_fair_8th_9th.images import create_images_labels_datasets, load_images
    global X_train, y_train, X_test, y_test, X_val, y_val, input_shape

    create_images_labels_datasets(128)
    X_train, y_train, X_test, y_test, X_val, y_val = load_images()
    input_shape = X_train.shape[1:]

    train_loss, train_acc, val_loss, val_acc = original_design()
    save_results('original_design', train_loss, train_acc, val_loss, val_acc)

    train_loss, train_acc, val_loss, val_acc = og_batch_before()
    save_results('og_batch_before', train_loss, train_acc, val_loss, val_acc)

    train_loss, train_acc, val_loss, val_acc = train_model(more_layers())
    save_results('more_layers', train_loss, train_acc, val_loss, val_acc)

    train_loss, train_acc, val_loss, val_acc = train_less_layers()
    save_results('less_layers', train_loss, train_acc, val_loss, val_acc)

    create_images_labels_datasets(212)
    X_train, y_train, X_test, y_test, X_val, y_val = load_images()
    input_shape = X_train.shape[1:]
    train_loss, train_acc, val_loss, val_acc = new_design_va75()
    save_results('new_design_va75', train_loss, train_acc, val_loss, val_acc)

    create_images_labels_datasets(64)
    X_train, y_train, X_test, y_test, X_val, y_val = load_images()
    input_shape = X_train.shape[1:]
    train_loss, train_acc, val_loss, val_acc = vgg16_transfer_learning()
    save_results('vgg16_transfer_learning', train_loss, train_acc, val_loss, val_acc)

    create_images_labels_datasets(128)


def load_results(metric='validation_accuracies'):
    # metric can be one of the following
    # training_losses, training_accuracies, validation_losses, validation_accuracies
    original_design = np.load(f'{metric}/original_design.npy', allow_pickle=True)
    og_batch_before = np.load(f'{metric}/og_batch_before.npy', allow_pickle=True)
    more_layers = np.load(f'{metric}/more_layers.npy', allow_pickle=True)
    less_layers = np.load(f'{metric}/less_layers.npy', allow_pickle=True)
    new_design_va75 = np.load(f'{metric}/new_design_va75.npy', allow_pickle=True)
    vgg16_transfer_learning = np.load(f'{metric}/vgg16_transfer_learning.npy', allow_pickle=True)
    TARnet = np.load(f'{metric}/TARnet.npy', allow_pickle=True)

    return original_design, og_batch_before, more_layers, less_layers, new_design_va75, vgg16_transfer_learning, TARnet


def create_graph(metric='validation_accuracies'):
    # metric can be one of the following
    # training_losses, training_accuracies, validation_losses, validation_accuracies
    titles = {'training_losses': 'Training Losses', 'training_accuracies': 'Training Accuracies',
              'validation_losses': "Validation Losses", 'validation_accuracies': ' Validation Accuracies'}

    import matplotlib.pyplot as plt

    original_design, og_batch_before, more_layers, less_layers, new_design_va75, vgg16_transfer_learning, TARnet = load_results(metric)

    plt.plot(original_design, label='original_design')
    plt.plot(og_batch_before, label='og_batch_before')
    plt.plot(more_layers, label='more_layers')
    plt.plot(less_layers, label='less_layers')
    plt.plot(new_design_va75, label='new_design_va75')
    plt.plot(vgg16_transfer_learning, label='vgg16_transfer_learning')
    plt.plot(TARnet, label='TARnet')
    plt.xlim(0, len(vgg16_transfer_learning))

    # plt.plot(range(1, original_design.size+1), original_design, label='original_design')
    # plt.plot(range(1, og_batch_before.size+1), og_batch_before, label='og_batch_before')
    # plt.plot(range(1, more_layers.size+1), more_layers, label='more_layers')
    # plt.plot(range(1, less_layers.size+1), less_layers, label='less_layers')
    # plt.plot(range(1, new_design_va75.size+1), new_design_va75, label='new_design_va75')
    # plt.plot(range(1, vgg16_transfer_learning.size+1), vgg16_transfer_learning, label='vgg16_transfer_learning')
    # plt.plot(range(1, MXnet.size+1), MXnet, label='MXnet')

    plt.xlabel('Epoch')
    if metric == 'training_accuracies' or metric == 'validation_accuracies':
        plt.ylabel('Accuracy')
        plt.hlines(0.85, 0, len(vgg16_transfer_learning), label='85% Accuracy')
    else:
        plt.ylabel('Loss')
    plt.title(titles[metric] + " of All 7 Models", fontsize=18)
    plt.legend(loc='lower right')
    plt.show()


create_graph('training_accuracies')
