from PIL import Image
import numpy as np
import tensorflow as tf

materials = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']


def get_images(data_set):
    X = []
    y = []
    text_file_name = 'one-indexed-files-notrash_' + data_set + '.txt'
    with open('all_the_images/' + text_file_name, 'r') as file:
        all_images = file.readlines()

    for im in all_images:
        name, label = im.split(' ')
        label = int(label[:-1]) - 1
        material = ''
        for char in name:
            if char.isdigit():
                break
            material += char

        picture = Image.open('all_the_images/Garbage_classification/' + material + '/' + name)
        # picture = picture.resize((64, 64)) ideal
        picture = picture.resize((32, 32))
        picture = np.array(picture).reshape(-1)
        X.append(picture)
        y.append(label)

    return np.array(X), np.array(y)


def past_load_images():
    X_train, y_train = get_images('train')
    X_test, y_test = get_images('test')
    X_val, y_val = get_images('val')
    # print('Done fetching data.')
    # print('The images are reshaped into vectors before being preprocessed. You will have to change the fetching images before implementing the neural network.')
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    # print('Done preprocessing data.')
    return X_train, y_train, X_test, y_test, X_val, y_val


def load_images(shuffle_before=True, num_augments=0, train_size=1768, test_size=431, val_size=328, pytorch=False):
    images = np.load('images_dataset.npy')
    labels = np.load('labels_dataset.npy')

    indx = np.arange(len(images))
    if shuffle_before:
        np.random.shuffle(indx)
    images = images[indx]
    labels = labels[indx]

    X_train = images[:train_size]
    y_train = labels[:train_size]

    X_test = images[train_size:train_size+test_size]
    y_test = labels[train_size:train_size+test_size]

    X_val = images[train_size+test_size:train_size+test_size+val_size]
    y_val = labels[train_size+test_size:train_size+test_size+val_size]

    # data_gen_args = dict(rotation_range=90,
    #                      width_shift_range=0.1,
    #                      height_shift_range=0.1,
    #                      zoom_range=0.2)

    data_gen_args = dict(rotation_range=360,
                         width_shift_range=0.2,
                         height_shift_range=0.15,
                         zoom_range=0.3,
                         horizontal_flip=True,
                         vertical_flip=True)

    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

    for _ in range(num_augments):
        X_train2, y_train2 = image_datagen.flow(X_train, y_train, batch_size=X_train.shape[0])[0]
        X_train, y_train = np.concatenate((X_train, X_train2)), np.concatenate((y_train, y_train2))

    pix_mean = np.mean(X_train, axis=(0, 1, 2), keepdims=True)
    pix_std = np.std(X_train, axis=(0, 1, 2), keepdims=True)


    X_train = (X_train - pix_mean) / pix_std
    X_test = (X_test - pix_mean) / pix_std
    X_val = (X_val - pix_mean) / pix_std

    if pytorch:
        X_train = X_train.transpose((0, 3, 1, 2))
        X_val = X_val.transpose((0, 3, 1, 2))
        X_test = X_test.transpose((0, 3, 1, 2))

    return X_train, y_train, X_test, y_test, X_val, y_val


def show_single_image():
    import matplotlib.pyplot as plt

    data_set = 'train'
    text_file_name = 'one-indexed-files-notrash_' + data_set + '.txt'
    with open('all_the_images/' + text_file_name, 'r') as file:
        all_images = file.readlines()

    np.random.shuffle(all_images)
    for im in all_images:
        name, label = im.split(' ')
        label = int(label[:-1]) - 1
        material = ''
        for char in name:
            if char.isdigit():
                break
            material += char

        picture = Image.open('all_the_images/Garbage_classification/' + material + '/' + name)
        # plt.subplot(2, 2, 1)
        # plt.imshow(picture)
        # plt.title(f'{material} | Label: {label} | Size: {picture.size}')
        #
        # scaled_picture = picture.resize((32, 32))  # og
        # plt.subplot(2, 2, 2)
        # plt.imshow(scaled_picture)
        # plt.title(f'{material} | Label: {label} | Size: {scaled_picture.size}')
        #
        # scaled_picture = picture.resize((64, 64))
        # plt.subplot(2, 2, 3)
        # plt.imshow(scaled_picture)
        # plt.title(f'{material} | Label: {label} | Size: {scaled_picture.size}')
        #
        # scaled_picture = picture.resize((128, 128))
        # plt.subplot(2, 2, 4)
        # plt.imshow(scaled_picture)
        # plt.title(f'{material} | Label: {label} | Size: {scaled_picture.size}')
        #
        # plt.tight_layout(h_pad=1)

        # print(np.asarray(picture).shape) images converted to numpy array have the shape (384, 512, 3)
        # fig, axs = plt.subplots(2, 2, constrained_layout=True)
        # fig.suptitle(f'{material}, {label}', fontsize=18)
        #
        # axs[0, 0].imshow(picture)
        # axs[0, 0].set_title(f'Size: {picture.size}')
        #
        # scaled_picture = picture.resize((64, 64))
        # axs[0, 1].imshow(scaled_picture)
        # axs[0, 1].set_title(f'Size: {scaled_picture.size}')
        #
        # scaled_picture = picture.resize((128, 128))
        # axs[1, 0].imshow(scaled_picture)
        # axs[1, 0].set_title(f'Size: {scaled_picture.size}')
        #
        # scaled_picture = picture.resize((256, 256))
        # axs[1, 1].imshow(scaled_picture)
        # axs[1, 1].set_title(f'Size: {scaled_picture.size}')

        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        fig.suptitle(f'{material}, {label}', fontsize=18)

        axs[0].imshow(picture)
        axs[0].set_title(f'Size: {picture.size}')

        scaled_picture = picture.resize((128, 128))
        axs[1].imshow(scaled_picture)
        axs[1].set_title(f'Size: {scaled_picture.size}')

        plt.show()


def create_images_labels_datasets(resolution=128):
    X = []
    y = []
    with open('all_the_images/' + 'zero-indexed-files.txt', 'r') as file:
        all_images = file.readlines()

    for im in all_images:
        name, label = im.split(' ')
        label = int(label[:-1])

        material = ''.join([char for char in name if not char.isdigit()][:-4])

        picture = Image.open('all_the_images/Garbage_classification/' + material + '/' + name)
        picture = picture.resize((resolution, resolution))
        X.append(np.asarray(picture))
        y.append(np.asarray(label))
    assert len(X) == len(y), 'The number of images does not equal the number of labels'

    indx = np.arange(len(X))
    np.random.shuffle(indx)
    X, y = np.asarray(X)[indx], np.asarray(y)[indx]
    np.save('images_dataset.npy', X)
    np.save('labels_dataset.npy', y)


def new_load_vs_past_load():
    from time import time

    old_start = time()
    past_load_images()
    old_time = time() - old_start

    new_start = time()
    load_images()
    new_time = time() - new_start

    print(f'Past load time: {old_time}')
    print(f'New load time: {new_time}')


def show_augmented_images():
    import tensorflow as tf
    import matplotlib.pyplot as plt

    images = np.load('images_dataset.npy')
    labels = np.load('labels_dataset.npy')
    augmentor = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    augmented_images = augmentor.flow(images, labels, shuffle=False)  # don't forget to manually set batch size
    for new, image, label in zip(range(len(images)), images, labels):
        plt.subplot(2, 1, 1)
        plt.title(f'{materials[label]}')
        plt.imshow(image)
        plt.subplot(2, 1, 2)
        plt.imshow(augmented_images[0][0][new]/255)
        plt.show()


def show_many_images(images_per_class=5):
    # images per class can not be too large or too big to prevent large white spaces between subplots
    # (hypothesis) The width and height of all of the subplots has too be the same. When the
    # images_per_class is too large, the height is inherently going to much larger than the width
    # since the number of columns is going to increase. Thus, to maintain the equal aspect ratio
    # (the default aspect ratio in which the height of each image is equal to its width), matplotlib
    # adds white space between the images. The same affect happens when images_per_class is too small.
    # You could manually change the aspect parameter in imshow to 'auto,' so the aspect ratio of each image can change
    # to make the width and height of all the subplots to be the same, but then the images may not be
    # squares. To use aspect='auto' and keep the images as squares,
    # you could manually adjust the matplotlib window (i.e. shrink it) to shrink the images.

    import matplotlib.pyplot as plt

    num_classes = 6
    images = np.load('images_dataset.npy')
    labels = np.load('labels_dataset.npy')
    while True:
        for category in range(6):
            category_labels = np.flatnonzero(labels == category)
            np.random.shuffle(category_labels)
            for i in range(images_per_class):
                index = 1 + category + num_classes * i
                plt.subplot(images_per_class, num_classes, index)
                if i == 0:
                    plt.title(materials[category], fontsize=16)
                plt.imshow(images[category_labels[i]])
                plt.axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # show_many_images()
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt

    data_gen_args = dict(rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    X_train, y_train, X_test, y_test, X_val, y_val = load_images(shuffle_before=True)
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    test = image_datagen.flow(X_train, y_train, batch_size=X_train.shape[0])
    x, y = test[0]
    for i in range(X_train.shape[0]):
        plt.title(materials[y[i]])
        plt.imshow((x[i]*255).astype(np.uint8))
        plt.show()






