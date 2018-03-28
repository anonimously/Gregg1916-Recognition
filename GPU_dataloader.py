import matplotlib.image as mpimg
import numpy as np
import os
from configurations import CONFIG
import keras
import keras.utils.data_utils
from PIL import Image
import PIL.ImageOps


def rgb2grey(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def data_split(folder=CONFIG.data_folder, val_proportion=CONFIG.val_proportion):
    """
    split the data by filenames into train, val and test sets
    :param folder:
    :return:
    """
    files = os.listdir(folder)
    train_files, val_files, test_files = list(), list(), list()
    period = int(np.round(1 / val_proportion))

    corruption = list()

    max_H, max_W = 0, 0

    for (i, file) in enumerate(files):

        for char in file[:-4]:
            if char not in 'abcdefghijklmnopqrstuvwxyz':
                print(file)

        if i % period == 0:
            val_files.append(file)
        elif i % period == 1:
            test_files.append(file)
        else:
            train_files.append(file)
        try:
            image = rgb2grey(mpimg.imread(os.path.join(folder, file)))
        except ValueError:
            corruption.append(file)
            print(file)
        max_H = np.max([np.shape(image)[0], max_H])
        try:
            max_W = np.max([np.shape(image)[1], max_W])
        except IndexError:
            corruption.append(file)
            print(file)
    max_label_length = np.max([len(file) - 4 for file in files])

    return train_files, val_files, test_files, max_H, max_W, max_label_length + 2


'''
def data_inverse(folder=CONFIG.inv_data_folder):
    files = os.listdir(folder)
    for file in files:
        inv_file_name = file[:-4][::-1] + '_' + file[-4:]
        os.rename(os.path.join(folder, file), dst=os.path.join(folder, inv_file_name))
    print('stage 1 finished')
    files = os.listdir(folder)
    for file in files:
        new_name = file[:-5] + file[-4:]
        os.rename(os.path.join(folder, file), os.path.join(folder, new_name))
'''


def augmentation_simple(filename, aug_type, max_H, max_W, folder=CONFIG.data_folder):
    """
    augmentation on a single image.
    !! currently type = [0..5]
    :param filename:
    :param aug_type:
    :param folder:
    :return:
    """

    image = rgb2grey(mpimg.imread(os.path.join(folder, filename)))
    image_augmented = np.ones(shape=(max_H, max_W))
    (h, w) = np.shape(image)
    stride_0, stride_1 = max_H - h, (max_W - w) // 2
    offset = ((aug_type % 2) * stride_0, (aug_type % 3) * stride_1)
    image_augmented[offset[0]: h + offset[0], offset[1]: w + offset[1]] = image

    return image_augmented


def augmentation_nine(filename, aug_type, max_H, max_W, folder=CONFIG.data_folder):
    """
    augmentation on a single image.
    !! currently type = [0..5]
    :param filename:
    :param aug_type:
    :param folder:
    :return: 2D H*W image
    """

    # image = rgb2grey(mpimg.imread(os.path.join(folder, filename)))

    # rotating a 214 pixel image by 2 deg yield 8 more pixels
    image_augmented = np.ones(shape=(max_H, max_W))
    image = Image.open(os.path.join(folder, filename))
    image = image.convert('RGB')
    # note that Image read rgb imgs as 0-255
    #################################
    # aug_type = 8

    w_ori, h_ori = image.size

    rotate_ind = aug_type % 3
    scale_ind = aug_type // 3

    image = PIL.ImageOps.invert(image)
    if rotate_ind == 1:
        image = image.rotate(2, expand=True)
    elif rotate_ind == 2:
        image = image.rotate(-2, expand=True)
    image = PIL.ImageOps.invert(image)

    h, w = image.size

    if scale_ind == 1:
        h, w = np.int(np.floor(h * 0.98)), np.int(np.floor(w * 0.98))
        image = image.resize((h, w))
    elif scale_ind == 2:
        h, w = np.int(np.floor(h * 0.96)), np.int(np.floor(w * 0.96))
        image = image.resize((h, w))

    # put image there. 9 images in total. this enhalts shifting.
    # scale to (0-1)
    image = rgb2grey(np.array(image) / 255)

    h, w = np.shape(image)

    stride_0, stride_1 = (max_H - 10 - h_ori) // 2, (max_W - 10 - w_ori) // 2
    offset = ((aug_type % 3) * stride_0, (aug_type % 3) * stride_1)
    try:
        image_augmented[offset[0]: h + offset[0], offset[1]: w + offset[1]] = image
    except ValueError:
        print(filename)

    return image_augmented


class ShorthandGenerationSequence(keras.utils.data_utils.Sequence):
    """
    generate seq for img->seq models
    encodes data augmentation
    """
    def __init__(self, file_list, max_H, max_W, max_label_leng, aug_types,
                 channels=1, batchsize=CONFIG.batch_size):
        """
        max_label_length is the length of label PLUS begin / end markers
        aug_types:
            9: aug 9
            6: aug 6
            1: no augmentation, validation mode.
        """
        self.file_list = file_list
        self.H, self.W = max_H, max_W
        self.batch_size = batchsize
        self.channels = channels
        self.vocabulary = 'abcdefghijklmnopqrstuvwxyz+#'
        self.dict_c2i = dict()
        self.max_label_length = max_label_leng
        self.max_context_length = self.max_label_length - 1

        self.num_batches_by_length = dict()
        self.aug_types = aug_types

        self.instance_counts = dict()
        self.instance_indices_by_length = dict()

        for char in self.vocabulary:
            self.dict_c2i[char] = len(self.dict_c2i)

        for i in range(1, self.max_context_length + 1):
            self.instance_indices_by_length[i] = list()

        for file in file_list:
            "kicks out '.png' in the file names"
            seq = file[:-4]
            seq = '+' + seq + '#'
            max_context_len = len(seq) - 1
            for length in range(1, max_context_len + 1):
                for aug in range(self.aug_types):
                    self.instance_indices_by_length[length].append([seq, aug, length])

        for i in range(1, self.max_context_length + 1):
            self.num_batches_by_length[i] = len(self.instance_indices_by_length[i]) // self.batch_size

        self.total_size = np.sum([len(self.instance_indices_by_length[i]) for i in range(1, self.max_context_length)])

    def __len__(self):
        # we disgard all incomplete batches
        return np.int(np.sum([self.num_batches_by_length[length] for length in self.num_batches_by_length]))

    def __getitem__(self, idx):
        """
        returns NHWC images and one-hot labels
        :param idx:
        :return:
        """
        context_length = 1
        while np.int(np.sum([self.num_batches_by_length[length]
                             for length in self.num_batches_by_length if length <= context_length])) < idx:
            context_length += 1

        num_batch_in_length = idx - \
            np.int(np.sum([self.num_batches_by_length[length]
                           for length in self.num_batches_by_length if length <= context_length]))

        starting_index = num_batch_in_length * self.batch_size

        batch_img = np.zeros(shape=(self.batch_size, self.H, self.W, 1))
        batch_x_context = np.zeros(shape=(self.batch_size, context_length))
        batch_y = np.zeros(shape=(self.batch_size,))

        for ind_offset in range(self.batch_size):
            seq, augmentation_type, instance_context_length = \
                self.instance_indices_by_length[context_length][starting_index + ind_offset]

            file_name = seq[1:-1] + '.png'
            batch_img[ind_offset, :, :, 0] = augmentation_nine(
                file_name, augmentation_type, self.H, self.W)
            # batch_img[ind_offset, :, :, 0] = augmentation_simple(
            #      file_name, augmentation_type, self.H, self.W)

            for i in range(instance_context_length):
                batch_x_context[ind_offset, i] = self.dict_c2i[self.vocabulary[i]]

            batch_y[ind_offset] = self.dict_c2i[seq[instance_context_length]]

        batch_y = keras.utils.to_categorical(batch_y, num_classes=len(self.vocabulary))
        if self.channels == 3:
            batch_img = np.concatenate([batch_img, batch_img, batch_img], -1)
        return [batch_img, batch_x_context], batch_y


# train_files, val_files, test_files, max_H, max_W, max_seq_length = data_split()
# max_W += 10
# max_H += 10

# dill.dump((train_files, val_files, test_files, max_H, max_W, max_seq_length), open(CONFIG.file_list, 'wb'))

# data_inverse()


'''
train_files, val_files, test_files, max_H, max_W, max_seq_length = data_split()
max_W += 10
max_H += 10

for i, file in enumerate(train_files+val_files+test_files):
    for aug in range(9):
        try:
            img = augmentation_nine(file, aug, max_H, max_W)
        except ValueError:
            try:
                img = augmentation_nine(file, aug, max_H, max_W)
            except ValueError:
                pass
        if i % 100 == 0:
            print(i)
'''
