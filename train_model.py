

from sklearn.model_selection import train_test_split
import os
import random

import tensorflow as tf
import cv2
import numpy as np

from skimage import morphology

# Import thu vien segmentation_models
from segmentation_models.metrics import iou_score
from segmentation_models import Unet
import segmentation_models as sm
sm.set_framework("tf.keras")
sm.framework()
from matplotlib import pyplot as plt

# Dinh nghia bien
data_path  = "dataset"
w, h = 256, 256
batch_size = 16

# Dataset va Dataloader

BACKBONE = "resnet34"
preprocess_input = sm.get_preprocessing(BACKBONE)

# Dung de tao toan bo du lieu va load theo batch
class Dataset:
    def __init__(self, image_path, mask_path, w, h):
        # the paths of images
        self.image_path = image_path
        # the paths of segmentation images
        self.mask_path = mask_path

        self.w = w
        self.h = h

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.image_path[i])
        image = cv2.resize(image, (self.w, self.h), interpolation=cv2.INTER_AREA)
        image = preprocess_input(image)

        mask = cv2.imread(self.mask_path[i], cv2.IMREAD_UNCHANGED)
        image_mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_AREA)

        image_mask = [(image_mask == v) for v in [1]]
        image_mask = np.stack(image_mask, axis=-1).astype('float')

        return image, image_mask

class Dataloader(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size,shape, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shape = shape
        self.indexes = np.arange(self.shape)

    def __getitem__(self, i):
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

#  Load thong tin tu folder dataset de tao 2 bien image_path, mask_path
def load_path(data_path):
    # Get normal image and mask
    classes = ['Class1']

    # Lop qua cac thu muc khong loi
    normal_image_path = []
    normal_mask_path = []
    for class_ in classes:
        current_folder = os.path.join(data_path, class_)
        for file in os.listdir(current_folder):
            if file.endswith("jpg") and (not file.startswith(".")):
                image_path = os.path.join(current_folder, file)
                mask_path = os.path.join(current_folder + "_mask", file)
                normal_mask_path.append(mask_path)
                normal_image_path.append(image_path)

    # Get defect image and mask
    defect_image_path = []
    defect_mask_path = []
    for class_ in classes:
        class_ = class_ + "_def"
        current_folder = os.path.join(data_path, class_)
        for file in os.listdir(current_folder):
            if file.endswith("jpg") and (not file.startswith(".")):
                image_path = os.path.join(current_folder, file)
                mask_path = os.path.join(current_folder + "_mask", file)
                defect_mask_path.append(mask_path)
                defect_image_path.append(image_path)

    # Normal:   normal_mask_path - chua toan bo duong dan cua mask
    #            normal_image_path-  chua toan bo duong dan den image
    # Defect:   defect_mask_path - chua toan bo duong dan cua mask
    #            defect_image_path-  chua toan bo duong dan den image

    idx = random.sample(range(len(normal_mask_path)), len(defect_mask_path))

    normal_mask_path_new = []
    normal_image_path_new = []

    for id in idx:
        normal_image_path_new.append(normal_image_path[id])
        normal_mask_path_new.append(normal_mask_path[id])

    image_path = normal_image_path_new + defect_image_path
    mask_path = normal_mask_path_new + defect_mask_path

    return image_path, mask_path

# Thu hien load va train model

# Load duong dan vao 2 bien
image_path, mask_path = load_path(data_path)

# Chia du lieu train, test
image_train, image_test, mask_train, mask_test = train_test_split(image_path, mask_path, test_size=0.05)

# Tao dataset va dataloader
train_dataset = Dataset(image_train, mask_train, w, h)
test_dataset = Dataset(image_test, mask_test, w, h)

train_loader = Dataloader(train_dataset, batch_size, shape=len(image_train), shuffle=True)
test_loader = Dataloader(test_dataset, batch_size, shape=len(image_test), shuffle=True)



# Khoi tao model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
opt=tf.keras.optimizers.Adam(0.001)

#model= Unet(backbone_name='resnet34',classes=1,activation="sigmoid",input_shape=(256,256,3),encoder_freeze=True)
model= Unet(backbone_name='vgg16',encoder_weights="imagenet",classes=1,activation="sigmoid",input_shape=(256,256,3),encoder_freeze=True)
#model= Unet(input_shape=(256, 256, 3), classes=1, activation='sigmoid',  encoder_features='default', decoder_block_type='upsampling', decoder_filters=(256, 128, 64, 32, 16), decoder_use_batchnorm=True)
loss1 = sm.losses.categorical_focal_dice_loss
#model.compile(optimizer=opt,loss=loss1,metrics=[iou_score])
model.compile(optimizer=opt,loss=loss1,metrics=[iou_score,'accuracy'])
# Train model
is_train = True

if is_train:
    from keras.callbacks import ModelCheckpoint
    filepath="checkpoint13.hdf5"
    callback = ModelCheckpoint(filepath, monitor='val_iou_score', verbose=1, save_best_only=True,mode='max')

    #history=model.fit_generator( train_loader, validation_data=test_loader, epochs=10, callbacks=[callback])
    history = model.fit_generator(train_loader, validation_data=test_loader, epochs=2, callbacks=[callback])
    final_accuracy = history.history["val_accuracy"][-5:]
    print("FINAL ACCURACY MEAN-5: ", np.mean(final_accuracy))
    plt.plot(history.history['accuracy'])
    plt.show()
else:
    # Load model de test
    model.load_weights("checkpoint13.hdf5")

    ids = range(len(image_test))
    index = random.sample(ids, 20)

    #import matplotlib.pyplot as plt

    for id in index:

        # Anh dau vao, ko phai mask
        image = cv2.imread(image_test[id])
        image = cv2.resize(image, (256, 256))
        # Dua qua model de predicted segmentation map
        img_expand = image[np.newaxis, ...]
        mask_predict = model.predict(image[np.newaxis, :, :, :])
        #prediction1 = model1.predict(image)
        # Doc mask thuc te
        image_mask = cv2.imread(mask_test[id], cv2.IMREAD_UNCHANGED)
        image_mask = cv2.resize(image_mask, (256, 256))

        plt.figure(figsize=(10, 6))
        plt.subplot(221)
        plt.title("Hình ảnh sản phẩm")
        plt.imshow(image)
        plt.subplot(222)
        plt.title("Vết lỗi thật")
        plt.imshow(image_mask)
        plt.subplot(223)
        plt.title("Vết lỗi dự đoán")
        #z = mask_predict[0, :, :]
        t = mask_predict[0,:,:]
        z=t
        plt.imshow(t)

        plt.subplot(224)

        z[z < 0.5] = 0
        z[z >= 0.5] = 1

        ##kernel2 = np.ones((20, 20), np.uint8)
        #kernel3 = np.ones((10, 10), np.uint8)

        kernel = morphology.diamond(3)
        kernel2 = morphology.diamond(7)
        kernel3 = morphology.diamond(10)
        kernel4 = morphology.ball(10)

        #kernel2 = cv2.getGaussianKernel(10,5)
        h, w = image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)


        #z = cv2.erode(z, kernel3)

        def fillhole(input_image):
            '''
            input gray binary image  get the filled image by floodfill method
            Note: only holes surrounded in the connected regions will be filled.
            :param input_image:
            :return:
            '''
            im_flood_fill = input_image.copy()
            h, w = input_image.shape[:2]
            mask = np.zeros((h + 2, w + 2), np.uint8)
            im_flood_fill = im_flood_fill.astype("uint8")
            cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
            im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
            img_out = im_flood_fill_inv
            return img_out
        #[z < 0.5] = 0
        #z[z >= 0.5] = 1
        z = cv2.morphologyEx(z, cv2.MORPH_GRADIENT, kernel)

        z = cv2.dilate(z, kernel2)
        z=fillhole(z)
        z=cv2.erode(z, kernel2)

        plt.imshow(z)


        plt.show()

#point6- batch4 epoch 20
#point7- batch8 epoch 20
#     8       8       20    freeze faslse
#     9       8       30    -------------
#     10      8       10  --------

#     11      8       10    tốt resnet34
#----------------------20









