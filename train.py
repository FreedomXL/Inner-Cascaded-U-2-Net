
import os
from HDF5DatasetGenerator import HDF5DatasetGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from loss import *

from Models.UNet import UNet
from Models.UNet_cascaded import UNet_cascaded
from Models.UNet2Plus import UNet2Plus
from Models.U2_Net import U2_Net, U2_Net_light
from Models.Inner_Cascaded_UNet import Inner_Cascaded_UNet
from Models.Inner_Cascaded_U2_Net import Inner_Cascaded_U2_Net


dataset = 'LiTS'     # change dataset to 'BraTS' when training using BraTS 2013 dataset
save_path = 'save/'  # path for saving models and logs

train_path = '/home/user/datasets/LiTS/train.h5'
val_path = '/home/user/datasets/LiTS/val.h5'
TOTAL_TRAIN = 13144  # total train images in LiTS dataset
TOTAL_VAL = 3308     # total validation images in LiTS dataset
BATCH_SIZE = 2
input_shape = (512, 512, 1)
patience = 4
epochs = 20
if dataset == 'BraTS':
    train_path = '/home/user/datasets/LiTS/train.h5'
    val_path = '/home/user/datasets/LiTS/val.h5'
    TOTAL_TRAIN = 3361  # total train images in BraTS 2013 dataset
    TOTAL_VAL = 1110    # total validation images in BraTS 2013 dataset
    BATCH_SIZE = 8
    input_shape = (256, 256, 4)
    patience = 10
    epochs = 50


def train():
    train_reader = HDF5DatasetGenerator(db_path=train_path, batch_size=BATCH_SIZE)
    train_iter = train_reader.generator()

    val_reader = HDF5DatasetGenerator(db_path=val_path, batch_size=BATCH_SIZE)
    val_iter = val_reader.generator()

    model = UNet_cascaded(input_shape=input_shape)
    model.compile(optimizer=Adam(lr=1e-4), loss=focal_tversky_loss, metrics=[dice_coef])

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path + '/model')
        os.mkdir(save_path + '/model/logs')

    model_checkpoint = ModelCheckpoint(save_path + '/model/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=save_path + '/model/logs')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience, mode='auto')
    callbacks = [model_checkpoint, tensorboard, reduce_lr]

    model.fit_generator(train_iter,
                        steps_per_epoch=TOTAL_TRAIN//BATCH_SIZE,
                        epochs=epochs,
                        validation_data=val_iter,
                        validation_steps=TOTAL_VAL//BATCH_SIZE,
                        verbose=1,
                        callbacks=callbacks)

    train_reader.close()
    val_reader.close()

    print('Finished training ......')


if __name__ == '__main__':
    train()