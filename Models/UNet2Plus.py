
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, concatenate, UpSampling2D
from keras.layers.normalization import BatchNormalization as BN


def conv_bn_relu(input, filters, kernel_size=3):

    conv = Conv2D(filters, (kernel_size, kernel_size), padding='same')(input)
    conv = BN(axis=-1)(conv)
    conv = Activation('relu')(conv)

    return conv


def conv_block(input, filters, kernel_size=3):

    conv = conv_bn_relu(input, filters, kernel_size=kernel_size)
    conv = conv_bn_relu(conv, filters, kernel_size=kernel_size)

    return conv


def UNet2Plus(input_shape):

    inputs = Input(input_shape)

    conv0_0 = conv_block(inputs, 32)

    conv1_0 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv0_0), 64)
    conv0_1 = conv_block(concatenate([conv0_0, UpSampling2D(size=(2, 2))(conv1_0)], axis=3), 32)

    conv2_0 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv1_0), 128)
    conv1_1 = conv_block(concatenate([conv1_0, UpSampling2D(size=(2, 2))(conv2_0)], axis=3), 64)
    conv0_2 = conv_block(concatenate([conv0_0, conv0_1, UpSampling2D(size=(2, 2))(conv1_1)], axis=3), 32)

    conv3_0 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv2_0), 256)
    conv2_1 = conv_block(concatenate([conv2_0, UpSampling2D(size=(2, 2))(conv3_0)], axis=3), 128)
    conv1_2 = conv_block(concatenate([conv1_0, conv1_1, UpSampling2D(size=(2, 2))(conv2_1)], axis=3), 64)
    conv0_3 = conv_block(concatenate([conv0_0, conv0_1, conv0_2, UpSampling2D(size=(2, 2))(conv1_2)], axis=3), 32)

    conv4_0 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv3_0), 512)
    conv3_1 = conv_block(concatenate([conv3_0, UpSampling2D(size=(2, 2))(conv4_0)], axis=3), 256)
    conv2_2 = conv_block(concatenate([conv2_0, conv2_1, UpSampling2D(size=(2, 2))(conv3_1)], axis=3), 128)
    conv1_3 = conv_block(concatenate([conv1_0, conv1_1, conv1_2, UpSampling2D(size=(2, 2))(conv2_2)], axis=3), 64)
    conv0_4 = conv_block(concatenate([conv0_0, conv0_1, conv0_2, conv0_3, UpSampling2D(size=(2, 2))(conv1_3)], axis=3), 32)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(conv0_4)

    model = Model(inputs=inputs, outputs=output, name='UNet2Plus')
    return model


if __name__ == '__main__':

    model = UNet2Plus(input_shape=(512, 512, 1))
    model.summary()