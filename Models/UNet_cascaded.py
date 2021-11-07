
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation, UpSampling2D
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


def UNet_cascaded(input_shape):

    inputs = Input(input_shape)

    # first U-Net
    # ---------------------- Encoder ----------------------
    conv1_1 = conv_block(inputs, 32)
    conv1_2 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv1_1), 64)
    conv1_3 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv1_2), 128)
    conv1_4 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv1_3), 256)
    conv1_5 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv1_4), 512)

    # ---------------------- Decoder ----------------------
    deconv1_4 = conv_block(concatenate([UpSampling2D(size=(2, 2))(conv1_5), conv1_4], axis=-1), 256)
    deconv1_3 = conv_block(concatenate([UpSampling2D(size=(2, 2))(deconv1_4), conv1_3], axis=-1), 128)
    deconv1_2 = conv_block(concatenate([UpSampling2D(size=(2, 2))(deconv1_3), conv1_2], axis=-1), 64)
    deconv1_1 = conv_block(concatenate([UpSampling2D(size=(2, 2))(deconv1_2), conv1_1], axis=-1), 32)

    # second U-Net
    # ---------------------- Encoder ----------------------
    conv2_1 = conv_block(deconv1_1, 32)
    conv2_2 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv2_1), 64)
    conv2_3 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv2_2), 128)
    conv2_4 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv2_3), 256)
    conv2_5 = conv_block(MaxPooling2D(pool_size=(2, 2))(conv2_4), 512)

    # ---------------------- Decoder ----------------------
    deconv2_4 = conv_block(concatenate([UpSampling2D(size=(2, 2))(conv2_5), conv2_4], axis=-1), 256)
    deconv2_3 = conv_block(concatenate([UpSampling2D(size=(2, 2))(deconv2_4), conv2_3], axis=-1), 128)
    deconv2_2 = conv_block(concatenate([UpSampling2D(size=(2, 2))(deconv2_3), conv2_2], axis=-1), 64)
    deconv2_1 = conv_block(concatenate([UpSampling2D(size=(2, 2))(deconv2_2), conv2_1], axis=-1), 32)

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(deconv2_1)

    model = Model(inputs=inputs, outputs=output, name='UNet_base')
    return model


if __name__ == '__main__':

    model = UNet_cascaded(input_shape=(512, 512, 1))
    model.summary()