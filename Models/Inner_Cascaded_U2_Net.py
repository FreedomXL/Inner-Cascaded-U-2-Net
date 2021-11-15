
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation, UpSampling2D, add
from keras.layers.normalization import BatchNormalization as BN


def conv_bn_relu(input, filters, kernel_size=3, dilation=1):

    conv = Conv2D(filters, (kernel_size, kernel_size), dilation_rate=(dilation, dilation), padding='same')(input)
    conv = BN(axis=-1)(conv)
    conv = Activation('relu')(conv)

    return conv


def RSU7(input, mid_channel, out_channel):

    conv_in = conv_bn_relu(input, out_channel, dilation=1)

    # ---------------------- Encoder ----------------------
    conv1 = conv_bn_relu(conv_in, mid_channel, dilation=1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_bn_relu(pool1, mid_channel, dilation=1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_bn_relu(pool2, mid_channel, dilation=1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_relu(pool3, mid_channel, dilation=1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_relu(pool4, mid_channel, dilation=1)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

    conv6 = conv_bn_relu(pool5, mid_channel, dilation=1)

    conv7 = conv_bn_relu(conv6, mid_channel, dilation=2)

    # ---------------------- Decoder ----------------------
    deconv6 = conv_bn_relu(concatenate([conv7, conv6], axis=-1), mid_channel, dilation=1)
    deconv6 = UpSampling2D(size=(2, 2))(deconv6)

    deconv5 = conv_bn_relu(concatenate([deconv6, conv5], axis=-1), mid_channel, dilation=1)
    deconv5 = UpSampling2D(size=(2, 2))(deconv5)

    deconv4 = conv_bn_relu(concatenate([deconv5, conv4], axis=-1), mid_channel, dilation=1)
    deconv4 = UpSampling2D(size=(2, 2))(deconv4)

    deconv3 = conv_bn_relu(concatenate([deconv4, conv3], axis=-1), mid_channel, dilation=1)
    deconv3 = UpSampling2D(size=(2, 2))(deconv3)

    deconv2 = conv_bn_relu(concatenate([deconv3, conv2], axis=-1), mid_channel, dilation=1)
    deconv2 = UpSampling2D(size=(2, 2))(deconv2)

    deconv1 = conv_bn_relu(concatenate([deconv2, conv1], axis=-1), out_channel, dilation=1)

    return add([deconv1, conv_in])


def RSU6(input, mid_channel, out_channel):

    conv_in = conv_bn_relu(input, out_channel, dilation=1)

    # ---------------------- Encoder ----------------------

    conv1 = conv_bn_relu(conv_in, mid_channel, dilation=1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_bn_relu(pool1, mid_channel, dilation=1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_bn_relu(pool2, mid_channel, dilation=1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_relu(pool3, mid_channel, dilation=1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_bn_relu(pool4, mid_channel, dilation=1)

    conv6 = conv_bn_relu(conv5, mid_channel, dilation=2)

    # ---------------------- Decoder ----------------------
    deconv5 = conv_bn_relu(concatenate([conv6, conv5], axis=-1), mid_channel, dilation=1)
    deconv5 = UpSampling2D(size=(2, 2))(deconv5)

    deconv4 = conv_bn_relu(concatenate([deconv5, conv4], axis=-1), mid_channel, dilation=1)
    deconv4 = UpSampling2D(size=(2, 2))(deconv4)

    deconv3 = conv_bn_relu(concatenate([deconv4, conv3], axis=-1), mid_channel, dilation=1)
    deconv3 = UpSampling2D(size=(2, 2))(deconv3)

    deconv2 = conv_bn_relu(concatenate([deconv3, conv2], axis=-1), mid_channel, dilation=1)
    deconv2 = UpSampling2D(size=(2, 2))(deconv2)

    deconv1 = conv_bn_relu(concatenate([deconv2, conv1], axis=-1), out_channel, dilation=1)

    return add([deconv1, conv_in])


def RSU5(input, mid_channel, out_channel):

    conv_in = conv_bn_relu(input, out_channel, dilation=1)

    # ---------------------- Encoder ----------------------

    conv1 = conv_bn_relu(conv_in, mid_channel, dilation=1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_bn_relu(pool1, mid_channel, dilation=1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_bn_relu(pool2, mid_channel, dilation=1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_bn_relu(pool3, mid_channel, dilation=1)

    conv5 = conv_bn_relu(conv4, mid_channel, dilation=2)

    # ---------------------- Decoder ----------------------
    deconv4 = conv_bn_relu(concatenate([conv5, conv4], axis=-1), mid_channel, dilation=1)
    deconv4 = UpSampling2D(size=(2, 2))(deconv4)

    deconv3 = conv_bn_relu(concatenate([deconv4, conv3], axis=-1), mid_channel, dilation=1)
    deconv3 = UpSampling2D(size=(2, 2))(deconv3)

    deconv2 = conv_bn_relu(concatenate([deconv3, conv2], axis=-1), mid_channel, dilation=1)
    deconv2 = UpSampling2D(size=(2, 2))(deconv2)

    deconv1 = conv_bn_relu(concatenate([deconv2, conv1], axis=-1), out_channel, dilation=1)

    return add([deconv1, conv_in])


def RSU4(input, mid_channel, out_channel):

    conv_in = conv_bn_relu(input, out_channel, dilation=1)

    # ---------------------- Encoder ----------------------
    conv1 = conv_bn_relu(conv_in, mid_channel, dilation=1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_bn_relu(pool1, mid_channel, dilation=1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_bn_relu(pool2, mid_channel, dilation=1)

    conv4 = conv_bn_relu(conv3, mid_channel, dilation=2)

    # ---------------------- Decoder ----------------------
    deconv3 = conv_bn_relu(concatenate([conv4, conv3], axis=-1), mid_channel, dilation=1)
    deconv3 = UpSampling2D(size=(2, 2))(deconv3)

    deconv2 = conv_bn_relu(concatenate([deconv3, conv2], axis=-1), mid_channel, dilation=1)
    deconv2 = UpSampling2D(size=(2, 2))(deconv2)

    deconv1 = conv_bn_relu(concatenate([deconv2, conv1], axis=-1), out_channel, dilation=1)

    return add([deconv1, conv_in])


def RSU4F(input, mid_channel, out_channel):

    conv_in = conv_bn_relu(input, out_channel, dilation=1)

    # ---------------------- Encoder ----------------------
    conv1 = conv_bn_relu(conv_in, mid_channel, dilation=1)
    conv2 = conv_bn_relu(conv1, mid_channel, dilation=2)
    conv3 = conv_bn_relu(conv2, mid_channel, dilation=4)

    conv4 = conv_bn_relu(conv3, mid_channel, dilation=8)

    # ---------------------- Decoder ----------------------
    deconv3 = conv_bn_relu(concatenate([conv4, conv3], axis=-1), mid_channel, dilation=4)
    deconv2 = conv_bn_relu(concatenate([deconv3, conv2], axis=-1), mid_channel, dilation=2)
    deconv1 = conv_bn_relu(concatenate([deconv2, conv1], axis=-1), out_channel, dilation=1)

    return add([deconv1, conv_in])


def Inner_Cascaded_U2_Net(input_shape):

    inputs = Input(input_shape)

    # ---------------------- Encoder ----------------------
    conv1_1 = RSU7(inputs, 32, 64)
    pool1_1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)

    conv1_2 = RSU6(pool1_1, 32, 128)
    pool1_2 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv1_3 = RSU5(pool1_2, 64, 256)
    pool1_3 = MaxPooling2D(pool_size=(2, 2))(conv1_3)

    conv1_4 = RSU4(pool1_3, 128, 512)
    pool1_4 = MaxPooling2D(pool_size=(2, 2))(conv1_4)

    conv1_5 = RSU4F(pool1_4, 256, 512)

    # second U2_Net
    conv2_1 = RSU7(concatenate([conv1_1, UpSampling2D(size=(2, 2))(conv1_2)]), 32, 64)
    conv2_2 = RSU6(concatenate([conv1_2, MaxPooling2D(pool_size=(2, 2))(conv2_1), UpSampling2D(size=(2, 2))(conv1_3)]), 32, 128)
    conv2_3 = RSU5(concatenate([conv1_3, MaxPooling2D(pool_size=(2, 2))(conv2_2), UpSampling2D(size=(2, 2))(conv1_4)]), 64, 256)
    conv2_4 = RSU4(concatenate([conv1_4, MaxPooling2D(pool_size=(2, 2))(conv2_3), UpSampling2D(size=(2, 2))(conv1_5)]), 128, 256)
    conv2_5 = RSU4F(concatenate([conv1_5, MaxPooling2D(pool_size=(2, 2))(conv2_4)]), 256, 512)

    # ---------------------- Decoder ----------------------
    deconv1_4 = RSU4(concatenate([UpSampling2D(size=(2, 2))(conv1_5), conv1_4], axis=-1), 128, 256)
    deconv1_3 = RSU5(concatenate([UpSampling2D(size=(2, 2))(deconv1_4), conv1_3], axis=-1), 64, 128)
    deconv1_2 = RSU6(concatenate([UpSampling2D(size=(2, 2))(deconv1_3), conv1_2], axis=-1), 32, 64)
    deconv1_1 = RSU7(concatenate([UpSampling2D(size=(2, 2))(deconv1_2), conv1_1], axis=-1), 16, 32)  # 16, 64

    # second U2_Net
    deconv2_4 = RSU4(concatenate([UpSampling2D(size=(2, 2))(conv2_5), conv2_4, deconv1_4]), 128, 256)
    deconv2_3 = RSU5(concatenate([UpSampling2D(size=(2, 2))(deconv2_4), conv2_3, deconv1_3]), 64, 128)
    deconv2_2 = RSU6(concatenate([UpSampling2D(size=(2, 2))(deconv2_3), conv2_2, deconv1_2]), 32, 64)
    deconv2_1 = RSU7(concatenate([UpSampling2D(size=(2, 2))(deconv2_2), conv2_1, deconv1_1]), 16, 32)  # 16, 64

    output = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', padding='same')(deconv2_1)

    model = Model(inputs=inputs, outputs=output, name='Inner_Cascaded_U2_Net')
    return model


if __name__ == '__main__':

    model = Inner_Cascaded_U2_Net(input_shape=(512, 512, 1))
    model.summary()