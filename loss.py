
import keras.backend as K


def dice_coef(y_true, y_pred):

    smooth = 1.
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)


def dice_coef_loss(y_true, y_pred):

    return 1.0 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):

    return K.binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred):

    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)

    TP = K.sum(y_true_pos * y_pred_pos)
    FP = K.sum((1-y_true_pos) * y_pred_pos)
    FN = K.sum(y_true_pos * (1-y_pred_pos))

    smooth = 1e-6
    alpha = 0.4
    tversky = (TP + smooth) / (TP + alpha*FP + (1-alpha)*FN + smooth)

    gamma = 0.75
    return K.pow((1-tversky), gamma)


def focal_loss(y_true, y_pred):

    alpha = 0.7
    gamma = 2

    BCE = K.binary_crossentropy(y_true, y_pred)
    BCE_EXP = K.exp(-BCE)

    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
    return focal_loss


def dice_focal_loss(y_true, y_pred):

    alpha = 0.1
    return alpha * focal_loss(y_true, y_pred) + dice_coef_loss(y_true, y_pred)