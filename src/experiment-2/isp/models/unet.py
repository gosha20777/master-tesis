from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Convolution2D, MaxPooling2D, UpSampling2D


def get_model(input_shape): 
    inputs = Input(shape=input_shape)
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Convolution2D(512, (3, 3), activation='relu', padding='same')(conv5)
    
    up6 = Concatenate()([Convolution2D(256, (2, 2), activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5)), conv4])
    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Convolution2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Concatenate()([Convolution2D(128, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6)), conv3])
    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Convolution2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Concatenate()([Convolution2D(64, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7)), conv2])
    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Convolution2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Concatenate()([Convolution2D(32, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8)), conv1])
    conv9 = Convolution2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Convolution2D(32, (3, 3), activation='relu', padding='same')(conv9)
    
    up10 = Convolution2D(16, (2, 2),activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv9))
    conv10 = Convolution2D(16, (3, 3), activation='relu', padding='same')(up10)
    conv10 = Convolution2D(16, (3, 3), activation='relu', padding='same')(conv10)

    x_out = Convolution2D(3, (1, 1), activation='sigmoid')(conv10)
    outputs = {
        'mae_out': x_out,
        'mssim_out': x_out,
        'color_out': x_out,
        'exp_out': x_out,
        'vgg_out': x_out,
    }
    return Model(inputs=inputs, outputs=outputs)