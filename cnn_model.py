from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, Input, Concatenate, AvgPool2D, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from keras import backend
# from numba import jit

class ImageClassificationModel:
#     @jit(nopython=True)
    @staticmethod
    def build(width, height, number_of_classes, final_activation='softmax'):
        input_size = (height, width, 3) 
        if backend.image_data_format() == 'channels_first':
            input_size = (3, height, width)
        
        input_tensor=Input(input_size)
        initial = Conv2D(64, (7,7), strides=2, activation='relu')(input_tensor)
        max_pooling_initial = MaxPooling2D(pool_size=(2,2))(initial)

        batch_1_batchNorm = BatchNormalization()(max_pooling_initial)
        batch_1_activ = Activation('relu')(batch_1_batchNorm)
        batch_1_conv2d_1 = Conv2D(128, (1,1), activation='relu', padding='same')(batch_1_activ)
        batch_1_drop = Dropout(0.3)(batch_1_conv2d_1)
        batch_1_conv2d_2 = Conv2D(32, (3,3), activation='relu', padding='same')(batch_1_drop)

        batch_2 = Concatenate()([max_pooling_initial, batch_1_conv2d_2])

        batch_2_batchNorm = BatchNormalization()(batch_2)
        batch_2_activ = Activation('relu')(batch_2_batchNorm)
        batch_2_conv2d_1 = Conv2D(128, (1,1), activation='relu', padding='same')(batch_2_activ)
        batch_2_drop = Dropout(0.4)(batch_2_conv2d_1)
        batch_2_conv2d_2 = Conv2D(32, (3,3), activation='relu', padding='same')(batch_2_drop)

        batch_3 = Concatenate()([batch_2, batch_2_conv2d_2])

        batch_3_batchNorm = BatchNormalization()(batch_3)
        batch_3_activ = Activation('relu')(batch_3_batchNorm)
        batch_3_conv2d_1 = Conv2D(128, (1,1), activation='relu', padding='same')(batch_3_activ)
        batch_3_drop = Dropout(0.4)(batch_3_conv2d_1)
        batch_3_conv2d_2 = Conv2D(32, (3,3), activation='relu', padding='same')(batch_3_drop)

        batch_4 = Concatenate()([batch_3, batch_3_conv2d_2])

        batch_4_batchNorm = BatchNormalization()(batch_4)
        batch_4_activ = Activation('relu')(batch_4_batchNorm)
        batch_4_conv2d_1 = Conv2D(128, (1,1), activation='relu', padding='same')(batch_4_activ)
        batch_4_drop = Dropout(0.4)(batch_4_conv2d_1)
        batch_4_conv2d_2 = Conv2D(32, (3,3), activation='relu', padding='same')(batch_4_drop)

        final_batch = Concatenate()([batch_4_conv2d_2, batch_4])

        downsampling_batchNorm = BatchNormalization()(final_batch)
        downsampling_activ = Activation('relu')(downsampling_batchNorm)
        downsampling_conv2d_1 = Conv2D(32, (1,1), activation='relu')(downsampling_activ)
        downsampling_avg = AvgPool2D(pool_size=(2,2), strides=2)(downsampling_conv2d_1)

        flatten = Flatten()(downsampling_avg)
        top_layer_dense_1 = Dense(1024, activation='relu')(flatten)
        top_layer_dropout = Dropout(0.4)(top_layer_dense_1)
        top_layer_dense_2 = Dense(number_of_classes, activation=final_activation)(top_layer_dropout)
        model = Model(inputs=input_tensor, outputs=top_layer_dense_2)

        return model