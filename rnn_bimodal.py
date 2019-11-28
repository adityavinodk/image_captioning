from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, Input, Concatenate, AvgPool2D, GlobalAveragePooling2D, Embedding, LSTM
from tensorflow.keras.models import Sequential, Model
from tensorflow.math import add
from keras import backend
import numpy as np

class RNNBimodal:
    @staticmethod
    def build(vocab_size, max_length):
        # feature extractor model
        inputs1 = Input(shape=(80,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        
        # sequence model
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        
        # decoder model
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
       
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        return model