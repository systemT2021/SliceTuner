from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import copy

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

class CNN:
    def __init__(self, train_data, train_label, val_data, val_label, val_data_dict, batch_size, epochs, lr, num_class):
        """
            Args: 
                train_data: Training data
                train_label: Traning label
                val_data: Validatoin data
                val_label: Validation label
                val_data_dict: Validation data per each slice
                batch_size: batch size for model training
                epochs: number of eopch for model training
                lr: learning rate
                num_class: Number of class
        """
        
        gpu_options = tf.GPUOptions(visible_device_list="2")
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        tf.keras.backend.set_session(self.session)
        self.session.run(tf.global_variables_initializer())
        
        self.train_data = (copy.deepcopy(train_data), copy.deepcopy(train_label))
        self.val_data = (copy.deepcopy(val_data), copy.deepcopy(val_label))
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.val_data_dict = copy.deepcopy(val_data_dict)
        self.num_class = num_class

        self.model = self.build_CNN_classifier()
        
        
    def build_CNN_classifier(self):
        """ 
            Builds a network 
            
            Return: model 
        """
        
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(self.num_class, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(lr=self.lr),
        metrics=['accuracy'])
        
        return model

    def cnn_train(self, process_num):
        """ 
            Trains the network and return the losses of slices 
            
            Return:
                loss_dict: validation loss per slice.
                slice_num: Number of examples per slice.
                process_num: Process number used for parallel training.
        """
        
        loss_dict = []
        slice_num = self.check_num(self.train_data[1])
        
        file_path = "Model_%d.h5" % process_num
        es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        cp = ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                verbose=0, save_best_only=True)
        
        history = self.model.fit(self.train_data[0], self.train_data[1],
                            batch_size=self.batch_size,
                            epochs=self.epochs,
                            verbose=0,
                            validation_data=(self.val_data[0], self.val_data[1]),
                            callbacks=[es, cp])
        
        self.model.load_weights(file_path)
        for i in range(self.num_class):
            loss_dict.append(self.model.evaluate(self.val_data_dict[i][0], self.val_data_dict[i][1], verbose=0)[0])
        
        tf.keras.backend.clear_session()
        return loss_dict, slice_num, process_num
    

    def check_num(self, labels):
        """ 
            Checks the number of data per each slice 
            
            Args:
                labels: Array that contains only label
                
            Return:
                slice_num: Number of examples per slice.
        """        
        slice_num = dict()
        for j in range(self.num_class):
            idx = np.argmax(labels, axis=1) == j
            slice_num[j] = len(labels[idx])
            
        return slice_num
    