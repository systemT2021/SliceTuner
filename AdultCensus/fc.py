from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import copy

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

class FC:
    def __init__(self, train_data, train_label, val_data, val_label, val_data_dict, batch_size, epochs, lr, num_class, num_label, slice_index):
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
                num_class: Number of slices
                num_label: Number of class
                slice_index: index of features (e.g., [0, 3] for White-Male)
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
        self.num_label = num_label
        self.slice_index = slice_index

        self.model = self.build_FC_classifier()
        
        
    def build_FC_classifier(self):
        """ 
            Builds a network 
            
            Return: model 
        """
        
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=(62,)))
        model.add(layers.Dense(self.num_label, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(lr=self.lr),
        metrics=['accuracy'])
        
        return model

    def fc_train(self, process_num):
        """ 
            Trains the network and return the losses of slices 
            
            Return:
                loss_dict: validation loss per slice.
                slice_num: Number of examples per slice.
                process_num: Process number used for parallel training.
        """
        
        loss_dict = []
        slice_num = self.check_num()
        
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
    

    def check_num(self):
        """ 
            Checks the number of data per each slice 
            
            Args:
                labels: Array that contains only label
                
            Return:
                slice_num: Number of examples per slice.
        """        
        slice_num = dict()
        for i in range(self.num_class):
            slice_num[i] = len(self.train_data[0][(self.train_data[0][:, self.slice_index[i][0]] == 1) & (self.train_data[0][:, self.slice_index[i][1]] == 1)])
        return slice_num
    
