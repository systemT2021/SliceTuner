from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import concurrent.futures
import cvxpy as cp
import numpy as np
from cnn import *

class Baseline:
    def __init__(self, train, val, val_data_dict, data_num_array, num_class, add_data_dict, method):
        """
        Args:
            train: Training data and label
            val: Valiation data and label
            val_data_dict: Validation data per each slice
            data_num_array: Initial slice sizes
            num_class: Number of class
            add_data_dict: Data assumed to be collected
            
            method: Choose the baseline method
                Uniform: Collects similar amounts of data per slic
                Waterfilling: Collects data such that the slices end up having similar amounts of data
        """
        
        self.train = copy.deepcopy(train[0]), copy.deepcopy(train[1])
        self.val = copy.deepcopy(val[0]), copy.deepcopy(val[1])
        self.val_data_dict = copy.deepcopy(val_data_dict)
        self.data_num_array = copy.deepcopy(data_num_array)
        self.add_data_dict = copy.deepcopy(add_data_dict)
        self.num_class = num_class
        self.method = method
        
    
    def performance(self, budget, cost_func, num_iter, batch_size, lr, epochs):
        """ 
        Args: 
            budget: Data collection budget
            cost_func: Represents the effort to collect an example for a slice
            num_iter: Number of training times
        """
        
        self.budget = budget
        self.cost_func = cost_func
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.loss_output = [0] * self.num_class
        self.eer = []
        self.max_eer = []
        self.total_loss = []
        
        if self.method == 'Uniform':
            num_examples = (np.ones(self.num_class)*(self.budget/np.sum(self.cost_func))).astype(int)
        elif self.method == 'Waterfilling':
            num_examples = np.add(self.waterfill(), 0.5).astype(int)
        self.train_after_collect_data(num_examples, num_iter)
        
        print("Method: %s, Budget: %s" % (self.method, budget))
        print("======= Collect Data =======")
        print(num_examples)
        print("======= Performance =======")
        print("Loss: %.5f (%.5f), Average EER: %.5f (%.5f), Max EER: %.5f (%.5f)\n" % tuple(self.show_performance()))
    
    
    def waterfill(self):
        """ 
            Return: Number of examples by Water filling algorithm
        """
        output = np.array(self.data_num_array.copy())
        while self.budget > 0:
            index = np.argmin(output)
            output[index] += 1
            self.budget -= self.cost_func[index]
        
        return output - self.data_num_array     
                    
                    
    def cnn_training(self, process_num):
        """ 
            Trains the model on training data.
            
            Args:
                process_num: Process number used for parallel training.
                
            Return:
                loss_dict: validation loss per slice.
                slice_num: Number of examples per slice.
                process_num: Process number used for parallel training.
        """
        
        network = CNN(self.train[0], self.train[1], self.val[0], self.val[1], self.val_data_dict, 
                      self.batch_size, epochs=self.epochs, lr = self.lr, num_class = self.num_class)
        loss_dict, slice_num, process_num = network.cnn_train(process_num)
        return loss_dict, slice_num, process_num 
    
    
    def train_after_collect_data(self, num_examples, num_iter):
        """ 
            Trains the model after we collect num_examples of data
        
            Args:
                num_examples: Number of examples to collect per slice 
                num_iter: Number of training times
        """     

        max_workers = num_iter
        self.collect_data(num_examples)
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            batch_jobs = []
            for i in range(num_iter):
                batch_jobs.append(executor.submit(self.cnn_training, i))

            for job in concurrent.futures.as_completed(batch_jobs):
                if job.cancelled():
                    continue
                    
                elif job.done():
                    loss_dict, slice_num, process_num = job.result()
                    self.total_loss.append(np.average(loss_dict))
                    self.measure_eer(loss_dict)
                    for j in range(self.num_class):
                        self.loss_output[j] += (loss_dict[j] / num_iter)
    
    
    def collect_data(self, num_examples):
        """ 
            Collects num_examples of data from add_data_dict
            add_data_dict could be changed to any other data collection tool
        """
        
        def shuffle(data, label):
            shuffle = np.arange(len(data))
            np.random.shuffle(shuffle)
            data = data[shuffle]
            label = label[shuffle]
            return data, label
        
        train_data = self.train[0]
        train_label = self.train[1]
        for i in range(self.num_class):
            train_data = np.concatenate((train_data, self.add_data_dict[i][0][:num_examples[i]]), axis=0)
            train_label = np.concatenate((train_label, self.add_data_dict[i][1][:num_examples[i]]), axis=0)      
            self.add_data_dict[i]= self.add_data_dict[i][0][num_examples[i]:], self.add_data_dict[i][1][num_examples[i]:]
        
        train_data, train_label = shuffle(train_data, train_label)
        self.train = (train_data, train_label)
    
    
    def measure_eer(self, loss_dict):
        
        max_eer, avg_eer = 0, 0
        avg = np.average(loss_dict)
        for i in range(self.num_class):
            diff_eer = abs(loss_dict[i] - avg)
            if max_eer < diff_eer:
                max_eer = diff_eer             
                
            avg_eer += diff_eer
        avg_eer = avg_eer / self.num_class
        self.eer.append(avg_eer)
        self.max_eer.append(max_eer)
    
    def show_performance(self):
        """ Average validation loss, average equalized error rate(Avg. EER), maximum equalized error rate (Max. EER) """

        final_loss, max_eer, avg_eer = [], 0, 0
        for i in range(self.num_class):
            final_loss.append(self.loss_output[i])
        
        avg = np.average(final_loss)    
        for i in range(self.num_class):
            diff_eer = abs(final_loss[i] - avg)
            if max_eer < diff_eer:
                max_eer = diff_eer             
                
            avg_eer += diff_eer
            
        avg_eer = avg_eer / self.num_class 
        return np.average(final_loss), np.std(self.total_loss), np.average(self.eer), np.std(self.eer), np.average(self.max_eer), np.std(self.max_eer)
        #return np.average(final_loss), np.std(self.total_loss), avg_eer, max_eer 
    