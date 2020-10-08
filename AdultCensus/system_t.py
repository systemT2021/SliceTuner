from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import curve_fit
from fc import *

import concurrent.futures
import copy

class System_T:
    def __init__(self, train, val, val_data_dict, data_num_array, num_class, num_label, slice_index, add_data_dict):
        """
        Args:
            train: Training data and label
            val: Valiation data and label
            val_data_dict: Validation data per each slice
            data_num_array: Initial slice sizes
            num_class: Number of slices
            num_label: Number of class
            slice_index: index of features (e.g., [0, 3] for White-Male)
            add_data_dict: Data assumed to be collected
        """
        
        self.train = copy.deepcopy(train[0]), copy.deepcopy(train[1])
        self.val = copy.deepcopy(val[0]), copy.deepcopy(val[1])
        self.val_data_dict = copy.deepcopy(val_data_dict)
        self.data_num_array = copy.deepcopy(data_num_array)
        self.add_data_dict = copy.deepcopy(add_data_dict)
        self.num_class = num_class
        self.num_label = num_label
        self.slice_index = slice_index
    
    def selective_collect(self, budget, k, batch_size, lr, epochs, cost_func, Lambda, num_iter, slice_desc, strategy="one-shot", show_figure=False):
        """ 
        Selecitve data collection function to determin how much data collection is needed for each slice 
        given a budget in order to optimzie both accuracy and fairness.
        
        Args: 
            budget: Data collection budget
            k: Number of subsets of data to fit a learning curve
            cost_func: Represents the effort to collect an example for a slice
            Lambda: Balancing term between loss and unfairness
            num_iter: Number of training times for averaging
            strategy: Strategy for updating the limit T (e.g., one-shot, aggressive, linear, conservative)
            show_figure: Plot the learning curve if show_figure is True
            slice_desc: Slice description (e.g., Slice: Shirt)
        """
        
        self.budget = budget
        self.batch_size = batch_size
        self.Lambda = Lambda
        self.cost_func = cost_func
        self.lr = lr
        self.epochs = epochs
        
        
        initial_k = 100
        num_k_ = initial_k + np.arange(0, k) * (len(self.train[0]) - initial_k)/ (k-1)
        num_k = [int(i) for i in num_k_]
        
        iteration = 0
        self.T = 1
        self.train_on_subsets(num_k, num_iter)
        IR = self.get_imbalance_ratio(self.data_num_array)        
        total_num_examples = np.zeros(self.num_class)
        
        while True: 
            num_examples = self.one_shot(slice_desc, show_figure)
            after_IR = self.get_imbalance_ratio(self.data_num_array + num_examples)
            
            # Limit the number of examples acquired by multiplying num_examples with the change ratio
            if strategy != "one-shot" and abs(after_IR - IR) > self.T:
                target_ratio = IR + self.T * np.sign(after_IR - IR)
                change_ratio = self.get_change_ratio(self.data_num_array, num_examples, target_ratio)
                num_examples = np.array([int(num_examples[i] * change_ratio) for i in range(self.num_class)])
                after_IR = self.get_imbalance_ratio(self.data_num_array + num_examples)
                
            self.budget = self.budget - np.sum(np.add(np.multiply(num_examples, self.cost_func), 0.5).astype(int))
            self.data_num_array = np.array([self.data_num_array[i] + num_examples[i] for i in range(self.num_class)])
            self.increase_limit(strategy)
            IR = after_IR
            total_num_examples += num_examples
            iteration += 1
            
            print("======= Collect Data =======")
            print(num_examples)
            print("Total Cost: %s, Remaining Budget: %s" 
                  % (np.sum(np.add(np.multiply(num_examples, self.cost_func), 0.5).astype(int)), self.budget))
            
            # Acquire num_examples data
            self.collect_data(num_examples)
            
            # Repeat until we run out of budget
            if self.budget < 5:
                self.train_after_collect_data(num_examples, num_iter)
                print("\n======= Performance =======")
                print(total_num_examples)
                print("Number of iteration: %d" % iteration)
                print("Strategy: %s, C: %s, Budget: %s" % (strategy, self.Lambda, budget))
                print("Loss: %.5f (%.5f), Average EER: %.5f (%.5f), Max EER: %.5f (%.5f)\n" % tuple(self.show_performance()))
                break
            else:
                num_k_ = initial_k + np.arange(0, k) * (len(self.train[0]) - initial_k)/ (k-1)
                num_k = [int(i) for i in num_k_]
                self.train_on_subsets(num_k, num_iter)
    
    
    
    def train_on_subsets(self, num_k, num_iter):
        """ 
            For each slice, we generate k random subsets of data to fit a power-law curve. 
            Then, we train the models on different subsets in parallel.
            
            Args: 
                num_k: Subsets of the training set with different sizes
                num_iter: Number of training times for averaging
        """
        
        max_workers = len(num_k)
        self.loss_output, self.slice_num = [], []
        
        for i in range(self.num_class):
            self.loss_output.append([0] * len(num_k))
            self.slice_num.append([0] * len(num_k))
            
        for i in range(num_iter):
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                batch_jobs = []
                for k in range(len(num_k)):
                    batch_jobs.append(executor.submit(self.fc_training, num_k, k))
                
                for job in concurrent.futures.as_completed(batch_jobs):
                    if job.cancelled():
                        continue
                        
                    elif job.done():
                        loss_dict, slice_num, process_num = job.result()
                        for j in range(self.num_class):
                            self.loss_output[j][process_num] += (loss_dict[j] / num_iter)
                            if i == 0:
                                self.slice_num[j][process_num] = slice_num[j]    
                                
                                
    def train_after_collect_data(self, num_examples, num_iter):
        """ 
            Trains the model after we run out of budget by num_iter times 
            
            Args:
                num_examples: Number of examples to collect per slice 
                num_iter: Number of training times for averaging
        """
        
        for i in range(self.num_class):
            self.loss_output[i].append(0)
            self.slice_num[i].append(0)

        num_iter = 10
        num_k = [len(self.train[0])] * num_iter
        self.total_loss = []
        self.eer = []
        self.max_eer = []
        max_workers = num_iter
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            batch_jobs = []
            for i in range(num_iter):
                batch_jobs.append(executor.submit(self.fc_training, num_k, i))

            for job in concurrent.futures.as_completed(batch_jobs):
                if job.cancelled():
                    continue

                elif job.done():
                    loss_dict, slice_num, process_num = job.result()
                    self.total_loss.append(np.average(loss_dict))
                    self.measure_eer(loss_dict)
                    for j in range(self.num_class):
                        self.loss_output[j][-1] += (loss_dict[j] / num_iter)
                        if process_num == 0:
                            self.slice_num[j][-1] = slice_num[j]
                    
                    
    def collect_data(self, num_examples):
        """ 
            Collects num_examples of data from add_data_dict
            add_data_dict could be changed to any other data collection tool
                    
            Args:
                num_examples: Number of examples to collect per slice 
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

    
    def fc_training(self, num_k, k):
        """ 
            Trains the model on (num_k[k]) number of training examples.
            
            Args:
                num_k: Subsets of the training set with different sizes.
                k: Training on k'th subset of training data.
                
            Return:
                loss_dict: validation loss per slice.
                slice_num: Number of examples per slice.
                process_num: Process number used for parallel training.
        """
        
        network = FC(self.train[0][:num_k[k]], self.train[1][:num_k[k]], self.val[0], self.val[1], self.val_data_dict, 
                      self.batch_size, epochs=self.epochs, lr = self.lr, num_class = self.num_class, 
                     num_label = self.num_label, slice_index = self.slice_index)
        loss_dict, slice_num, process_num = network.fc_train(k)
        return loss_dict, slice_num, process_num                              
            
                        
    def one_shot(self, slice_desc, show_figure):
        """ 
            Fits the learning curve, and solves the optimization problem using entire budget 

            Args:
                slice_desc: Array for slice description (e.g., Slice: White_Male)
                show_figure: If True, show learning curves for all slices

            Return: Number of examples to collect for the slices within the budget
        """
        
        A, B, estimate_loss = self.fit_learning_curve(slice_desc, show_figure)
        return self.op_func(A, B, self.data_num_array, estimate_loss)
        
        
    def fit_learning_curve(self, slice_desc, show_figure):
        """ 
            Fits the learning curve, assuming power-law form 

            Args:
                slice_desc: Array for slice description (e.g., Slice: White_Male)
                show_figure: If True, show learning curves for all slices

            Returns:
                A: Exponent of power-law equation
                B: of power-law equation
                estimate_loss: estimated loss on given data using power-law equation
        """
        
        def weight_list(weight):
            w_list = []
            for i in weight:
                w_list.append(1/(i**0.5))
            return w_list        

        def power_law(x, a, b):
            return (b*x**(-a))
        
        A, B, estimate_loss = [], [], []
        for i in range(self.num_class):
            xdata = np.linspace(self.slice_num[i][0], self.slice_num[i][-1], 1000)
            sigma = weight_list(self.slice_num[i])    
            popt, pcov = curve_fit(power_law, self.slice_num[i], self.loss_output[i], sigma=sigma, absolute_sigma=True, bounds = (0, [np.inf, np.inf]), maxfev=5000)
            
            A.append(-popt[0])
            B.append(popt[1])
            estimate_loss.append(popt[1] * (self.data_num_array[i] ** (-popt[0])))

            if show_figure == True:
                plt.figure(1, figsize=(12,8))
                plt.plot(self.slice_num[i], self.loss_output[i], 'o-', linewidth=1.0, markersize=4, label=slice_desc[i])
                plt.plot(xdata, power_law(xdata, *popt), linewidth=2.0, label='$y={%0.3f}x^{-{%0.3f}}$' % (popt[1], popt[0]))

                plt.tick_params(labelsize=20)
                plt.xlabel('Number of training examples', fontsize=25)
                plt.ylabel('Validation Loss', fontsize=25)
                plt.legend(prop={'size':25})

                plt.tight_layout()
                plt.show()        
                
        return A, B, estimate_loss
    
    
    def increase_limit(self, strategy):
        """ 
            Updates the limit T according to the strategy
                Aggressive: For each iteration, multiply T by 2
                Moderate: For each iteration, increase T by 1
                Conservative: Leave T as a constant
        """
        
        if strategy == "Aggressive":
            self.T = self.T * 2
        elif strategy == "Moderate":
            self.T = self.T + 1
        else:
            self.T = self.T
        
        
    def get_imbalance_ratio(self, data_array):
        """ 
            Calculate imbalance ratio of the data 

            Args:
                data_array: Number of data for slices

            Return: Imbalance ratio
        """
        
        return max(data_array) / min(data_array)

    
    def get_change_ratio(self, data_array, num_examples, target_ratio):
        """
            If the imbalance ratio change would exceed T when using the entire budget, 
            limit the number of examples collected by the target_ratio by solving a non-linear equation

            Args:
                data_array: Number of data for slices
                num_examples: Number of examples to collect per slice 
                target_ratio: Imbalance ratio limit

            Return: ratio that makes the imbalance ratio change would not exceed target ratio
        """
        
        def F(x, num, add, target):
            func1 = max([int(num_examples[i]*x) + num[i] for i in range(self.num_class)])
            func2 = min([int(num_examples[i]*x) + num[i] for i in range(self.num_class)])
            return func1 - target * func2

        ratio = scipy.optimize.fsolve(F, x0=(0.5), args=(data_array, num_examples, target_ratio))
        if ratio > 1:
            ratio = scipy.optimize.fsolve(F, x0=(0.25), args=(data_array, num_examples, target_ratio))
        elif ratio < 0:
            ratio =  scipy.optimize.fsolve(F, x0=(0.75), args=(data_array, num_examples, target_ratio))
            
        return ratio
    
    
    def op_func(self, A, B, N, estimate_loss):
        """ 
        Solves the convex optimzation problem 
        
        Args:
            A: Exponent of power-law equation
            B: of power-law equation
            estimate_loss: estimated loss on given data using power-law equation
        
        Return: Number of examples to collect for the slices within the budget
        """
        
        try:
            x = cp.Variable(self.num_class)
            for i in range(self.num_class):
                loss = cp.multiply(B[i], cp.power((x[i]+N[i]), A[i]))
                counter_loss = (np.sum(estimate_loss) - estimate_loss[i]) / (self.num_class - 1)

                counter_loss = np.sum(estimate_loss) / self.num_class
                if i==0:
                    ob_func = loss + self.Lambda * cp.maximum(0, (loss / counter_loss) - 1)
                else:
                    ob_func += loss + self.Lambda * cp.maximum(0, (loss / counter_loss) - 1)

            constraints = [cp.sum(cp.multiply(x, self.cost_func)) - self.budget == 0] + [x>=0]
            objective = cp.Minimize(ob_func)
            prob = cp.Problem(objective, constraints)
            prob.solve(solver="ECOS_BB")
        
        except:        
            x = cp.Variable(self.num_class, integer=True)
            for i in range(self.num_class):
                loss = cp.multiply(B[i], cp.power((x[i]+N[i]), A[i]))
                counter_loss = (np.sum(estimate_loss) - estimate_loss[i]) / (self.num_class - 1)

                counter_loss = np.sum(estimate_loss) / self.num_class
                if i==0:
                    ob_func = loss + self.Lambda * cp.maximum(0, (loss / counter_loss) - 1)
                else:
                    ob_func += loss + self.Lambda * cp.maximum(0, (loss / counter_loss) - 1)

            constraints = [cp.sum(cp.multiply(x, self.cost_func)) - self.budget == 0] + [x>=0]
            objective = cp.Minimize(ob_func)
            prob = cp.Problem(objective, constraints)
            prob.solve(solver="ECOS_BB")

        return np.add(x.value, 0.5).astype(int)

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
            final_loss.append(self.loss_output[i][-1])
        
        avg = np.average(final_loss)    
        for i in range(self.num_class):
            diff_eer = abs(final_loss[i] - avg)
            if max_eer < diff_eer:
                max_eer = diff_eer             
                
            avg_eer += diff_eer
            
        avg_eer = avg_eer / self.num_class 
        return np.average(final_loss), np.std(self.total_loss), np.average(self.eer), np.std(self.eer), np.average(self.max_eer), np.std(self.max_eer)
    
