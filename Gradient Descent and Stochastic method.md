### Gradient descnet In laymen's term 
To draw an analogy, imagine a pit in the shape of U and you are standing at the top most of point in the pit and your objective is to reach the bottom of the pit. There is a catch, you can only take discrete number of steps to reach the bottom. If you decide to take one step at a time you would eventually reach the bottom of the pit but this would take a longer time.

If you choose to jump over steps and decide to cover more than one step at a time you would reach sooner but, there is a chance that you could overshoot the bottom of the pit and end up at the other side of the pit or only closer to the bottom of the pit and not exactly at the bottom.

In the gradient descent algorithm, the number of steps you take is the learning rate. This decides on how fast the algorithm converges to the minima. The smaller the learning rate, the more closer you are to the minima but the more time it takes to reach the minima, a larger learning rate converges sooner but there is a chance that you could end up in a value slightly far from the minima.


## Three variants of gradient descent algorithm

* **Batch gradient descent (BGD):** calculate the error for each example in the training dataset, but only updates the model after all training examples have been evaluated.
* **Stochastic gradient descent (SGD):** calculate the error and updates the model for each examplein the training dataset.
* **Mini-Batch gradient descent:** split the training dataset into small batches that are used to calculate model error and updated model coefficients. (the most common implementation of gradient descent used in the field of deep learning)

> Mini-Batch gradient descent can find a balance between the robustness of SGD and the efficiency of BGD.


* Epochs: One Epoch is when an ENTIRE dataset is passed to the model 
* Batchs: Divide one epoch in several smaller bacthes 
* Batch Size: Total number of training samples present in a single batch 
* Iterations/ stpes per epoch
  *  Num of batches needed to complete one epoch 
  *  = epoch size / batch size
  *  ```python steps_per_epoch = int( np.ceil(x_train.shape[0] / batch_size) ) ```


## Code for finding the optimal learning rate 

```python

import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import Callback


class LRFinder(Callback):
    
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    '''
    
    def __init__(self, min_lr=1e-5, max_lr=1e-2, steps_per_epoch=None, epochs=None):
        super().__init__()
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        
    def clr(self):
        '''Calculate the learning rate.'''
        x = self.iteration / self.total_iterations 
        return self.min_lr + (self.max_lr-self.min_lr) * x
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        K.set_value(self.model.optimizer.lr, self.clr())
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        plt.show()
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')
        plt.show()
        
 ```
 
 [code_reference](https://www.jeremyjordan.me/nn-learning-rate/)
