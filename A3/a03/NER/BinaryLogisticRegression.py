from __future__ import print_function
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye, Patrik Jonell and Dmytro Kalpakchi.
"""

class BinaryLogisticRegression(object):
    """
    This class performs binary logistic regression using batch gradient descent
    or stochastic gradient descent
    """

    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.01  # The learning rate.
    CONVERGENCE_MARGIN = 0.001  # The convergence criterion.
    MAX_ITERATIONS = 100 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 10000 # Minibatch size (only for minibatch gradient descent)
    PATIENCE = 500

    # ----------------------------------------------------------------------


    def __init__(self, x=None, y=None, theta=None):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        if not any([x, y, theta]) or all([x, y, theta]):
            raise Exception('You have to either give x and y or theta')

        if theta:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)

            self.LEARNING_RATE = 0.01

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------

    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + np.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """
        # conditional probability
        if label == 1:
            return self.sigmoid(np.matmul(self.theta, self.x[datapoint]))

        return 1 - self.sigmoid(np.matmul(self.theta, self.x[datapoint]))

    def compute_gradient_for_all(self):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """
        for feature in range(self.FEATURES):
            new_gradient = 0

            for datapoint in range(self.DATAPOINTS):
                new_gradient += self.compute_feat_gradient(datapoint, feature)
            self.gradient[feature] = new_gradient / self.DATAPOINTS

            # self.theta[feature] -= self.LEARNING_RATE * self.gradient[feature]


    def compute_feat_gradient(self, datapoint, feature):
        return self.x[datapoint][feature] * (
                self.sigmoid(np.matmul(self.theta, self.x[datapoint])) - self.y[datapoint])


    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on a minibatch
        (used for minibatch gradient descent).
        """
        
        start_point = (minibatch-1) * self.MINIBATCH_SIZE
        end_point = minibatch * self.MINIBATCH_SIZE

        for feature in range(self.FEATURES):
            new_gradient = 0
            for datapoint in range(start_point, end_point):
                new_gradient += self.compute_feat_gradient(datapoint, feature)
            self.gradient[feature] = new_gradient / self.MINIBATCH_SIZE


    def compute_gradient(self, datapoint):
        """
        Computes the gradient based on a single datapoint
        (used for stochastic gradient descent).
        """

        for feature in range(self.FEATURES):
            self.gradient[feature] = self.compute_feat_gradient(datapoint, feature)

    def stochastic_fit_with_early_stopping(self):
        """
        Performs Stochastic Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        epochs = 10
        for epoch in tqdm(range(epochs)):
            print('Epoch: ', epoch)
            train_itr = 0
            while train_itr < 10000:
                datapoint = np.random.randint(0, self.DATAPOINTS)
                self.compute_gradient(datapoint)
                self.upd_theta()
                # plot every 100th iteration
                if not (train_itr % 100):
                    self.update_plot(np.sum(np.square(self.gradient)))
                train_itr += 1
            

    def minibatch_fit_with_early_stopping(self):
        """
        Performs Mini-batch Gradient Descent.
        """
        self.init_plot(self.FEATURES)

        max_batch = self.DATAPOINTS // self.MINIBATCH_SIZE

        epochs = 10
        epoch = 0
        while epoch < epochs:
            print('Epoch: ', epoch)
            batch_nr = 1
            while batch_nr < max_batch+1:
                if batch_nr % 10 == 0:
                    print('Batch nr: ', batch_nr)
                self.compute_gradient_minibatch(batch_nr)
                self.upd_theta()
                self.update_plot(np.sum(np.square(self.gradient)))
                batch_nr += 1

            epoch += 1


    def fit(self):
        """
        Performs Batch Gradient Descent
        """
        self.init_plot(self.FEATURES)

        '''
        while np.sum(np.square(self.gradient)) > self.CONVERGENCE_MARGIN:
            self.compute_gradient_for_all()
            self.upd_theta()
            self.update_plot(np.sum(np.square(self.gradient)))
        '''

        train_itr = 0
        while  train_itr < 50:
            print('train_nr ', train_itr)
            self.compute_gradient_for_all()
            self.upd_theta()
            self.update_plot(np.sum(np.square(self.gradient)))
            train_itr += 1


    def upd_theta(self):
        for k in range(self.FEATURES):
            self.theta[k] -= self.LEARNING_RATE * self.gradient[k]
        


    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        print('Model parameters:');

        print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))

        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))


    # ----------------------------------------------------------------------

    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)

    # ----------------------------------------------------------------------


def main():
    """
    Tests the code on a toy example.
    """
    x = [
        [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ], [ 0,0 ], [ 0,0 ],
        [ 0,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 0,0 ], [ 1,0 ],
        [ 1,0 ], [ 0,0 ], [ 1,1 ], [ 0,0 ], [ 1,0 ], [ 0,0 ]
    ]

    #  Encoding of the correct classes for the training material
    y = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
    b = BinaryLogisticRegression(x, y)
    b.fit()
    b.print_result()


if __name__ == '__main__':
    main()
