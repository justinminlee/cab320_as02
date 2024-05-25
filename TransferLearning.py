# -*- coding: utf-8 -*-
'''

The functions and classes defined in this module will be called by a marker script. 
You should complete the functions and classes according to their specified interfaces.

No partial marks will be awarded for functions that do not meet the specifications
of the interfaces.

Last modified 2024-05-07 by Anthony Vanderkop.
Hopefully without introducing new bugs.
'''


### LIBRARY IMPORTS HERE ###
import os
import numpy as np
import keras.applications as ka
import keras
import cv2
import math
import matplotlib.pyplot as plt

import seaborn as sns
    
def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(11198885, "Minjae", "Lee"), (10804072, 'Zaichic', 'Turner'), (11225271, 'Chanyoung', 'Kim')]
    
def load_model():
    '''
    Load in a model using the tf.keras.applications model and return it.
    Insert a more detailed description here
    '''
    return ka.MobileNetV2(input_shape=(128,128,3), include_top=False)
    

def load_data(path):
    '''
    Load in the dataset from its home path. Path should be a string of the path
    to the home directory the dataset is found in. Should return a numpy array
    with paired images and class labels.
    
    Insert a more detailed description here.
    '''
    images = []
    directories = [ f.path for f in os.scandir('./small_flower_dataset') ]
    for directory in directories:
        images.extend([ (directory.split('/')[2], cv2.resize(cv2.imread(f.path), (128,128))) for f in os.scandir(directory) ])
    images = np.array(images, dtype=object)
    return images
    
    
    
def split_data(X, Y, train_fraction, randomize=False, eval_set=True):
    """
    Split the data into training and testing sets. If eval_set is True, also create
    an evaluation dataset. There should be two outputs if eval_set there should
    be three outputs (train, test, eval), otherwise two outputs (train, test).
    
    To see what type train, test, and eval should be, refer to the inputs of 
    transfer_learning().
    
    Insert a more detailed description here.
    """
    raise NotImplementedError
    
    

def confusion_matrix(predictions, ground_truth, plot=False, all_classes=None):
    '''
    Given a set of classifier predictions and the ground truth, calculate and
    return the confusion matrix of the classifier's performance.

    Inputs:
        - predictions: np.ndarray of length n where n is the number of data
                       points in the dataset being classified and each value
                       is the class predicted by the classifier
        - ground_truth: np.ndarray of length n where each value is the correct
                        value of the class predicted by the classifier
        - plot: boolean. If true, create a plot of the confusion matrix with
                either matplotlib or with sklearn.
        - classes: a set of all unique classes that are expected in the dataset.
                   If None is provided we assume all relevant classes are in 
                   the ground_truth instead.
    Outputs:
        - cm: type np.ndarray of shape (c,c) where c is the number of unique  
              classes in the ground_truth
              
              Each row corresponds to a unique class in the ground truth and
              each column to a prediction of a unique class by a classifier
    '''
    
    # If all_classes is not provided, determine the unique classes from the ground_truth
    if all_classes is None:
        all_classes = np.unique(ground_truth)

    # Get the number of unique classes
    num_classes = len(all_classes)

    # Create a dictionary mapping each class to its index
    class_to_index = {cls: idx for idx, cls in enumerate(all_classes)}

    # Initialize the confusion matrix with zeros
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # Iterate through each pair of ground truth and prediction
    for gt, pred in zip(ground_truth, predictions):
        # Increment the corresponding cell in the confusion matrix
        cm[class_to_index[gt], class_to_index[pred]] += 1

    # If plot is True, create a heatmap plot of the confusion matrix
    if plot:
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_classes, yticklabels=all_classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    
    return cm
    

def precision(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's precision
    
    Inputs: see confusion_matrix above
    Outputs:
        - precision: type np.ndarray of length c,
                     values are the precision for each class
    '''
    
    # Calculate the confusion matrix using the predictions and ground truth
    cm = confusion_matrix(predictions, ground_truth)

    # Calculate precision for each class by dividing the diagonal of the confusion matrix by the sum of each column
    precision = np.diag(cm) / np.sum(cm, axis=0)

    return precision

def recall(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's recall
    
    Inputs: see confusion_matrix above
    Outputs:
        - recall: type np.ndarray of length c,
                     values are the recall for each class
    '''
    
    # Calculate the confusion matrix using the predictions and ground truth
    cm = confusion_matrix(predictions, ground_truth)

    # Calculate recall for each class by dividing the diagonal of the confusion matrix by the sum of each row
    recall = np.diag(cm) / np.sum(cm, axis=1)

    return recall

def f1(predictions, ground_truth):
    '''
    Similar to the confusion matrix, now calculate the classifier's f1 score
    Inputs:
        - see confusion_matrix above for predictions, ground_truth
    Outputs:
        - f1: type nd.ndarry of length c where c is the number of classes
    '''
    
    # Calculate precision and recall using the precision and recall functions
    prec = precision(predictions, ground_truth)
    rec = recall(predictions, ground_truth)

    # Calculate the F1 score using the formula: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (prec * rec) / (prec + rec)

    return f1

def k_fold_validation(features, ground_truth, classifier, k=2):
    '''
    Inputs:
        - features: np.ndarray of features in the dataset
        - ground_truth: np.ndarray of class values associated with the features
        - fit_func: f
        - classifier: class object with both fit() and predict() methods which
        can be applied to subsets of the features and ground_truth inputs.
        - predict_func: function, calling predict_func(features) should return
        a numpy array of class predictions which can in turn be input to the 
        functions in this script to calculate performance metrics.
        - k: int, number of sub-sets to partition the data into. default is k=2
    Outputs:
        - avg_metrics: np.ndarray of shape (3, c) where c is the number of classes.
        The first row is the average precision for each class over the k
        validation steps. Second row is recall and third row is f1 score.
        - sigma_metrics: np.ndarray, each value is the standard deviation of 
        the performance metrics [precision, recall, f1_score]
    '''
    
    #split data
    ### YOUR CODE HERE ###
    
    # Initialize metrics storage
    all_precisions = []
    all_recalls = []
    all_f1s = []

    # Generate an array of indices and shuffle them
    indices = np.arange(features.shape[0])
    np.random.shuffle(indices)

    # Split the shuffled indices into k roughly equal-sized folds
    folds = np.array_split(indices, k)

    #go through each partition and use it as a test set.
    for partition_no in range(k):
        #determine test and train sets
        ### YOUR CODE HERE###

        # Determine test indices and training indices for the current fold
        test_indices = folds[partition_no]
        train_indices = np.concatenate(folds[:partition_no] + folds[partition_no + 1:])

        # Split the features and ground truth into training and test sets based on the indices
        train_features, test_features = features[train_indices], features[test_indices]
        train_classes, test_classes = ground_truth[train_indices], ground_truth[test_indices]

        #fit model to training data and perform predictions on the test set
        classifier.fit(train_features, train_classes)
        predictions = classifier.predict(test_features)
        
        #calculate performance metrics
        ### YOUR CODE HERE###

        # Calculate precision, recall, and f1 scores for the current fold
        prec = precision(predictions, test_classes)
        rec = recall(predictions, test_classes)
        f1_scores = f1(predictions, test_classes)

        # Append the metrics for the current fold to the lists
        all_precisions.append(prec)
        all_recalls.append(rec)
        all_f1s.append(f1_scores)
    
    #perform statistical analyses on metrics
    ### YOUR CODE HERE###

    # Compute the average and standard deviation of the metrics across all folds
    avg_precision = np.mean(all_precisions, axis=0)
    avg_recall = np.mean(all_recalls, axis=0)
    avg_f1 = np.mean(all_f1s, axis=0)

    std_precision = np.std(all_precisions, axis=0)
    std_recall = np.std(all_recalls, axis=0)
    std_f1 = np.std(all_f1s, axis=0)

    # Combine the average metrics into a single array
    avg_metrics = np.array([avg_precision, avg_recall, avg_f1])

    # Combine the standard deviation metrics into a single array
    sigma_metrics = np.array([std_precision, std_recall, std_f1])

    return avg_metrics, sigma_metrics


##################### MAIN ASSIGNMENT CODE FROM HERE ######################

def transfer_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform standard transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)


    Outputs:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the 
            model on the test_set (list of np.ndarray)

    '''
    raise NotImplementedError
    return model, metrics
    
def accelerated_learning(train_set, eval_set, test_set, model, parameters):
    '''
    Implement and perform accelerated transfer learning here.

    Inputs:
        - train_set: list or tuple of the training images and labels in the
            form (images, labels) for training the classifier
        - eval_set: list or tuple of the images and labels used in evaluating
            the model during training, in the form (images, labels)
        - test_set: list or tuple of the training images and labels in the
            form (images, labels) for testing the classifier after training
        - model: an instance of tf.keras.applications.MobileNetV2
        - parameters: list or tuple of parameters to use during training:
            (learning_rate, momentum, nesterov)


    Outputs:
        - model : an instance of tf.keras.applications.MobileNetV2
        - metrics : list of classwise recall, precision, and f1 scores of the 
            model on the test_set (list of np.ndarray)

    '''
    raise NotImplementedError
    return model, metrics

if __name__ == "__main__":
    
    model = load_model()
    dataset = load_data('./small_flower_dataset')
    train_eval_test = split_data()
    
    model, metrics = transfer_learning()
    
    model, metrics = accelerated_learning()
    
    
#########################  CODE GRAVEYARD  #############################
