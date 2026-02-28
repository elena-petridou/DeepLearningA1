"""
Helper functions to the 'Tell-the-Time' model
"""

import numpy as np
from numpy.typing import ArrayLike
import tensorflow as tf

def time_to_angle(x: ArrayLike) -> np.ndarray:
    """
    Function to transform hours and minutes into two angles in degrees
    
    Args:
        x (ArrayLike): An array of hours and minutes of shape (x, 2)
    
    Returns:
        x_angle (np.ndarray): The array of angles corresponding to the hands positions
    """
    x = np.asarray(x)
    hour_angles = (x[:, 0] % 12) * 30 + x[:, 1] * 0.5
    minute_angles = x[:, 1] * 6

    return np.column_stack((hour_angles, minute_angles))

def angle_to_time(x: ArrayLike) -> np.ndarray:
    """
    Function to transform two angles in degrees into hours and minutes
    
    Args:
        x (ArrayLike): An array of angles of shape (x, 2)
    
    Returns:
        x_time (np.ndarray): The array of hours and minutes corresponding to the angles
    """
    x = np.asarray(x)
    hours = (x[:, 0] // 30) % 12
    minutes = (x[:, 1] // 6) % 60

    return np.column_stack((hours, minutes))

def time_to_float(x: ArrayLike) -> np.ndarray:
    """
    Function to transform hours and minutes into a float with an hour fraction
    and normalise them on a scale 0-1.
    
    Args:
        x (ArrayLike): An array of hours and minutes of shape (x, 2)
    
    Returns:
        x_float (np.ndarray): A one-dimensional array of floats
    """
    x = np.asarray(x)
    numbers = x[:, 0] % 12 + x[:,1]/60

    return numbers / 12.0

def circular_mae(y_true: tf.Tensor,
                y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes a circular Mean Abolute Error, meaning it will give a correct
    accuracy given the fact that a clock is round.
    Set on a scale 0-1.

    Args:
        y_true (tf.Tensor): Validation y for the model
        y_pred (tf.Tensor): Predicted values

    Returns:
        reduced_tensor (tf.Tensor): Tensor of mean circular MAEs  
    """
    y_true_total = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred_total = tf.cast(tf.squeeze(y_pred), tf.float32)
    diff = tf.abs(tf.subtract(y_true_total, y_pred_total))
    diff_wrapped = tf.minimum(diff, 1.0 - diff)

    return tf.reduce_mean(diff_wrapped)

def circular_mae_hours(y_true: tf.Tensor,
                y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes a circular Mean Abolute Error, meaning it will give a correct
    accuracy given the fact that a clock is round.
    Set on a scale 0-12.

    Args:
        y_true (tf.Tensor): Validation y for the model
        y_pred (tf.Tensor): Predicted values

    Returns:
        reduced_tensor (tf.Tensor): Tensor of mean circular MAEs  
    """
    y_true_total = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred_total = tf.cast(tf.squeeze(y_pred), tf.float32)
    diff = tf.abs(tf.subtract(y_true_total, y_pred_total))
    diff_wrapped = tf.minimum(diff, 12.0 - diff)

    return tf.reduce_mean(diff_wrapped)

def time_to_float_old(x: ArrayLike) -> np.ndarray:
    """
    Function to transform hours and minutes into a float with an hour fraction

    Args:
        x (ArrayLike): An array of hours and minutes of shape (x, 2)

    Returns:
        x_float (np.ndarray): A one-dimensional array of floats
    """
    x = np.asarray(x)
    numbers = x[:, 0] % 12 + x[:,1]/60

    return numbers /12.0

def circular_mae_old(y_true: tf.Tensor,
                     y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes a circular Mean Abolute Error, meaning it will give a correct
    accuracy given the fact that a clock is round.

    Args:
        y_true (tf.Tensor): Validation y for the model
        y_pred (tf.Tensor): Predicted values

    Returns:
        reduced_tensor (tf.Tensor): Tensor of mean circular MAEs
    """
    y_true_total = tf.cast(tf.squeeze(y_true), tf.float32)
    y_pred_total = tf.cast(tf.squeeze(y_pred), tf.float32)
    diff = tf.abs(tf.subtract(y_true_total, y_pred_total))
    diff_wrapped = tf.minimum(diff, 1.0 - diff)

    return tf.reduce_mean(diff_wrapped)



def circular_mae_two_head(model, X, y_true):
    """
    Post-hoc application of the circular MAE to the predictions of the model

    Args: 
        model (keras.Model()): A keras model obtained via the compile method
        X (np.array): The inputs for the model (in our task the images' pixels)
        y_true (np.array): 2-dimensional numpy array of labels with the hours and minutes

    """
    # Acquire predictions for hour from the model
    predicted_probs = model.predict(X)
    predicted_hour = np.argmax(predicted_probs[0], axis=1)

    # Acquire predictions for minute from model and turn standardised minutes back into minutes
    predicted_minute = np.round(predicted_probs[1].flatten())

    full_prediction = np.column_stack((predicted_hour, predicted_minute))

    # Turn time into standardised float
    predicted_float = time_to_float(full_prediction)
    actual_float = time_to_float(y_true)

    # Calculate circular MAE
    abs_diff = np.abs(actual_float - predicted_float)
    circular_diff = np.minimum(abs_diff, 1 - abs_diff)
    circular_mae_value = np.mean(circular_diff)

    return(circular_mae_value)

