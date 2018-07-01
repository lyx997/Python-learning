# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:32:46 2018

@author: 123
"""
import tensorflow as tf
import numpy as np
x=np.loadtxt('f1.txt')
x_image = tf.reshape(x, [240,6,-1,1])
print(x_image)