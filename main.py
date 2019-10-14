# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:22:14 2019

@author: bottersb
"""

import tensorflow as tf
session = tf.Session()
tf_s= tf.constant(u"Initial commit!")
print(session.run(tf_s))