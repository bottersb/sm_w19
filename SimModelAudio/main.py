#%%
from collections import Counter
import glob


import numpy as np
import random
import librosa
import sklearn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import keras
from keras.layers import LSTM, Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPooling1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from keras.initializers import glorot_normal
import matplotlib.pyplot as plt


from keras.models import Sequential
from sklearn.utils import resample, shuffle


#%%
DATA_DIR1 = 'data/kermit/'      # A dataset containing a refined set of very pure samples
# of Kermits voice without background noize etc.
#DATA_DIR1 = 'data/kermit_big/' # The full dataset of all audio files initially labelled as
# "Kermit present"
DATA_DIR2 = 'data/not_kermit/'
SEED = 123

n_features = 7
max_length = 4
n_classes = 1 # since its 2 --> only 1
# filePaths = glob.glob(DATA_DIR + "*.wav")
# random.seed(123)
# random.shuffle(filePaths)


filesKermit = glob.glob(DATA_DIR1 + "*.wav")
filesNotKermit = glob.glob(DATA_DIR2 + "*.wav")


# Upsample minority class
# filesKermit = resample(filesKermit, replace=True, n_samples=10000,
#                                  random_state=SEED)
#filesNotKermit = resample(filesNotKermit, replace=True, n_samples=5000,
#                                 random_state=SEED)

files = filesNotKermit + filesKermit



labels = []
for file in files:
    label_indicator = file.split('_', )[-2]
    if label_indicator == 'not':
        label = 0
    elif str.__contains__(label_indicator, 'split'):#For /kermit/ files
        label = 1
    elif label_indicator.isnumeric(): # For /kermit_big/ files
        label = 1
    else:
        raise ValueError("Seems like there is a problem with the naming convention!!")
    labels.append(label)

random.seed(a=1)
np.random.seed(2)

def load_files(files, labels):
    X=[]
    Y=[]
    keys=files.keys()
    for i,key in enumerate(keys):
        aud,s=librosa.load(key)
        mfcc=get_features(aud,s)
        X.append(mfcc)
        Y.append(labels[i])
    return X,Y

def get_features(y, sr):
    y = y[0:80000]  # analyze just 80k

    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_features)
    mfcc = np.pad(mfcc, ((0, 0), (0, max_length - len(mfcc[0]))), mode='constant', constant_values=0)
    # delta_mfcc = librosa.feature.delta(mfcc, width=5)
    # delta2_mfcc = librosa.feature.delta(mfcc,  width=5, order=2)

    feature_vector = mfcc
    return feature_vector.flatten()

filesDict = {}
for i, file in enumerate(files):
    filesDict[file] = labels[i]

Xl,Yl=load_files(filesDict, labels)
Yl = np.array(Yl)



X_train, X_test, y_train, y_test = train_test_split(Xl, Yl, test_size=0.2, random_state=SEED)


sm = SMOTE(random_state = 42)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train)


np.savetxt("data/prep_trainX_small.csv", X_train_new, delimiter=",")
np.savetxt("data/prep_trainY_small.csv", y_train_new, delimiter=",")
np.savetxt("data/prep_testX_small.csv", X_test, delimiter=",")
np.savetxt("data/prep_testY_small.csv", y_test, delimiter=",")


#%%
X_train_loaded = np.loadtxt("data/prep_trainX_small.csv",  delimiter=",")
y_train_loaded = np.loadtxt("data/prep_trainY_small.csv",  delimiter=",")
X_test_loaded = np.loadtxt("data/prep_testX_small.csv", delimiter=",")
y_test_loaded = np.loadtxt("data/prep_testY_small.csv", delimiter=",")

# clf = RandomForestClassifier(n_jobs=8, random_state=42, n_estimators=200, oob_score=True, bootstrap=True,
#                              class_weight='balanced')

# clf.fit(X_train, y_train)
# clf.fit(X_train_new, y_train_new)

n_estimators = [int(x) for x in np.linspace(start = 300, stop = 600, num = 2)]
max_features = ['auto', 'sqrt']
#max_depth = [int(x) for x in np.linspace(10, 110, num = 3)]
max_depth = [50]
max_depth.append(None)

#min_samples_split = [5, 9]
min_samples_leaf = [2, 4]
bootstrap = [True]
random_grid = {'n_estimators': n_estimators,
               #'max_features': max_features,
               'max_depth': max_depth,
               #'min_samples_split': min_samples_split,
               #'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#Runtime restrictions lead to very small hyper parameter searchspace
rf = RandomForestClassifier(random_state = 42)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,
                               n_iter = 10, cv = 5, verbose=2, random_state=42,
                               n_jobs=-1, scoring='f1')
rf_random.fit(X_train_loaded, y_train_loaded)

predictions = rf_random.best_estimator_.predict(X_test_loaded)

print("Accuracy:", accuracy_score(y_test_loaded, predictions))
print("F1 Score:", f1_score(y_test_loaded, predictions))
conf_matrix = sklearn.metrics.confusion_matrix(y_test_loaded, predictions)

print("Conf matrix" + str(conf_matrix))

ax = plt.gca()
rfc_disp = plot_roc_curve(rf_random.best_estimator_, X_test_loaded, y_test_loaded, ax=ax, alpha=0.8)
rfc_disp.plot(ax=ax, alpha=0.8)
plt.show()