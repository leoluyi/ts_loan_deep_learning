
# coding: utf-8

# In[117]:

import matplotlib.pyplot as plt
import numpy as np; np.random.seed(123)
import pandas as pd

# Backend
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.regularizers import l1,l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import keras.backend as K
from keras.models import load_model

# Scikit learn
import sklearn as sk
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,     recall_score, confusion_matrix, classification_report,     accuracy_score, f1_score
    
# IO
import paratext


# ## Read Data

# In[125]:

print('Reading data')
#df_all = pd.read_csv('./casted_data_norm.csv', encoding='utf-8')
df_all = paratext.load_csv_to_pandas('data/casted_data_norm.csv', in_encoding='utf-8')
print('(Read data) End')

# In[3]:

df_all = df_all.assign(RSP_FLG_N = 1-df_all.RSP_FLG)
print(df_all.shape)
df_all.head()


# In[4]:

print(len(df_all.columns.values) - 4)
# 600 = 12 * 50


# In[5]:

X_train, X_test, Y_train, Y_test =     train_test_split(
        df_all.iloc[:, 1:-3].as_matrix(), 
        df_all[['RSP_FLG_N', 'RSP_FLG']].as_matrix(),
        test_size=0.33,
        random_state=42)


# In[6]:

[print(x.shape) for x in [X_train, X_test, Y_train, Y_test]];


# In[7]:

# input image dimensions
img_rows, img_cols = 12, 50
nb_classes = 2  # 2 digits from 0 to 1

# Reshape data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")


# In[8]:

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print("One hot encoding:\n{}".format(Y_train[0:5, :]))


# In[57]:

# Optimizer
sgd = SGD(lr=0.17,momentum=0.9,decay=0.01,nesterov=False)
adam = Adam(decay=0.001)

# Early Stopping
earlyStopping=EarlyStopping(monitor= 'val_loss', patience=20)

# Class weight
class_weight = {0: .1, 1: .9}

# Custom metrics
def custom_metrics(y_true_, y_pred_, beta=1):
    y_true = K.cast(K.argmax(y_true_, 1), dtype=tf.float32)
    y_pred = K.cast(K.argmax(y_pred_, 1), dtype=tf.float32)
    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    
    recall = true_positives / (possible_positives +  K.epsilon())
    precision = true_positives / (predicted_positives + K.epsilon())
    
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision
    r = recall
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return {'precision': precision, 'recall': recall, 'fbeta_score': fbeta_score}

# Callback - https://keras.io/callbacks/#create-a-callback
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
history = LossHistory()

# Checkpoint
checkpointer = ModelCheckpoint(filepath="model_keras/weights.hdf5", 
                               monitor='val_loss',
                               verbose=1, 
                               save_best_only=True)


# In[40]:

# Claim a sequential model
model = Sequential()

# Input layer
model.add(Convolution2D(512, 3, 3, input_shape=(img_rows, img_cols, 1), border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
# Dropout
model.add(Dropout(0.2))

# Full-connection
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dropout(0.2))
model.add(Dense(2, activation = 'softmax', name = 'output'))


# In[42]:

model.compile(loss='categorical_crossentropy', 
              optimizer = adam, 
              metrics=['accuracy', custom_metrics])


# In[ ]:

batch_size = 500
nb_epoch = 30

history_callback = model.fit(
    X_train[:50000], Y_train[:50000], 
    batch_size=batch_size, 
    nb_epoch=nb_epoch,
    class_weight=class_weight,
    verbose=1, 
    shuffle=True,
    validation_split=0.33,
    callbacks=[earlyStopping, history, checkpointer])


# ## Saving Model

# In[134]:

'''saving model'''
model.save('model_keras/loan_keras_echoch_30.hdf5')
# del model


# ## Visualize model

# In[113]:

# Kplot(model, to_file='model.png')
#SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[49]:

#history_callback.history[]


# In[147]:

# summarize history for loss
plt.subplot(121)
plt.plot(history_callback.history['loss'])
plt.plot(history_callback.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');
# summarize history for f1-score
plt.subplot(122)
plt.plot(history_callback.history['fbeta_score'])
plt.plot(history_callback.history['val_fbeta_score'])
plt.title('model f1_score')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left');


# In[142]:

history_callback.history.keys()


# In[ ]:




# In[ ]:




# ## Prediction

# In[66]:

score = model.predict_proba(X_test[:])
y_pred = np.argmax(score, 1)


# In[100]:

print(score.shape)
print(Y_test[:, 1].shape)


# In[105]:

print(score[:5])
print(y_pred[:5])


# In[106]:

# http://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-an-numpy-array
y_score = np.c_[score, Y_test[:, 1]]


# In[102]:

# save prediction
np.savetxt("prediction/y_score.csv", y_score, delimiter=",")


# In[110]:

y_test = np.argmax(Y_test, 1)
y_test


# In[116]:

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy: {:.3f}'.format(accuracy))
print('Recall: {:.3f}'.format(recall))
print('Precision: {:.3f}'.format(precision))
print('F1: {:.3f}'.format(f1))


# In[ ]:



