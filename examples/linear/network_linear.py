import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

train_df = pd.read_csv('./data/train.csv')
np.random.shuffle(train_df.values)

print(train_df.head())

model = keras.Sequential([
	keras.layers.Dense(4, input_shape=(2,), activation='relu'),  # Input of a 2neurones layer that feeds into a 4-nodes Dense layer (forward fully connected layer) with a ReLu activation 
	keras.layers.Dense(2, activation='sigmoid')]) # output layer contains as many nodes as labels

model.compile(optimizer='adam', 
	          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # calculate the loss between outputs and labels (labels provided as a vector of integers, not one hot format)  |  from_logits=True when the values of the loss obtained by the model are not normalized
	          metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values)) # get the data feature values in a np.array

model.fit(x, train_df.color.values, batch_size=4, epochs=5)

test_df = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

print("EVALUATION")
model.evaluate(test_x, test_df.color.values)





