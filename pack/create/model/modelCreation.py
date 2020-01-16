import pickle

import numpy as np

from create.model import loadImages

#make our model!
X = []
Y = []

for features,label in loadImages.training_data:
    X.append(features)
    Y.append(label)

print(X[0].reshape(-1, loadImages.IMG_SIZE, loadImages.IMG_SIZE, 1))

X = np.array(X).reshape(-1, loadImages.IMG_SIZE, loadImages.IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle", "wb")
pickle.dump(Y, pickle_out)
pickle_out.close()

print('Model created')