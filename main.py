from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import numpy as np

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

X_train = []
Y_train = []
x_test = []
y_test = []


# https://www.thepythoncode.com/article/hog-feature-extraction-in-python
def loading_images(train_test, class_name, number_of_images, x, y):
    for i in range(1, number_of_images + 1):
        path = train_test + '/' + class_name + '/' + str(i) + '.jpg'
        img = imread(path)
        resized_img = resize(img, (128, 64))

        # check whether image is colored or greyscale
        try:
            # Colored
            fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True, channel_axis=-1)
        except ValueError as e:
            # greyscale
            fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True)
        x.append(fd)
        y.append(class_name)


# Training set
loading_images('train', 'accordian', 14, X_train, Y_train)
loading_images('train', 'dollar_bill', 14, X_train, Y_train)
loading_images('train', 'motorbike', 14, X_train, Y_train)
loading_images('train', 'Soccer_Ball', 15, X_train, Y_train)

# Testing set
loading_images('test', 'accordian', 2, x_test, y_test)
loading_images('test', 'dollar_bill', 2, x_test, y_test)
loading_images('test', 'motorbike', 2, x_test, y_test)
loading_images('test', 'Soccer_Ball', 3, x_test, y_test)

# pre-processing
X_train = np.array(X_train)
Y_train = np.array(Y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# Addding the column to fix the shape
Y_train = Y_train[:, np.newaxis]
y_test = y_test[:, np.newaxis]

# print(X_train.shape)
# print(Y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# DON'T FORGET TO NORMALIZE >>>> e+ >>FOUND
X_train = preprocessing.normalize(X_train)
x_test = preprocessing.normalize(x_test)

# print(X_train)
# print(x_test)


# Tunning hyperparameters
# https://www.analyticsvidhya.com/blog/2021/06/tune-hyperparameters-with-gridsearchcv/

# Defining the parameters grid for GridSearchCv
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.0001, 0.00, 0.1, 1],
    'kernel': ['linear', 'rbf', 'poly']
}

# Creating a support vector classifier
svc = svm.SVC(probability=True)

# Model Creation using GridSearch
model = GridSearchCV(svc, param_grid)

# Training
model.fit(X_train, Y_train.ravel())

# Testing
y_pred = model.predict(x_test)

# Calculate Accuracy
accuracy = accuracy_score(y_pred, y_test)

print("The model accuracy = " + str(accuracy))
print("Model accurate with percent " + str(accuracy*100)+"%")