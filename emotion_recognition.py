'''
Data set introduction
The data consists of 48x48 pixel grayscale images of faces
0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The faces have been automatically registered so that the face is more or less centered
and occupies about the same amount of space in each image
'''

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree


''' ### Read csv data '''

df = pd.read_csv('datasets\\fer2013\\train.csv')
print("There are total ", len(df), " sample in the loaded dataset.")
print("The size of the dataset is: ", df.shape)
# get a subset of the whole data for now
df = df.sample(frac=0.1, random_state=46)
print("The size of the dataset is: ", df.shape)

### Extract images and label from the dataframe df
width, height = 48, 48
images = df['pixels'].tolist()
faces = []
for sample in images:
    # Splitting the string by space character as a list
    face = [int(pixel) for pixel in sample.split(' ')]
    # convert pixels to images and # Resizing the image
    face = np.asarray(face).reshape(width*height)
    faces.append(face.astype('float32') / 255.0)       # Normalization
faces = np.asarray(faces)

# Get labels
y = df['emotion'].values

class_names = ['Angry', 'Disgust', 'Fear',
               'Happy', 'Sad', 'Surprise', 'Neutral']
# Visualization a few sample images
plt.figure(figsize=(5, 5))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.squeeze(faces[i].reshape(width, height)), cmap='gray')
    plt.xlabel(class_names[y[i]])
plt.show()

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    faces, y, test_size=0.1, random_state=46)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

# Now that our classifier has been trained, let's make predictions on the test data. To make predictions, the predict method of the DecisionTreeClassifier class is used.
y_pred = svclassifier.predict(X_test)

# For classification tasks some commonly used metrics are confusion matrix, precision, recall, and F1 score.
# These are calculated by using sklearn's metrics library contains the classification_report and confusion_matrix methods
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



''' Decision Tree '''
emotions_dataset = pd.read_csv('datasets\\fer2013\\normalizedData-collectedData-sorted.csv')
print("\n\n\nThere are total ", len(emotions_dataset), " sample in the loaded dataset.")
# print(emotions_dataset)

X = emotions_dataset.iloc[:,:-1].values
Y = emotions_dataset.iloc[:,10]

print(X)
print(Y)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

prediction = clf.predict([
    [0.2312, 0.67535, 0.34, 0, 0.1122, 0.514453, 0.1221, 0.8643, 0.323, 0.367613],
    [0.2, 0, 0.34, 0, 0.232, 0.9, 0.234, 0.86, 0.435, 0.787],
    [0.9475984379, 1, 0.058230485348, 0.999, 0.223341, 0.57634, 0.1111, 0.8654, 0, 0.24],
    [0, 1, 0, 0, 0, 0, 0.9999, 0.4543, 1, 0.23423],
    [0.3645645, 0.2223, 0.34323, 0.235521, 0.2452134, 0.256542, 0.45616, 0.49765, 0.4162, 0.6007],
])

print(f"Prediction: {prediction}")

tree.plot_tree(clf)
plt.show()
