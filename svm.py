from sklearn.model_selection import train_test_split
import helpers

# Find paths and labels
dir_dataset = helpers.getPath()
filenames = helpers.getFilenames(dir_dataset)
labels = helpers.getLabels(dir_dataset)

# Flatten the data set
print("> ------ Creating SVM classifier ------")
print("> Collecting Image data ...")
X, y = helpers.flattenImages(filenames, labels)

print("> Splitting train and test data ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print("> Preprocess data ...")
X_train_HOG = helpers.SVM_preProcessing(X_train)
X_test_HOG = helpers.SVM_preProcessing(X_test)

print("> Creating a model ...")
# hog  pca = 70, kernel='rbf', C=5, gamma=0.16 leads to good results.
svc_clf = helpers.SVM_getModel(X_train_HOG,  y_train)

# Train the classifier
print("> Training the model ...")
y_pred = svc_clf.predict(X_test_HOG)  

accuracy = helpers.getAccuracy(y_pred, y_test)
print(f"> The accuracy of the model is {accuracy}")

#print(f"> Plotting the image {y[0]} ...")
#displayImage(X_train[0])

helpers.plotPredictions(X_test[0:9], y_test[0:9], y_pred[0:9], accuracy, 'SVM')
#helpers.plotPredictions(svm.X_test[6:14], svm.y_test[6:14], svm.y_pred[6:14], svm.accuracy, 'SVM')