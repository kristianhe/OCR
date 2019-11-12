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

print("> Preprocess data ...")
X = helpers.SVM_preProcessing(X)

print("> Splitting train and test data ...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print("> Cross validating parameters ...")
# hog  pca = 70, kernel='rbf', C=5, gamma=0.16 leads to good results.
svc_clf = helpers.SVM_getModel(X_train,  y_train)

svc_clf.fit(X_train, y_train) 
y_pred = svc_clf.predict(X_test)

accuracy = helpers.getAccuracy(y_pred, y_test)
print(f"> The accuracy of the model is {accuracy}")

#print(f"> Plotting the image {y[0]} ...")
#displayImage(X_train[0])
