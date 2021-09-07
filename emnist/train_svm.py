if __name__ == '__main__':
    from sklearn.metrics import plot_confusion_matrix


    def get_images(img_file, number):
        f = open(img_file, "rb")  # Open file in binary mode
        f.read(16)  # Skip 16 bytes header
        images = []

        for i in range(number):
            image = []
            for j in range(28 * 28):
                image.append(ord(f.read(1)))
            images.append(image)
        return images


    def get_labels(label_file, number):
        l = open(label_file, "rb")  # Open file in binary mode
        l.read(8)  # Skip 8 bytes header
        labels = []
        for i in range(number):
            labels.append(ord(l.read(1)))
        return labels


    import numpy as np
    from sklearn import svm, metrics
    from mnist import MNIST
    import cv2
    import matplotlib.pyplot as plt

    mndata = MNIST('data')
    # This will load the train and test data

    X_train, y_train = mndata.load('data/emnist-byclass-train-images-idx3-ubyte',
                                   'data/emnist-byclass-train-labels-idx1-ubyte')
    X_test, y_test = mndata.load('data/emnist-byclass-test-images-idx3-ubyte',
                                 'data/emnist-byclass-test-labels-idx1-ubyte')

    # Train

    print("Train")
    TRAINING_SIZE = 100

    X_train = get_images("data/emnist-byclass-train-images-idx3-ubyte", TRAINING_SIZE)

    X_train = np.array(X_train) / 255.0

    y_train = get_labels("data/emnist-byclass-train-labels-idx1-ubyte", TRAINING_SIZE)



    clf = svm.SVC()

    clf.fit(X_train, y_train)

    # Test

    img = cv2.imread("preview/000000-num35.png")
    i = np.array(img) / 255.0

    plt.imshow(img)
    plt.show()

    TEST_SIZE = 5
    X_test = get_images("data/emnist-byclass-test-images-idx3-ubyte", TEST_SIZE)

    X_test = np.array(X_test) / 255.0

    y_test = get_labels("data/emnist-byclass-test-labels-idx1-ubyte", TEST_SIZE)

    print(X_train.shape,'---',X_test.shape)



    # predict

    # print(X_train.shape)
    # print(X_train.shape[1])
    #

    print("Predict")

    print("------------")

    print("------------")

    # print(len(X_test[0]))

    # i2 = i.reshape(-1,784) / 255.0
    predict = clf.predict(X_test)
    # print(i2)

    ac_score = metrics.accuracy_score(y_test, predict)
    # cl_report = metrics.classification_report(y_test,predict)
    f1_score = metrics.f1_score(y_test, predict, average='macro')
    recall_score = metrics.recall_score(y_test, predict, average='macro')
    precision_score = metrics.precision_score(y_test, predict, average='macro')
    print("Score: ", ac_score)
    # print(cl_report)
    print('precision_score=', precision_score)
    print('f1_score=', f1_score)
    print('recall_score=', recall_score)
    # print('-----------------------------------')

    CM = metrics.confusion_matrix(y_test, predict);

    print('Confusion Matrix:\n', CM);

    # Plot non-normalized confusion matrix
    # titles_options = [("Confusion matrix, without normalization", None),
    #                   ("Normalized confusion matrix", 'true')]
    # for title, normalize in titles_options:
    #     disp = plot_confusion_matrix(clf, X_test, y_test,
    #                                  display_labels='class_names',
    #                                  cmap=plt.cm.Blues,
    #                                  normalize=normalize)
    #     disp.ax_.set_title(title)
    #
    #     print(title)
    #     print(disp.confusion_matrix)
    #
    # plt.show()

    # điều chỉnh parameters

    # from sklearn.model_selection import  GridSearchCV
    #
    # parameter_candidates = [
    #     {'C': [0.001,0.01,0.1,5,10,100,1000]}
    # ]
    #
    # clf = GridSearchCV(estimator=svm.SVC(),param_grid=parameter_candidates,n_jobs=-1)
    #
    # clf.fit(X_train,y_train)
    #
    # print('Best score: ',clf.best_score_)
    # print('Best C: ',clf.best_estimator_.C)

    # model_json = clf.to_json();
    # with open("model10.json", "w") as json_file:
    #     json_file.write(model_json);
    # saves the model info as json file

    # clf.save_weights("model10.h5")

    from joblib import dump, load

    dump(clf, 'mnist-svm.joblib')

    print('save file')

    # load file
    # clf = load('mnist-svm.joblib')










