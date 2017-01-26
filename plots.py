import numpy as np
import matplotlib.pyplot as plt
import itertools
import sklearn


def plot_feature_importances(model, dataset):
    n_features = dataset.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.columns.values)
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")


def __confusion_matrix__(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_conf_mat(model, dataset_x, dataset_y, classnames):
    true_x = dataset_x
    true_y = dataset_y
    pred = model.predict(true_x)

    cnf_matrix = sklearn.metrics.confusion_matrix(true_y, pred)
    plt.figure()
    __confusion_matrix__(cnf_matrix, classes=classnames, title='Confusion matrix')
