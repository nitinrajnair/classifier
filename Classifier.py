"""
__author__ = 'NN'

Author 1: NITINRAJ NAIR

description: This creates an histogram of the accuracy score of the different classifiers
when used on MNIST dataset. You can choose which classifier to be applied.
"""

"""You will have to downlasd and install the following 
modules to run the program."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Tkinter import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model


def read_data(train, test):
    X_train = train[0:, 1:]
    y_train = train[0:, 0]

    X_test = test[0:, 1:]
    y_test = test[0:, 0]
    return X_train, y_train, X_test, y_test


"""Here we are reading the dataset.
If you want to try on different dataset just 
download the csv file of that dataset in the samfolder as this file
 and use the same method as below"""

train = pd.read_csv("mnist_train.csv").values
test = pd.read_csv("mnist_test.csv").values
itrain, ltrain, itest, ltest = read_data(train, test)


def svc():
    clf = SVC(kernel="poly")
    clf.fit(itrain, ltrain)
    pred = clf.predict(itest)
    acc = accuracy_score(pred, ltest)
    print(acc)
    return acc * 100


def naivebayes():
    clf = GaussianNB()
    clf.fit(itrain, ltrain)
    pred = clf.predict(itest)
    acc = accuracy_score(pred, ltest)
    print(acc)
    return acc * 100


def decisiontree():
    clf = DecisionTreeClassifier()
    clf.fit(itrain, ltrain)
    pred = clf.predict(itest)
    acc = accuracy_score(pred, ltest)
    print(acc)
    return acc * 100


def random():
    clf = RandomForestClassifier(n_estimators=150, bootstrap=False)
    clf.fit(itrain, ltrain)
    pred = clf.predict(itest)
    acc = accuracy_score(pred, ltest)
    print(acc)
    return acc * 100


"""You can add additional clssifier code here."""
def nearestneighbor():
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(itrain, ltrain)
    pred = clf.predict(itest)
    acc = accuracy_score(pred, ltest)
    print(acc)
    return acc * 100


def main():
    root = Tk()
    naive = IntVar()
    svm = IntVar()
    tree = IntVar()
    near = IntVar()
    forest = IntVar()

    """Setting up the background frames"""
    back = Frame(root, width=100, height=100)
    back.pack()

    """Creating radio buttons for each classifier"""
    C1 = Checkbutton(root, text="Naive Bayes", variable=naive, onvalue=1, offvalue=0)
    C1.pack()
    C2 = Checkbutton(root, text="SVM", variable=svm, onvalue=1, offvalue=0)
    C2.pack()
    C3 = Checkbutton(root, text="Decision Tree", variable=tree, onvalue=1, offvalue=0)
    C3.pack()
    C4 = Checkbutton(root, text="Nearest Neighbor", variable=near, onvalue=1, offvalue=0)
    C4.pack()
    C5 = Checkbutton(root, text="Random Forest", variable=forest, onvalue=1, offvalue=0)
    C5.pack()

    """You can add more radio button for additional classifiers."""
    okButton = Button(root, text="OK", command=root.destroy)
    okButton.pack(side=RIGHT)
    root.mainloop()
    xaxis = []
    percentage = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    range = naive.get() + svm.get() + tree.get() + near.get() + forest.get()
    # set up some values for the bars
    bar_width = 0.4  # set the wid opacity = 0.5  # not so dark
    # setup the plots: both points and smooth curve
    x = np.arange(1)  # need an array of x values (but, see below)
    opacity = 0.5  # not so dark
    # setup the plots: both points and smooth curve
    if naive.get() == 1:
        plt.bar(0, naivebayes(), bar_width, color='green', label='Naive Bayes', alpha=opacity)
        xaxis.append("NAIVE")
    if svm.get() == 1:
        plt.bar(bar_width, svc(), bar_width, color='blue', label='SVM', alpha=opacity)
        xaxis.append("SVM")
    if tree.get() == 1:
        plt.bar(2 * bar_width, decisiontree(), bar_width, color='yellow', label='Decision Tree', alpha=opacity)
        xaxis.append("TREE")
    if near.get() == 1:
        plt.bar(3 * bar_width, nearestneighbor(), bar_width, color='black', label='Neighbor',
                alpha=opacity)
    if forest.get() == 1:
        plt.bar(4 * bar_width, random(), bar_width, color='orange', label='Forest',
                alpha=opacity)
        xaxis.append("NEAREST_NEIGHBOUR")

    """You can add more for additional classifiers."""
    if range != 0:
        plt.legend()
        plt.xlabel('Classifiers')
        plt.ylabel('Accuracy Score')
        plt.title('Accuracy Score vs Classfiers')
        plt.xticks(x, xaxis)
        plt.tight_layout()
        plt.show()
        main()
    elif range == 0:
        plt.close()


if __name__ == '__main__':
    main()
