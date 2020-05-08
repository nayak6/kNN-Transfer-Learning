#!/usr/bin/env python3

import csv
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def classifier(trainingSetFilePath, testSetFilePath):
    dfTrainingSet = pd.read_csv(trainingSetFilePath, delimiter = ';',
                                skip_blank_lines = True,
                                usecols=['school', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'traveltime', 'paid', 'nursery', 'age', 'studytime', 'failures', 'schoolsup',
                                         'famsup', 'activities', 'higher',
                                         'internet', 'romantic', 'famrel',
                                         'freetime', 'goout', 'Dalc',
                                         'Walc', 'health', 'absences', 'G'])

    # can be changed into a function for binary variable transformation

    lb = LabelEncoder()
    dfTrainingSet['school'] = lb.fit_transform(dfTrainingSet['school'])
    dfTrainingSet['address'] = lb.fit_transform(dfTrainingSet['address'])
    dfTrainingSet['famsize'] = lb.fit_transform(dfTrainingSet['famsize'])
    dfTrainingSet['Pstatus'] = lb.fit_transform(dfTrainingSet['Pstatus'])
    dfTrainingSet['Medu'] = lb.fit_transform(dfTrainingSet['Medu'])
    dfTrainingSet['Fedu'] = lb.fit_transform(dfTrainingSet['Fedu'])
    dfTrainingSet['paid'] = lb.fit_transform(dfTrainingSet['paid'])
    dfTrainingSet['nursery'] = lb.fit_transform(dfTrainingSet['nursery'])

    dfTrainingSet['schoolsup'] = lb.fit_transform(dfTrainingSet['schoolsup'])
    dfTrainingSet['famsup'] = lb.fit_transform(dfTrainingSet['famsup'])
    dfTrainingSet['activities'] = lb.fit_transform(dfTrainingSet['activities'])
    dfTrainingSet['higher'] = lb.fit_transform(dfTrainingSet['higher'])
    dfTrainingSet['internet'] = lb.fit_transform(dfTrainingSet['internet'])
    dfTrainingSet['romantic'] = lb.fit_transform(dfTrainingSet['romantic'])

    trainingSetdf = dfTrainingSet.drop(columns=['G'])

    dfTestSet = pd.read_csv(testSetFilePath, delimiter=';',
                            skip_blank_lines= True,
                            usecols= ['school', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'traveltime', 'paid', 'nursery','age', 'studytime', 'failures', 'schoolsup',
                                      'famsup', 'activities', 'higher',
                                      'internet', 'romantic', 'famrel',
                                      'freetime', 'goout', 'Dalc',
                                      'Walc', 'health', 'absences'])

    dfTestSet['school'] = lb.fit_transform(dfTestSet['school'])
    dfTestSet['address'] = lb.fit_transform(dfTestSet['address'])
    dfTestSet['famsize'] = lb.fit_transform(dfTestSet['famsize'])
    dfTestSet['Pstatus'] = lb.fit_transform(dfTestSet['Pstatus'])
    dfTestSet['Medu'] = lb.fit_transform(dfTestSet['Medu'])
    dfTestSet['Fedu'] = lb.fit_transform(dfTestSet['Fedu'])
    dfTestSet['paid'] = lb.fit_transform(dfTestSet['paid'])
    dfTestSet['nursery'] = lb.fit_transform(dfTestSet['nursery'])

    dfTestSet['schoolsup'] = lb.fit_transform(dfTestSet['schoolsup'])
    dfTestSet['famsup'] = lb.fit_transform(dfTestSet['famsup'])
    dfTestSet['activities'] = lb.fit_transform(dfTestSet['activities'])
    dfTestSet['higher'] = lb.fit_transform(dfTestSet['higher'])
    dfTestSet['internet'] = lb.fit_transform(dfTestSet['internet'])
    dfTestSet['romantic'] = lb.fit_transform(dfTestSet['romantic'])

    # Standard Normalization

    scalar = MinMaxScaler()
    scalar.fit(trainingSetdf)
    trainingSetdf = scalar.transform(trainingSetdf)
    scalar.fit(dfTestSet)
    dfTestSet = scalar.transform(dfTestSet)

    # test_train split data

    #
    # trainingSetdf, dfTestSet, train, test = train_test_split(trainingSetdf,  dfTrainingSet['G'].values, test_size= 0.20)
    # # print(dfTestSet[1:10])
    # print(Counter(test))


    # kNN Classifier

    knnClassifier = KNeighborsClassifier(n_neighbors= 29)

    # Getting the labels

    labels = dfTrainingSet['G'].values

    # knnClassifier.fit(trainingSetdf, train)

    knnClassifier.fit(trainingSetdf, labels)

    predictClass = knnClassifier.predict(dfTestSet)

    numCount = Counter(predictClass)
    totalCount = numCount["+"] + numCount["-"]

    # positives distribution in the test dataset

    positivePercentage = (numCount["+"]/totalCount)*100


    # 20% of the positives
    topPercent = (positivePercentage*0.2/100)

    #top 20% of the all the individuals (10 = 20% of 50 based on 50-50
    # distribution idea)

    topTenpercent = int(totalCount*topPercent)

    gCLassPredictProbab = knnClassifier.predict_proba(dfTestSet)

    def labelSapFun(x):
        if x[0] > x[1]:
            return x[0]
        else:
            return -1


    classLabelsList = (np.apply_along_axis(labelSapFun, axis=1, arr=gCLassPredictProbab))

    # getting indices of top '+' individuals

    if topTenpercent != 0:
        topPositives = sorted(range(len(classLabelsList)), key=lambda i: classLabelsList[i])[-topTenpercent:]
    else:
        topPositives = []

    # print(len(topPositives))
    # print((totalCount - len(topPositives)))
    for index, classLabel in enumerate(predictClass):
        if index in topPositives:
            print('+')

        else:
            print('-')

    from sklearn.metrics import accuracy_score
    # print(accuracy_score(test, predictClass))

try:
    trainingSetFilePath = sys.argv[1]
    testSetFilePath = sys.argv[2]
    classifier(trainingSetFilePath, testSetFilePath)
except IndexError:
    print("Provide both training and test file path")
    exit(0)


