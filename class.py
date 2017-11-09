import csv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import numpy as np
import pickle
from getFeatures import getInput
from plot import plotFeatures
###accuracy over iterations

##infant state
if __name__ == '__main__':
    #states=[3,4,6,7]
    states=[2,3,5,6]
    AC=[]
    LR=[]
    RF=[]
    SVM=[]
    #states=[2]
    
    for s in range(0,len(states)):
        
        
        X, Y, temp_dict=getInput(windowSize = 1, numStates = states[s])
        #[(trainX, trainY), (testX, testY), temp_dict]=getInput(windowSize=1,numStates=states[s])
        print states[s]," states"     
        X=np.array(X)
        Y=np.array(Y)

        #X_train=np.array(trainX)
        #X_test=np.array(testX)
        #Y_train=np.array(trainY)
        #Y_test=np.array(testY)
        
        
        dictionary = {v: k for k, v in temp_dict.iteritems()}
        labels=[]
        for i in range(0,len(dictionary)):
            labels.append(dictionary[i])


        X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)

        """
        print "#########################"
        print "Gradient Boosting Classifier"
        for estimator in range(2,33,2):
            print "num of iterations: ",estimator
            clf = GradientBoostingClassifier(n_estimators=estimator)
            clf.fit(X_train, Y_train)
            prediction=clf.predict(X_test)
            accuracy=accuracy_score(Y_test, prediction)
            print "accuracy = ",accuracy
            confusion=confusion_matrix(Y_test, prediction)
            print "labels = ",labels
            print "confusion matrix = ",confusion
        
        
        print "\n#########################"
        print "Neural Networks"
        for estimator in range(16,101,8):
            print "num of iterations: ",estimator
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(2,6), random_state=1,max_iter=estimator)
            clf.fit(X_train, Y_train)
            prediction=clf.predict(X_test)
            accuracy=accuracy_score(Y_test, prediction)
            print "accuracy = ",accuracy
            confusion=confusion_matrix(Y_test, prediction)
            print "labels = ",labels
            print "confusion matrix = ",confusion
        """
        ##print "#########################"
        print "Adaboost Classifier"
        it=[]
        acc=[]
        for estimator in range(2,49,2):
            #print "num of iterations: ",estimator
            it.append(estimator)
            clf = GradientBoostingClassifier(n_estimators=estimator)
            clf.fit(X_train, Y_train)

            prediction=clf.predict(X_test)
            accuracy=accuracy_score(Y_test, prediction)
            #print "accuracy = ",accuracy
            acc.append(accuracy)
            confusion=confusion_matrix(Y_test, prediction)
            #print "labels = ",labels
        print "confusion matrix = ",confusion

        #plt.plot(it,acc,label="Adaboost Classifier")
        AC.append(accuracy)
        #plotFeatures(prediction,Y_test,"AC")
            
        ##print "#########################"
        print "SVM"
        it=[]
        acc=[]
        for estimator in range(2,49,2):
            #print "num of iterations: ",estimator
            it.append(estimator)
            clf = clf = svm.SVC()
            clf.fit(X_train, Y_train)

            prediction=clf.predict(X_test)
            accuracy=accuracy_score(Y_test, prediction)
            #print "accuracy = ",accuracy
            acc.append(accuracy)
            confusion=confusion_matrix(Y_test, prediction)
            #print "labels = ",labels
        print "confusion matrix = ",confusion
   
        #plt.plot(it,acc,label="SVM")
        SVM.append(accuracy)
        #plotFeatures(prediction,Y_test,"SVM")


        print "Logistic Regression"
        it=[]
        acc=[]
        for estimator in range(2,49,2):
            #print "num of iterations: ",estimator
            it.append(estimator)
            clf = LogisticRegression(max_iter=estimator)
            clf.fit(X_train, Y_train)

            prediction=clf.predict(X_test)
            accuracy=accuracy_score(Y_test, prediction)
            #print "accuracy = ",accuracy
            acc.append(accuracy)
            confusion=confusion_matrix(Y_test, prediction)
            #print "labels = ",labels
        print "confusion matrix = ",confusion
        #plt.plot(it,acc,label="Logistic Regession")
        LR.append(accuracy)
        #plotFeatures(prediction,Y_test,"LG")

            
            
        #print "#########################"
        print "Random Forest Classifier"
        it=[]
        acc=[]
        for estimator in range(2,49,2):
            #print "num of trees: ",estimator
            it.append(estimator)
            clf = RandomForestClassifier(n_estimators=estimator)
            clf.fit(X_train, Y_train)

            prediction=clf.predict(X_test)
            accuracy=accuracy_score(Y_test, prediction)
            acc.append(accuracy)
            #print "accuracy = ",accuracy
            confusion=confusion_matrix(Y_test, prediction)
            #print "labels = ",labels
        print "confusion matrix = ",confusion
        #plt.plot(it,acc,label="Random Forest")
        RF.append(accuracy)
        #plotFeatures(prediction,Y_test,"RF")
        ##plt.ylim(ymin=0.5,ymax=1)
        #plt.xlabel("num of estimators")
        #plt.ylabel("accuracy")
        #plt.title(str(states[s])+" states")
        #plt.legend(loc='lower right')
        
        #plt.show()

    plt.plot(states,LR,label="Logistic Regression")
    plt.plot(states,RF,label="Random Forest")
    plt.plot(states,SVM,label="Support Vector Machine")
    plt.plot(states,AC,label="Adaboost Classifier")


    plt.ylim(ymin=0.3,ymax=1)


    plt.xlabel("num of states")
    plt.ylabel("Accuracy")
    plt.title("Nov11")

    plt.legend(loc='upper right')
    plt.show()

