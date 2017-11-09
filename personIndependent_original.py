import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import butter, lfilter
from scipy.signal import savgol_filter
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.stats.stats import pearsonr   
from sklearn import svm
from scipy.signal import freqz
from scipy.fftpack import rfft, irfft, fftfreq
import scipy
import csv
from scipy.stats import entropy
from confusion import drawCM

from sklearn import svm

data_folder="users/data/"
labels_folder="users/labels/"

file_for_testing='P7 (Feb. 18)'

def ml(X_train, X_test,Y_train, Y_test):

    #print "num of trees: ",estimator
    clf = RandomForestClassifier(n_estimators=47)
    clf.fit(X_train, Y_train)

    prediction=clf.predict(X_test)
    accuracy=accuracy_score(Y_test, prediction)
    #print "accuracy = ",accuracy
    alll = list(set(Y_test)) + list(set(prediction))
    print alll
    alll=list(set(alll))
    confusion=confusion_matrix(Y_test, prediction, labels = alll)
    drawCM(confusion, alll, file_for_testing)
    #print "labels = ",labels
    #print "confusion matrix = ",confusion

    #plt.plot(it,acc,label="Random Forest")
    #RF.append(max(acc))
    RF=accuracy
    #plotFeatures(prediction,Y_test,"RF")
    ##plt.ylim(ymin=0.5,ymax=1)
    #plt.xlabel("num of estimators")
    #plt.ylabel("accuracy")
    #plt.title(str(states[s])+" states")
    #plt.legend(loc='lower right')
    
    #plt.show()
    return RF


if __name__ == '__main__':

	training_data=[]
	training_label=[]
	test_data=[]
	test_label=[]

	files = os.listdir(data_folder)

	for f in files:
		file=open(data_folder+str(f),"r")
		reader = csv.reader(file) 

		if file_for_testing in f:	    	
			for row in reader:
				test_data.append(row)
		else:
			for row in reader:
				training_data.append(row)


		file.close()


	files=os.listdir(labels_folder)


	for f in files:
		
		file=open(labels_folder+str(f),"r")
		reader = csv.reader(file) 

		if file_for_testing in f:		
			for row in reader:
				test_label.append(row)

		else:
			for row in reader:
				training_label.append(row)


		file.close()

	print  np.array(training_data).shape, np.array(training_label).shape, np.array(test_data).shape, np.array(test_label).shape

	states =  [8, 7, 5, 3, 2]

	final=[]

	for i in range(1,len(training_label[0])):

		RF=ml(np.array(training_data),np.array(test_data),np.array(training_label)[:,i],np.array(test_label)[:,i])

		print RF
		final.append(RF)


	print final



