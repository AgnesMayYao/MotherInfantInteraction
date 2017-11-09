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

from sklearn import svm

data_folder="users/data/"
labels_folder="users/labels/"

file_for_testing='P19 (May 26)'
graph_title=file_for_testing

def ml(X_train, X_test,Y_train, Y_test):

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
    #print "confusion matrix = ",confusion

    #plt.plot(it,acc,label="Random Forest")
    #RF.append(max(acc))
    global real,guess
    real.extend(Y_test)
    guess.extend(prediction)
    RF=max(acc)
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

		file.close()


	files=os.listdir(labels_folder)


	for f in files:
		
		file=open(labels_folder+str(f),"r")
		reader = csv.reader(file) 

		if file_for_testing in f:		
			for row in reader:
				test_label.append(row)


		file.close()


	print  np.array(test_data).shape, np.array(test_label).shape


	'''
	features=np.concatenate(([np.array(hardCodedPressure)[:,1]],[np.array(accMomVariance)[:,1]],[np.array(accBabyVariance)[:,1]],[np.array(preMomVariance)[:,1]],[np.array(preBabyVariance)[:,1]],[np.array(accCorrelation)[:,1]], \
        [np.array(accDifference)[:,1]],[np.array(preDifference)[:,1]],[np.array(selfAccDiffMom)[:,1]],[np.array(selfAccDiffBaby)[:,1]],[np.array(selfPreDiffMom)[:,1]], \
        [np.array(selfPreDiffBaby)[:,1]],[np.array(preCorrelation)[:,1]],[np.array(filteredPreMom)[:,1]],[np.array(filteredPreBaby)[:,1]]),axis=0)
    '''

	

	#features=np.array(test_data)[:,[0,3,4,7,10,11,12,13,14]]
	features=np.array(test_data)[:,[1,2,5,6,8,9]]
	labels=np.array(test_label)



	RF=[]
	states=[]

	la=1
	while la<6:

		global real, guess
		real=[]
		guess=[]

		tRF=[]

		numOfStates=len(np.unique(np.array(labels)[:,la]))
		states_total=[0]*numOfStates
		unique=np.unique(np.array(labels)[:,la])
		print "unique = ",unique

		newlabels=np.array(labels)[:,la]
		states.append(numOfStates)

		kf = KFold(n_splits=5)

		for train_index, test_index in kf.split(features):
			X_train, X_test = features[train_index], features[test_index]
			Y_train, Y_test = newlabels[train_index], newlabels[test_index]
			if len(np.unique(Y_train))>1:
				rf=ml(X_train, X_test,Y_train,Y_test)
				tRF.append(rf)
		RF.append(np.mean(tRF))

		get_it_right=[0]*len(unique)
		ttt=[0]*len(unique)
		unique=list(unique)
		for ii in range(len(real)):
			ttt[unique.index(real[ii])]+=1
			if real[ii]==guess[ii]:
				get_it_right[unique.index(real[ii])]+=1

		print "guessed_right = ", get_it_right
		print "total = ", ttt

		statistics=[]
		for ii in range(len(get_it_right)):
			statistics.append(get_it_right[ii]*1.0/ttt[ii])
		print "statistics =", statistics

		print "========================================="
		la=la+1

	plt.plot(states,RF,label="Random Forest")

	print "states = ", states
	print "RF = ", RF

	plt.ylim(ymin=0,ymax=1)

	plt.xlabel("num of states")
	plt.ylabel("Accuracy")
	plt.title(graph_title)

	plt.legend(loc='lower right')
	plt.show()
