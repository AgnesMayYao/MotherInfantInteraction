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
from personIndependent import giant_plot, getOutput
from sklearn.metrics import cohen_kappa_score

from sklearn import svm


from collections import Counter



data_folder="users/data_normalized/"
labels_folder="users/labels_new/"

files_for_testing=['P2 (Jan. 17)', 'P7 (Feb. 18)', 'P9 (Mar. 31)', \
'P10 (May 1)', 'P11 (May 7)', 'P15 (May 11)', 'P18 (May 22)', 'P19 (May 26)']
priority = ['', '', '', '']

window_size= [32, 20, 16, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
#window_size = [1]
#window_gap = 1


def modifyUsingChunk(data,window_size,clf):

	#hardcode_data = data[:,0]

	length_of_data = len(data)
	output = []

	most_frequent_list = []
	temp = []


	for i in range(0, length_of_data - window_size + 1):
		temp.append(np.median(data[i:i+window_size, :], axis = 0))
	predict = clf.predict(temp)
	most_frequent_list = predict


	for i in range(length_of_data):
		if i < window_size - 1:
			output.append(Counter(np.array(most_frequent_list[0 : (i + 1)])).most_common(1)[0][0])
		elif i >= length_of_data - window_size + 1:
			output.append(Counter(np.array(most_frequent_list[(i - window_size + 1) : ])).most_common(1)[0][0])
		else:
			output.append(Counter(np.array(most_frequent_list[(i - window_size + 1) : (i + 1)])).most_common(1)[0][0])


	return output


 


def ml(X_train, X_test,Y_train, Y_test,window, files_for_testing):
	RF = []
	cohens = []
	modified_predictions = []
	clf = RandomForestClassifier(n_estimators=43, criterion = 'gini')
	#clf = svm.SVC(C = 3)
	clf.fit(X_train, Y_train)

	#for w in window:

	prediction = modifyUsingChunk(X_test, window, clf)

	accuracy=accuracy_score(Y_test, prediction)

	alll = list(set(Y_test)) + list(set(prediction))
	#print alll
	alll=list(set(alll))
	confusion=confusion_matrix(Y_test, prediction, labels = alll)
	drawCM(confusion, alll, file_for_testing)
	#cohen = cohen_kappa_score(Y_test, prediction)

	#cohens.append(cohen)
	#RF.append(accuracy)
	#modified_predictions.append(prediction)

	return accuracy, prediction
	#return RF, modified_predictions


if __name__ == '__main__':

	dump =[]

	#maxWindowSize = [2, 3, 5, 14, 20, 9, 12, 6, 9, 2]

	#maxWindowSize = [2, 20, 20, 2, 20, 9, 2, 6, 5, 4]
	#maxWindowSize = [2, 6, 7, 9, 9, 3, 32, 6, 5, 2]
	maxWindowSize = [2, 4, 2, 10, 7, 3, 32, 20, 3, 12]

	for k, file_for_testing in enumerate(files_for_testing): 


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
					test_data.append(map(float, row)) 
			else:
				for row in reader:
					training_data.append(map(float, row))


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

		#print  np.array(training_data).shape, np.array(training_label).shape, np.array(test_data).shape, np.array(test_label).shape

		states =  [8, 4, 3, 2]

		final=[]

		#print len(training_label[0])

		for i in range(3,len(training_label[0])-2):

			training_label = np.array(training_label)
			test_label = np.array(test_label)
			training_data = np.array(training_data)
			test_data = np.array(test_data)


			RF, prediction =ml(training_data,test_data,training_label[:,i],test_label[:,i], maxWindowSize[k], files_for_testing)

			#giant_plot(np.array(test_label)[:,0], np.array(test_data), np.array(test_label)[:,i], prediction, './RF_median_bestWindow_3States/', file_for_testing, str(maxWindowSize[k]))
	
			final = RF

			print getOutput(str(file_for_testing))," = ", final

