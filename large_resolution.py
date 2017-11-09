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

files_for_testing=['P2 (Jan. 17)', 'P7 (Feb. 18)', 'P9 (Mar. 31)','P10 (May 1)', 'P11 (May 7)', 'P15 (May 11)', 'P18 (May 22)', 'P19 (May 26)']

window_size= [32, 20, 16, 14, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
#window_size = [1]
#window_gap = 1


def modifyUsingChunk(data, predict,window_size, time):

	#hardcode_data = data[:,0]

	length_of_data = len(predict)
	output = []

	most_frequent_list = []


	for i in range(0, length_of_data - window_size + 1):
		temp = []
		for j in range(i, i + window_size):
			#if time[j] <= time[i] + window_size -1:
			temp.append(predict[j])
		most_frequent = Counter(np.array(temp)).most_common(1)[0][0]
		most_frequent_list.append(most_frequent)


	for i in range(length_of_data):
		if i < window_size - 1:
			output.append(Counter(np.array(most_frequent_list[0 : (i + 1)])).most_common(1)[0][0])
		elif i >= length_of_data - window_size + 1:
			output.append(Counter(np.array(most_frequent_list[(i - window_size + 1) : ])).most_common(1)[0][0])
		else:
			output.append(Counter(np.array(most_frequent_list[(i - window_size + 1) : (i + 1)])).most_common(1)[0][0])


	return output


def ml(X_train, X_test,Y_train, Y_test, window, time):
	RF = []
	modified_predictions = []
	cohens = []
	clf = RandomForestClassifier(n_estimators=43, criterion = 'gini')
	#clf = svm.SVC(C = 3)
	clf.fit(X_train, Y_train)

	prediction=clf.predict(X_test)

	for w in window:

		modified_prediction = modifyUsingChunk(X_test, prediction, w, time)

		accuracy=accuracy_score(Y_test, modified_prediction)
		cohen = cohen_kappa_score(Y_test, modified_prediction)

		cohens.append(cohen)

		RF.append(accuracy)
		modified_predictions.append(modified_prediction)
	return RF, modified_predictions, cohens


if __name__ == '__main__':

	#maxWindowSize = [2, 3, 5, 14, 20, 9, 12, 6, 9, 2]

	#maxWindowSize = [2, 20, 20, 2, 20, 9, 2, 6, 5, 4]

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

		for i in range(5,len(training_label[0])):

			training_label = np.array(training_label)
			test_label = np.array(test_label)
			time = np.array(test_data)[:, 0]


			RF, prediction,cohens=ml(np.array(training_data),np.array(test_data),np.array(training_label)[:,i],np.array(test_label)[:,i], [11], time)

			giant_plot(np.array(test_label)[:,0], np.array(test_data), np.array(test_label)[:,i], prediction, './new/', file_for_testing, str(11))
			final = RF


			print getOutput(str(file_for_testing))," = ", final



