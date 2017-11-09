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

data_folder="users/data_normalized/"
labels_folder="users/labels_new/"

directory = "./personIndependent_visulization/"

files_for_testing=['P2 (Jan. 17)', 'P3 (Jan. 27)', 'P5 (Feb. 13)', 'P7 (Feb. 18)', 'P9 (Mar. 31)', \
'P10 (May 1)', 'P11 (May 7)', 'P15 (May 11)', 'P18 (May 22)', 'P19 (May 26)']


small_font_size=10
large_font_size=20



def forLabels(data):

	available_states = list(set(data))

	states = ['nearby', 'hovering','touching','picking up', 'holding', 'carrying','bouncing','putting down', 'uncarrying']
	states = [x for x in states if x in available_states]
	output= [0]*len(data)

	for i in range(len(output)):
	    output[i]=states.index(data[i])
	return output, states



def giant_plot(time, data, label, predict,output_directory, file_for_testing, windowSize = str(1)):

	plt.clf()

	fig = plt.figure(figsize=(20, 15))

	
	number_labels_label, ordered_label = forLabels(label)
	num_states = len(ordered_label)
	elan=fig.add_subplot(10,1,10)
	elan.plot(time, number_labels_label,'black')
	elan.set_ylabel("ground truth",fontsize=small_font_size)
	elan.set_xlabel('time (s)',fontsize=small_font_size)

	elan.yaxis.set_major_locator(ticker.FixedLocator(xrange(len(ordered_label))))
	elan.yaxis.set_major_formatter(ticker.FixedFormatter(ordered_label))



	number_labels_predict, ordered_predict = forLabels(predict)
	prediction=fig.add_subplot(10,1,9, sharex=elan)
	prediction.plot(time, number_labels_predict,'black')
	prediction.set_ylabel("predicted labels",fontsize=small_font_size)
	plt.setp(prediction.get_xticklabels(),visible=False)
	prediction.yaxis.tick_right()
	prediction.yaxis.set_label_position("right")

	prediction.yaxis.set_major_locator(ticker.FixedLocator(xrange(len(ordered_predict))))
	prediction.yaxis.set_major_formatter(ticker.FixedFormatter(ordered_predict))

	'''
	preDiff = fig.add_subplot(714,sharex=elan)
	preDiff.plot(time, data[:,7])
	#acc.text(data_time[len(data_time)-1], data_acc[len(data_acc)-1], "mother", fontsize=small_font_size, color='purple')
	preDiff.set_xlabel('time (s)',fontsize=small_font_size)
	preDiff.set_ylabel('Absolute Pressure Difference',fontsize=small_font_size)
	preDiff.yaxis.tick_right()
	preDiff.yaxis.set_label_position("right")
	plt.setp(preDiff.get_xticklabels(),visible=False)
	'''

    
	selfAccDiff=fig.add_subplot(10,1,8,sharex=elan)
	selfAccDiff.plot(time, data[:,8],'purple')
	selfAccDiff.plot(time, data[:,9],'green')
	selfAccDiff.set_ylabel('acc derivative',fontsize=small_font_size)
	plt.setp(selfAccDiff.get_xticklabels(),visible=False)


	selfPreDiff=fig.add_subplot(10,1,7,sharex=elan)
	selfPreDiff.plot(time, data[:,10],'purple')
	selfPreDiff.plot(time, data[:,11],'green')
	selfPreDiff.set_ylabel('pressure derivative',fontsize=small_font_size)
	plt.setp(selfPreDiff.get_xticklabels(),visible=False)
	selfPreDiff.yaxis.tick_right()
	selfPreDiff.yaxis.set_label_position("right")


   

	accC=fig.add_subplot(10,1,6,sharex=elan)
	accC.plot(time, data[:,5],'purple')
	accC.set_ylabel('acc KL divergence',fontsize=small_font_size)
	plt.setp(accC.get_xticklabels(),visible=False)


	preC=fig.add_subplot(10,1,5,sharex=elan)
	preC.plot(time, data[:,12],'purple')
	preC.set_ylabel('Pressure Correlation',fontsize=small_font_size)
	plt.setp(preC.get_xticklabels(),visible=False)
	preC.yaxis.tick_right()
	preC.yaxis.set_label_position("right")


	accVariance=fig.add_subplot(10,1,4,sharex=elan)
	accVariance.plot(time, data[:,1],'purple')
	accVariance.plot(time, data[:,2],'green')
	accVariance.set_ylabel('acc Variance',fontsize=small_font_size)
	plt.setp(accVariance.get_xticklabels(),visible=False)


	'''
	accDiff=fig.add_subplot(614,sharex=elan)
	accDiff.plot(forPlot(accDifference)[0],forPlot(accDifference)[1],'purple')
	accDiff.set_ylabel("absolute acc Difference",fontsize=small_font_size)
	plt.setp(accDiff.get_xticklabels(),visible=False)
	'''

	preVariance=fig.add_subplot(10,1,3,sharex=elan)
	preVariance.plot(time, data[:,3],'purple')
	preVariance.plot(time, data[:,4],'green')
	preVariance.set_ylabel('pre Variance',fontsize=small_font_size)
	preVariance.yaxis.tick_right()
	preVariance.yaxis.set_label_position("right")
	plt.setp(preVariance.get_xticklabels(),visible=False)



	prefilter = fig.add_subplot(10,1,2,sharex=elan)
	prefilter.plot(time, data[:,13],'purple')
	prefilter.plot(time, data[:,13],'green')
	prefilter.set_ylabel('filtered pressure',fontsize=small_font_size)
	plt.setp(prefilter.get_xticklabels(),visible=False)




	hard=fig.add_subplot(10,1,1,sharex=elan)
	hard.plot(time, data[:,0],'purple')
	hard.set_ylabel('Hard Coded Pressure',fontsize=small_font_size)
	plt.setp(hard.get_xticklabels(),visible=False)
	hard.yaxis.tick_right()
	plt.setp(hard.get_xticklabels(),visible=False)

	hard.yaxis.set_major_locator(ticker.FixedLocator([1,2,3,4]))
	hard.yaxis.set_major_formatter(ticker.FixedFormatter(['low->high (both)','high->low (both)','low->high (mom)','high->low (mom)']))

	
	'''
	ax = [elan, prediction, selfPreDiff, selfAccDiff, accC, preC, accVariance, preVariance, prefilter, hard]



	#doing some painting stuff
	#bouncing:yellow,picking up: orange,putting down: pink,hovering: green, touching:grey,holding:blue,carrying: purple,	nearby: red
	correct_alpha = 0.1
	wrong_alpha = 0.1
	for i in range(len(time)-1):
		if label[i]=="holding":
			if label[i]==predict[i]:
				[j.axvspan(time[i],time[i+1],color='blue',alpha=correct_alpha) for j in ax]
			else:
				[j.axvspan(time[i],time[i+1],color='blue',alpha=wrong_alpha) for j in ax]
		elif label[i]=="nearby":
			if label[i]==predict[i]:
				[j.axvspan(time[i],time[i+1],color='red',alpha=correct_alpha) for j in ax]
			else:
				[j.axvspan(time[i],time[i+1],color='red',alpha=wrong_alpha) for j in ax]
		elif label[i]=="hovering":
			if label[i]==predict[i]:
				[j.axvspan(time[i],time[i+1],color='green',alpha=correct_alpha) for j in ax]
			else:
				[j.axvspan(time[i],time[i+1],color='green',alpha=wrong_alpha) for j in ax]
		elif label[i]=="carrying":
			if label[i]==predict[i]:
				[j.axvspan(time[i],time[i+1],color='purple',alpha=correct_alpha) for j in ax]
			else:
				[j.axvspan(time[i],time[i+1],color='purple',alpha=wrong_alpha) for j in ax]
		elif label[i]=="touching":
			if label[i]==predict[i]:
				[j.axvspan(time[i],time[i+1],color='grey',alpha=correct_alpha) for j in ax]
			else:
				[j.axvspan(time[i],time[i+1],color='grey',alpha=wrong_alpha) for j in ax]
		elif label[i]=="bouncing":
			if label[i]==predict[i]:
				[j.axvspan(time[i],time[i+1],color='yellow',alpha=correct_alpha) for j in ax]
			else:
				[j.axvspan(time[i],time[i+1],color='yellow',alpha=wrong_alpha) for j in ax]
		elif label[i]=="picking up":
			if label[i]==predict[i]:
				[j.axvspan(time[i],time[i+1],color='orange',alpha=correct_alpha) for j in ax]
			else:
				[j.axvspan(time[i],time[i+1],color='orange',alpha=wrong_alpha) for j in ax]
		else:
			if label[i]==predict[i]:
				[j.axvspan(time[i],time[i+1],color='pink',alpha=correct_alpha) for j in ax]
			else:
				[j.axvspan(time[i],time[i+1],color='pink',alpha=wrong_alpha) for j in ax]

	'''

	#plt.setp(pre.get_xticklabels(), visible=False)
	#plt.setp(rawAcc.get_xticklabels(), visible=False)
	#plt.setp(selfAccDiff.get_xticklabels(),visible=False)

	plt.title(file_for_testing,fontsize=small_font_size,loc="center")
	fig.subplots_adjust(hspace=0)


	plt.savefig(output_directory+file_for_testing+" "+str(num_states)+" states "+"window "+windowSize+".png")
	#plt.show()


	'''

	features=np.concatenate(([np.array(hardCodedPressure)[:,1]],[np.array(accMomVariance)[:,1]],[np.array(accBabyVariance)[:,1]],[np.array(preMomVariance)[:,1]],[np.array(preBabyVariance)[:,1]],[np.array(accCorrelation)[:,1]], \
	[np.array(accDifference)[:,1]],[np.array(preDifference)[:,1]],[np.array(selfAccDiffMom)[:,1]],[np.array(selfAccDiffBaby)[:,1]],[np.array(selfPreDiffMom)[:,1]], \
	[np.array(selfPreDiffBaby)[:,1]],[np.array(preCorrelation)[:,1]],[np.array(filteredPreMom)[:,1]],[np.array(filteredPreBaby)[:,1]]),axis=0)
	'''




def ml(X_train, X_test,Y_train, Y_test, num_estimators):

    #print "num of trees: ",estimator

    clf = svm.SVC(C = 3, kernel = 'poly')
    clf.fit(X_train, Y_train)

    #print clf.feature_importances_

    prediction=clf.predict(X_test)
    accuracy=accuracy_score(Y_test, prediction)
    #print "accuracy = ",accuracy
    alll = list(set(Y_test)) + list(set(prediction))
    #print alll
    alll=list(set(alll))
    confusion=confusion_matrix(Y_test, prediction, labels = alll)
    ###drawCM(confusion, alll, file_for_testing)
    #print "labels = ",labels
    #print "confusion matrix = ",confusion

    #plt.plot(it,acc,label="Random Forest")
    #RF.append(max(acc))
    SVM=accuracy
    #plotFeatures(prediction,Y_test,"RF")
    ##plt.ylim(ymin=0.5,ymax=1)
    #plt.xlabel("num of estimators")
    #plt.ylabel("accuracy")
    #plt.title(str(states[s])+" states")
    #plt.legend(loc='lower right')
    
    #plt.show()
    return SVM, prediction


if __name__ == '__main__':

	for file_for_testing in files_for_testing:

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

		training_data = np.array(training_data)
		test_data = np.array(test_data)

		for i in range(1,len(training_label[0])):


			SVM, prediction=ml(np.array(training_data),np.array(test_data),np.array(training_label)[:,i],np.array(test_label)[:,i], 43)

				#print RF
			final.append(SVM)

			#time, all data, label, prediction
			#special = 'carrying vs. uncarrying' if i == 5 else ''
			#giant_plot(np.array(test_label)[:,0], np.array(test_data), np.array(test_label)[:,i], prediction,"./personIndependent_newStates_baseline_SVM/", file_for_testing, special)



		print str(file_for_testing), final



