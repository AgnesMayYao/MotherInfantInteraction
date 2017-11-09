import csv
import numpy as np
import os
import matplotlib.pyplot as plt
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



data_folder='data/Jan27/sensory reactivity/'
states=[]
states_detail=[]
baroOffset=0
small_font_size=10
large_font_size=20



def butter_bandpass(lowcut, highcut, fs, order=5):
    #nyq = 0.5 * fs=0.5*5000=2500
    #low = lowcut / nyq=500/2500=0.2
    #high = highcut / nyq=1250/2500=0.5

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def read_datafile(filename):
    file=open(filename,"r")
    reader = csv.reader(file)       
    row_no=1

    data=[]
    
    for row in reader:
        if row_no>1:
            temp=[]
            temp.append(int(int(row[0])*1.0/1000))
            for i in range(1,len(row)):
                temp.append(float(row[i]))

            if len(temp)>2:
                value=0
                for i in range(1,len(temp)):
                    value+=temp[i]*temp[i]
                value=np.sqrt(value)
                temp[1]=value
                temp=temp[0:2]

            data.append(temp)

        row_no+=1
    file.close()
    return data

def read_rawdatafile(filename):
    file=open(filename,"r")
    reader = csv.reader(file)       
    row_no=1

    data=[]
    
    for row in reader:
        if row_no>1:
            temp=[]
            temp.append(int(row[0])*1.0/1000)
            for i in range(1,len(row)):
                temp.append(float(row[i]))

            if len(temp)>2:
                value=0
                for i in range(1,len(temp)):
                    value+=temp[i]*temp[i]
                value=np.sqrt(value)
                temp[1]=value
                temp=temp[0:2]

            data.append(temp)

        row_no+=1
    file.close()
    return data


def combineIntoSeconds(data):
	seconds=data[0][0]

	new_data=[]
	buff=[]

	for i in range(len(data)):
		if data[i][0]==seconds:
			buff.append(data[i])
		else:
			if len(buff)>1:
				temp=list(np.mean(buff, axis=0))
			else:
				temp=buff[0]
			new_data.append(temp)
			buff=[data[i]]
			seconds=data[i][0]

	return new_data

def read_labelfile(filename):
    file=open(filename,"r")
    reader = csv.reader(file)       
    row_no=1

    data=[]
    this_detail=[]
    
    for row in reader:
    	global states
        if row_no>1:
        	temp=[]
        	temp.append(int(int(row[0])*1.0/1000))
        	for i in range(1,len(row)):
        		temp.append(row[i])
        	data.append(temp)
        else:
        	for i in range(1,len(row)):
        		states.append(int(row[i].split()[0]))

        row_no+=1
    file.close()

    return data

def crossCorr(baby, mom):
    correlations = []
    for i in range(len(baby)):
    	temp=[]
    	temp.append(baby[i][0])
        babyC = baby[i][1]
        momC = mom[i][1]
        temp.append(pearsonr(babyC, momC)[0])
        correlations.append(temp)
        print correlations
    return correlations


def ml(X_train, X_test,Y_train, Y_test):

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
        #confusion=confusion_matrix(Y_test, prediction)
        #print "labels = ",labels
    #print "confusion matrix = ",confusion
    #print acc
    #plt.plot(it,acc,label="Adaboost Classifier")
    #AC.append(max(acc))
    AC=max(acc)
    #plotFeatures(prediction,Y_test,"AC")
        
    ##print "#########################"
    print "SVM"
    it=[]
    acc=[]
    for estimator in range(2,49,2):
        #print "num of iterations: ",estimator
        it.append(estimator)
        clf = svm.SVC()
        clf.fit(X_train, Y_train)

        prediction=clf.predict(X_test)
        accuracy=accuracy_score(Y_test, prediction)
        #print "accuracy = ",accuracy
        acc.append(accuracy)
        #confusion=confusion_matrix(Y_test, prediction)
        #print "labels = ",labels
    #print "confusion matrix = ",confusion
    #print acc
    #plt.plot(it,acc,label="SVM")
    #SVM.append(max(acc))
    SVM=max(acc)
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
        #confusion=confusion_matrix(Y_test, prediction)
        #print "labels = ",labels
    #print "confusion matrix = ",confusion
    #print acc
    #plt.plot(it,acc,label="Logistic Regession")
    #LR.append(max(acc))
    LR=max(acc)
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
        #confusion=confusion_matrix(Y_test, prediction)
        #print "labels = ",labels
    #print "confusion matrix = ",confusion
    #print acc
    #plt.plot(it,acc,label="Random Forest")
    #RF.append(max(acc))
    RF=max(acc)
    #plotFeatures(prediction,Y_test,"RF")
    ##plt.ylim(ymin=0.5,ymax=1)
    #plt.xlabel("num of estimators")
    #plt.ylabel("accuracy")
    #plt.title(str(states[s])+" states")
    #plt.legend(loc='lower right')
    
    #plt.show()
    return AC, SVM, LR, RF





def diff(baby, mom):
    differences = []
    for i in range(len(baby)):
    	temp=[]
    	temp.append(baby[i][0])
        babyC = baby[i][1]
        momC = mom[i][1]
        temp.append(abs(babyC-momC))
        differences.append(temp)
    return differences

def selfDiff(data):
 	new_data=[[0,0]]
 	for i in range(1,len(data)):
 		temp=[data[i][0]]
 		temp.append(data[i][1]-data[i-1][1])
 		new_data.append(temp)

 	return new_data

def calculateCorrelation(mom,baby):
    seconds=0
    i=0 #mom
    j=0 #baby

    new_data=[]
    buffMom=[]
    buffBaby=[]

    while i<len(mom) and j<len(baby):
        if mom[i][0]==seconds or baby[j][0]==seconds:
            if mom[i][0]==seconds:
                buffMom.append(mom[i][1])
                i+=1
            if baby[j][0]==seconds:
                buffBaby.append(baby[j][1])
                j+=1

        else:
            if len(buffMom)!=0 and len(buffBaby)!=0:
                minLength=min(len(buffMom),len(buffBaby))
                buffMom=buffMom[0:minLength]
                buffBaby=buffBaby[0:minLength]
                new_data.append([seconds,pearsonr(buffMom, buffBaby)[0]])
                buffMom=[]
                buffBaby=[]
            else:
                new_data.append([seconds,0])

            seconds=mom[i][0]

    if len(buffMom)!=0 and len(buffBaby)!=0:
        while i<len(mom) and mom[i][0]==seconds:
            buffMom.append(mom[i][1])
            i+=1
        while j<len(baby) and baby[j][0]==seconds:
            buffBaby.append(baby[j][1])
            j+=1
        new_data.append([seconds,np.mean(np.correlate(buffMom, buffBaby))])


    return new_data

def calculateKLDivergence(mom,baby):
    seconds=mom[0][0]
    i=0 #mom
    j=0 #baby

    new_data=[]
    buffMom=[]
    buffBaby=[]

    while i<len(mom) and j<len(baby):
        if mom[i][0]==seconds or baby[j][0]==seconds:
            if mom[i][0]==seconds:
                buffMom.append(mom[i][1])
                i+=1
            if baby[j][0]==seconds:
                buffBaby.append(baby[j][1])
                j+=1

        else:
            #minLength=min(len(buffMom),len(buffBaby))
            #buffMom=buffMom[0:minLength]
            #buffBaby=buffBaby[0:minLength]

            if len(buffMom)!=0 and len(buffBaby)!=0:
                hisMom=np.histogram(buffMom, bins=np.arange(0,5.1,0.2))[0]
                hisBaby=np.histogram(buffBaby,bins=np.arange(0,5.1,0.2))[0]
                
                hisMomPro=[float(data)/sum(hisMom) for data in hisMom]
                hisBabyPro=[float(data)/sum(hisBaby) for data in hisBaby]

                for k in range(len(hisMomPro)):
                    if hisMomPro[k]==0:
                        hisMomPro[k]=1e-5
                for k in range(len(hisBabyPro)):
                    if hisBabyPro[k]==0:
                        hisBabyPro[k]=1e-5

                new_data.append([seconds,entropy(hisMomPro, qk=hisBabyPro, base=None)])

                #new_data.append([seconds,pearsonr(buffMom, buffBaby)[0]])
                buffMom=[]
                buffBaby=[]
                hisBaby=[]
                hisMom=[]
                hisBabyPro=[]
                hisMomPro=[]

            else:
                new_data.append([seconds,0])


            seconds=mom[i][0]


    while i<len(mom) and mom[i][0]==seconds:
        buffMom.append(mom[i][1])
        i+=1
    while j<len(baby) and baby[j][0]==seconds:
        buffBaby.append(baby[j][1])
        j+=1

    if len(buffMom)!=0 and len(buffBaby)!=0:
        hisMom=np.histogram(buffMom, bins=np.arange(0,5.1,0.2))[0]
        hisBaby=np.histogram(buffBaby,bins=np.arange(0,5.1,0.2))[0]

        
        hisMomPro=[float(data)/sum(hisMom) for data in hisMom]
        hisBabyPro=[float(data)/sum(hisBaby) for data in hisBaby]

        for k in range(len(hisMomPro)):
            if hisMomPro[k]==0:
                hisMomPro[k]=1e-5
        for k in range(len(hisBabyPro)):
            if hisBabyPro[k]==0:
                hisBabyPro[k]=1e-5
        new_data.append([seconds,entropy(hisMomPro, qk=hisBabyPro, base=None)])
    #new_data.append([seconds,np.mean(np.correlate(buffMom, buffBaby))])
    return new_data


def calculateVariance(data):
    seconds=data[0][0]

    new_data=[]
    buff=[]

    for i in range(len(data)):
        if data[i][0]==seconds:
            buff.append(data[i][1])
        else:
            if len(buff)>1:
                temp=np.var(buff)
            else:
                temp=buff[0]
            new_data.append([seconds,temp])
            buff=[data[i][1]]
            seconds=data[i][0]

    return new_data



def forPlot(data):
	lst1, lst2 = [], []
	for el in data:
		lst1.append(el[0])
		lst2.append(el[1])
	#lst2 = savgol_filter(lst2, window_length=5, polyorder=1,mode='mirror')
	return lst1, lst2


def forLabels(data):
	global states_detail
	lst1, lst2 = [], []
	for el in data:
		lst1.append(el[0])
		lst2.append(el[1])

	states_detail=list(np.unique(lst2))
	for i in range(len(lst2)):
		lst2[i]=states_detail.index(lst2[i])
	return lst1, lst2

def normal(baby,mom):
    all_data=[]
    for i in range(len(baby)):
        all_data.append(baby[i][1])
    for i in range(len(mom)):
        all_data.append(mom[i][1])

    mean_value=np.mean(all_data)
    sd_value=np.std(all_data)

    for i in range(len(baby)):
        baby[i][1]=np.divide(np.subtract(baby[i][1],mean_value),sd_value)
    for i in range(len(mom)):
        mom[i][1]=np.divide(np.subtract(mom[i][1],mean_value),sd_value)
    return baby,mom

def hardCode(mom,baby):
    output=[]
    boundaryMom=1
    boundaryBaby=1
    output.append([mom[0][0],0])  #0 for holding
    output.append([mom[1][0],0])

    
    '''
    for i in range(1,len(mom)):
        if abs(mom[i][1]-mom[i-1][1])<=boundary and abs(baby[i][1]-baby[i-1][1])<=boundary:
            last=output[-1][1]
            output.append([i,last])
        elif baby[i][1]-baby[i-1][1]>boundary:
            output.append([i,1]) #1 for low to high altitude, picking up
        elif baby[i][1]-baby[i-1][1]<-boundary:
            output.append([i,2]) #2 for high to low altitude, 
        elif mom[i][1]-mom[i-1][1]>boundary and abs(baby[i][1]-baby[i-1][1])<=boundary:
            output.append([i,3]) #(mom stand up alone- shift to nearby state)
        elif mom[i][1]-mom[i-1][1]<-boundary and abs(baby[i][1]-baby[i-1][1])<=boundary:
            output.append([i,4]) #(mom sits to/ grabs baby -- could be holding or hovering)
        print mom[i][1],mom[i-1][1],output[i]
    '''

    for i in range(2,len(mom)):

        if mom[i][0]-1==mom[i-1][0] and mom[i-1][0]-1==mom[i-2][0]:
            momData=[mom[i][1],mom[i-1][1],mom[i-2][1]]
            babyData=[baby[i][1],baby[i-1][1],baby[i-2][1]]
            if abs(max(momData)-min(momData))<=boundaryMom and abs(max(babyData)-min(babyData))<=boundaryBaby:
                last=output[-1][1]
                output.append([mom[i][0],last])
            elif max(babyData)-min(babyData)>boundaryBaby and babyData.index(max(babyData))-babyData.index(min(babyData))>0:
                output.append([mom[i][0],1]) #1 for low to high altitude, picking up
            elif max(babyData)-min(babyData)>boundaryBaby and babyData.index(max(babyData))-babyData.index(min(babyData))<0:
                output.append([mom[i][0],2]) #2 for high to low altitude, 
            elif max(momData)-min(momData)>boundaryMom and momData.index(max(momData))-momData.index(min(momData))>0 and abs(max(babyData)-min(babyData))<=boundaryBaby:
                output.append([mom[i][0],3]) #(mom stand up alone- shift to nearby state)
            elif max(momData)-min(momData)>boundaryMom and momData.index(max(momData))-momData.index(min(momData))<0 and abs(max(babyData)-min(babyData))<=boundaryBaby:
                output.append([mom[i][0],4]) #(mom sits to/ grabs baby -- could be holding or hovering)
        else:
            output.append([mom[i][0],0])

    return output

def plotHardCodeFeatures(filteredMom,filteredBaby,hardCodedPressure,labels):
    fig = plt.figure(figsize=(20, 10))

    elan=fig.add_subplot(313)
    elan.plot(forLabels(labels)[0],forLabels(labels)[1])
    elan.set_ylabel("labels",fontsize=small_font_size)
    elan.set_xlabel('time (s)',fontsize=small_font_size)


    
    ori_labels = [item for item in elan.get_yticklabels()]
    global states_detail
    #for i in range(len(states_detail)):
    #    ori_labels[i]=states_detail[i] 
    
    ori_labels[1]='carrying'
    ori_labels[2]='holding'
    #ori_labels[3]='holding bouncing'
    ori_labels[3]='hovering'
    #ori_labels[5]='nearby'
    ori_labels[4]='picking up'
    ori_labels[5]='putting down'
    ori_labels[6]='touching'
    


    print states_detail
    elan.set_yticklabels(ori_labels)
    


    prefilter = fig.add_subplot(311,sharex=elan)
    prefilter.plot(forPlot(filteredMom)[0],forPlot(filteredMom)[1],'purple')
    prefilter.plot(forPlot(filteredBaby)[0],forPlot(filteredBaby)[1],'green')
    prefilter.set_ylabel('Altitude',fontsize=small_font_size)
    plt.setp(prefilter.get_xticklabels(),visible=False)


    hard=fig.add_subplot(312,sharex=elan)
    hard.plot(forPlot(hardCodedPressure)[0],forPlot(hardCodedPressure)[1],'purple')
    hard.set_ylabel('Hard Coded Pressure',fontsize=small_font_size)
    plt.setp(hard.get_xticklabels(),visible=False)
    hard.yaxis.tick_right()
    hard.yaxis.set_label_position("right")


    fig.subplots_adjust(hspace=0)


    plt.show()







#def plotFeatures(rawPreMom,rawPreBaby,preMomVariance,preBabyVariance,preDifference,selfAccDiffMom,selfPreDiffMom,preCorrelation,labels):
#def plotFeatures(rawAccMom,rawAccBaby,accMomVariance,accBabyVariance,accDifference,selfAccDiffMom,selfPreDiffBaby,accCorrelation,labels):
def plotFeatures(rawPreMom,rawPreBaby,filteredMom,filteredBaby,preMomVariance,preBabyVariance,preDifference,selfPreDiffMom,selfPreDiffBaby,preCorrelation,labels):
    fig = plt.figure(figsize=(20, 10))

    elan=fig.add_subplot(717)
    elan.plot(forLabels(labels)[0],forLabels(labels)[1])
    elan.set_ylabel("labels",fontsize=small_font_size)
    elan.set_xlabel('time (s)',fontsize=small_font_size)

    ori_labels = [item for item in elan.get_yticklabels()]

    #print ori_labels[0],ori_labels[1],ori_labels[2],ori_labels[3]

    global states_detail
    #for i in range(len(states_detail)):
    #    ori_labels[i]=states_detail[i]
    #ori_labels[0]='carrying'
    ori_labels[1]='holding'
    #ori_labels[2]='hovering'
    ori_labels[3]='nearby'

    print states_detail
    elan.set_yticklabels(ori_labels)


    

    
    preDiff = fig.add_subplot(714,sharex=elan)
    preDiff.plot(forPlot(preDifference)[0],forPlot(preDifference)[1],'purple')
    #acc.text(data_time[len(data_time)-1], data_acc[len(data_acc)-1], "mother", fontsize=small_font_size, color='purple')
    preDiff.set_xlabel('time (s)',fontsize=small_font_size)
    preDiff.set_ylabel('Absolute Pressure Difference',fontsize=small_font_size)
    preDiff.yaxis.tick_right()
    preDiff.yaxis.set_label_position("right")
    plt.setp(preDiff.get_xticklabels(),visible=False)

    

    '''
    acc = fig.add_subplot(614)
    acc.plot(forPlot(accBabyData)[0],forPlot(accBabyData)[1],'green')
    acc.plot(forPlot(accMomData)[0],forPlot(accMomData)[1],'purple')
    #acc.text(data_time[len(data_time)-1], data_acc[len(data_acc)-1], "mother", fontsize=small_font_size, color='purple')
    acc.set_xlabel('time (s)',fontsize=small_font_size)
    acc.set_ylabel('Motion data\n(average of 3D axes)',fontsize=small_font_size)
    
    
    

    pre = fig.add_subplot(612,sharex=elan)
    pre.plot(forPlot(pressureMomData)[0],forPlot(pressureMomData)[1],'purple')
    pre.plot(forPlot(pressureBabyData)[0],forPlot(pressureBabyData)[1],'green')
    pre.set_ylabel('Altitude\n feet',fontsize=small_font_size)
    
    
    '''
    rawPre = fig.add_subplot(711,sharex=elan)
    rawPre.plot(forPlot(rawPreMom)[0],forPlot(rawPreMom)[1],'purple')
    rawPre.plot(forPlot(rawPreBaby)[0],forPlot(rawPreBaby)[1],'green')
    rawPre.set_ylabel('raw pressure',fontsize=small_font_size)
    plt.setp(rawPre.get_xticklabels(),visible=False)


    '''
    rawAcc=fig.add_subplot(611,sharex=elan)
    rawAcc.plot(forPlot(rawAccBaby)[0],forPlot(rawAccBaby)[1],'green')
    rawAcc.plot(forPlot(rawAccMom)[0],forPlot(rawAccMom)[1],'purple')
    rawAcc.set_ylabel('raw acc data',fontsize=small_font_size)
    plt.setp(rawAcc.get_xticklabels(),visible=False)
    rawAcc.yaxis.tick_right()
    rawAcc.yaxis.set_label_position("right")


    
    polar=fig.add_subplot(611,sharex=acc)
    polar.plot(forPlot(hrBabyData)[0],forPlot(hrBabyData)[1],'green')
    polar.plot(forPlot(hrMomData)[0],forPlot(hrMomData)[1],'purple')
    polar.set_ylabel('Beats per minute',fontsize=small_font_size)
    
    

    
    selfAccDiff=fig.add_subplot(615,sharex=elan)
    selfAccDiff.plot(forPlot(selfAccDiffMom)[0],forPlot(selfAccDiffMom)[1],'purple')
    selfAccDiff.plot(forPlot(selfAccDiffBaby)[0],forPlot(selfAccDiffBaby)[1],'green')
    selfAccDiff.set_ylabel('acc derivative',fontsize=small_font_size)
    plt.setp(selfAccDiff.get_xticklabels(),visible=False)
    selfAccDiff.yaxis.tick_right()
    selfAccDiff.yaxis.set_label_position("right")
    '''

    selfPreDiff=fig.add_subplot(715,sharex=elan)
    selfPreDiff.plot(forPlot(selfPreDiffMom)[0],forPlot(selfPreDiffMom)[1],'purple')
    selfPreDiff.plot(forPlot(selfPreDiffBaby)[0],forPlot(selfPreDiffBaby)[1],'green')
    selfPreDiff.set_ylabel('pressure derivative',fontsize=small_font_size)
    plt.setp(selfPreDiff.get_xticklabels(),visible=False)


    '''

    accC=fig.add_subplot(613,sharex=elan)
    accC.plot(forPlot(accCorrelation)[0],forPlot(accCorrelation)[1],'purple')
    accC.set_ylabel('acc KL divergence',fontsize=small_font_size)
    plt.setp(accC.get_xticklabels(),visible=False)
    accC.yaxis.tick_right()
    accC.yaxis.set_label_position("right")
    '''

    preC=fig.add_subplot(713,sharex=elan)
    preC.plot(forPlot(preCorrelation)[0],forPlot(preCorrelation)[1],'purple')
    preC.set_ylabel('Pressure Correlation',fontsize=small_font_size)
    plt.setp(preC.get_xticklabels(),visible=False)
    '''

    accVariance=fig.add_subplot(612,sharex=elan)
    accVariance.plot(forPlot(accMomVariance)[0],forPlot(accMomVariance)[1],'purple')
    accVariance.plot(forPlot(accBabyVariance)[0],forPlot(accBabyVariance)[1],'green')
    accVariance.set_ylabel('acc Variance',fontsize=small_font_size)
    plt.setp(accVariance.get_xticklabels(),visible=False)



    accDiff=fig.add_subplot(614,sharex=elan)
    accDiff.plot(forPlot(accDifference)[0],forPlot(accDifference)[1],'purple')
    accDiff.set_ylabel("absolute acc Difference",fontsize=small_font_size)
    plt.setp(accDiff.get_xticklabels(),visible=False)
    '''
    preVariance=fig.add_subplot(716,sharex=elan)
    preVariance.plot(forPlot(preMomVariance)[0],forPlot(preMomVariance)[1],'purple')
    preVariance.plot(forPlot(preBabyVariance)[0],forPlot(preBabyVariance)[1],'green')
    preVariance.set_ylabel('pre Variance',fontsize=small_font_size)
    preVariance.yaxis.tick_right()
    preVariance.yaxis.set_label_position("right")
    plt.setp(preVariance.get_xticklabels(),visible=False)
    
    

    prefilter = fig.add_subplot(712,sharex=elan)
    prefilter.plot(forPlot(filteredMom)[0],forPlot(filteredMom)[1],'purple')
    prefilter.plot(forPlot(filteredBaby)[0],forPlot(filteredBaby)[1],'green')
    prefilter.set_ylabel('filtered pressure',fontsize=small_font_size)
    prefilter.yaxis.tick_right()
    prefilter.yaxis.set_label_position("right")
    plt.setp(prefilter.get_xticklabels(),visible=False)
    
    

    
    


    #plt.setp(pre.get_xticklabels(), visible=False)
    #plt.setp(rawAcc.get_xticklabels(), visible=False)
    #plt.setp(selfAccDiff.get_xticklabels(),visible=False)
   
    plt.title('Nov 11',fontsize=small_font_size,loc="center")
    fig.subplots_adjust(hspace=0)


    plt.show()


if __name__ == '__main__':
    files = os.listdir(data_folder)
    

    for f in files:
        if 'Acc' in f:
            if 'Mom' in f:
                rawAccMom=read_rawdatafile(data_folder+str(f))
                accMomData = read_datafile(data_folder + str(f))      
            else:   
                accBabyData = read_datafile(data_folder + str(f))
                rawAccBaby=read_rawdatafile(data_folder+str(f))
		
        elif 'Barometer' in f:
            if 'Mom' in f:
                rawPreMom=read_rawdatafile(data_folder+str(f))
                pressureMomData = read_datafile(data_folder + str(f))
            else:
                rawPreBaby=read_rawdatafile(data_folder+str(f))
                pressureBabyData = read_datafile(data_folder + str(f))
        elif 'Polar' in f:
			if 'Mom' in f:
				hrMomData = read_datafile(data_folder + str(f))
			else:
				hrBabyData = read_datafile(data_folder + str(f))
        elif 'ManuallyCleanMultiStates' in f:
		    labels = read_labelfile(data_folder + str(f))

    accCorrelation=calculateKLDivergence(accMomData,accBabyData)
    preCorrelation=calculateCorrelation(pressureMomData,pressureBabyData)


    '''
    baroOffset=pressureMomData[0][1]-pressureBabyData[0][1]
    for i in range(len(pressureBabyData)):
        pressureBabyData[i][1]+=baroOffset
    
    pressureMomData=np.array(pressureMomData)
    pressureBabyData=np.array(pressureBabyData)
    pressureMomData[:,1]=savgol_filter(pressureMomData[:,1],window_length=25,polyorder=1,mode='mirror')
    pressureBabyData[:,1]=savgol_filter(pressureBabyData[:,1],window_length=25,polyorder=1,mode='mirror')
    '''



    baroOffset=rawPreMom[0][1]-rawPreBaby[0][1]
    for i in range(len(rawPreBaby)):
        rawPreBaby[i][1]+=baroOffset



    rawPreMom=np.array(rawPreMom)
    rawPreBaby=np.array(rawPreBaby)

    filteredMom=rawPreMom.copy()
    filteredBaby=rawPreBaby.copy()
    #rawPreBaby[:,1]= savgol_filter(rawPreBaby[:,1], window_length=101, polyorder=1,mode='mirror')
    #rawPreMom[:,1]=savgol_filter(rawPreMom[:,1],window_length=101,polyorder=1,mode='mirror')

    #y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
    #rawPreBaby[:,1]= butter_bandpass_filter(rawPreBaby[:,1],2000,6000,30,order=4)
    #rawPreMom[:,1]=butter_bandpass_filter(rawPreMom[:,1],2000,6000,30,order=4)

    f_signal = rfft(filteredBaby[:,1])
    #W = fftfreq(rawPreBaby[:,1], 25)
    # If our original signal time was in seconds, this is now in Hz    
    cut_f_signal = f_signal.copy()
    #cut_f_signal[0:1] = 0
    cut_f_signal[200:-1]=0

    filteredBaby[:,1] = irfft(cut_f_signal)

    f_signal = rfft(filteredMom[:,1])

    # If our original signal time was in seconds, this is now in Hz    
    cut_f_signal = f_signal.copy()
    print len(cut_f_signal)
    #cut_f_signal[0:1] = 0
    cut_f_signal[200:-1]=0

    filteredMom[:,1] = irfft(cut_f_signal)


    filteredPreMom=filteredMom.copy()
    filteredPreBaby=filteredBaby.copy()
    for i in range(len(filteredPreMom)):
        filteredPreMom[i][0]=int(filteredPreMom[i][0])
    for i in range(len(filteredPreBaby)):
        filteredPreBaby[i][0]=int(filteredPreBaby[i][0])



    '''
    pressureMomData=np.array(pressureMomData)
    pressureBabyData=np.array(pressureBabyData)
    pressureBabyData[:,1]= savgol_filter(pressureBabyData[:,1], window_length=21, polyorder=1,mode='mirror')
    pressureMomData[:,1]=savgol_filter(pressureMomData[:,1],window_length=21,polyorder=1,mode='mirror')
    preCorrelation=calculateCorrelation(pressureMomData,pressureBabyData)
    '''



    accMomVariance=calculateVariance(accMomData)
    accBabyVariance=calculateVariance(accBabyData)
    preMomVariance=calculateVariance(pressureMomData)
    preBabyVariance=calculateVariance(pressureBabyData)

    


    accMomData=combineIntoSeconds(accMomData)
    accBabyData=combineIntoSeconds(accBabyData)
    pressureMomData=combineIntoSeconds(pressureMomData)
    pressureBabyData=combineIntoSeconds(pressureBabyData)
    filteredPreMom=combineIntoSeconds(filteredPreMom)
    filteredPreBaby=combineIntoSeconds(filteredPreBaby)

    #minLength = np.min([len(accBabyData), len(accMomData), len(pressureBabyData), len(pressureMomData), len(labels),len(hrBabyData),len(hrMomData)])
    #minLength=np.min([len(accBabyData),len(accMomData),len(labels)])
    minLength=np.min([len(accBabyData), len(accMomData),len(pressureBabyData),len(pressureMomData),len(labels)])


    accBabyData=accBabyData[0:minLength]
    accMomData=accMomData[0:minLength]
    pressureMomData=pressureMomData[0:minLength]
    pressureBabyData=pressureBabyData[0:minLength]
    #hrBabyData=hrBabyData[0:minLength]
    #hrMomData=hrMomData[0:minLength]
    labels=labels[0:minLength]
    accCorrelation=accCorrelation[0:minLength]
    preCorrelation=preCorrelation[0:minLength]
    accMomVariance=accMomVariance[0:minLength]
    accBabyVariance=accBabyVariance[0:minLength]
    preMomVariance=preMomVariance[0:minLength]
    preBabyVariance=preBabyVariance[0:minLength]
    filteredPreMom=filteredPreMom[0:minLength]
    filteredPreBaby=filteredPreBaby[0:minLength]

    for data in filteredPreBaby:
        data[1]=300 + 30 * (1013 - data[1]/100)

    for data in filteredPreMom:
        data[1]=300 + 30 * (1013 - data[1]/100)



    hardCodedPressure=hardCode(filteredPreMom,filteredPreBaby)


    print np.array(hardCodedPressure).shape


	#plotFeatures(accBabyData,accMomData,pressureBabyData,pressureMomData,hrBabyData,hrMomData)
    #accBabyData,accMomData=normal(accBabyData,accMomData)
    #pressureBabyData,pressureMomData=normal(pressureBabyData,pressureMomData)
    #hrBabyData,hrMomData=normal(hrBabyData,hrMomData)

    accDifference=diff(accBabyData,accMomData)
    preDifference=diff(pressureBabyData,pressureMomData)

    
    selfAccDiffMom=selfDiff(accMomData)
    selfAccDiffBaby=selfDiff(accBabyData)

    selfPreDiffMom=selfDiff(pressureMomData)
    selfPreDiffBaby=selfDiff(pressureBabyData)

    #plotFeatures(rawPreMom,rawPreBaby,filteredPreMom,filteredPreBaby,preMomVariance,preBabyVariance,preDifference,selfPreDiffMom,selfPreDiffBaby,preCorrelation,labels)
    #plotHardCodeFeatures(filteredPreMom,filteredPreBaby,hardCodedPressure,labels)
    #plotFeatures(rawPreMom,rawPreBaby,filteredMom,filteredBaby,preDifference,selfAccDiffMom,selfPreDiffMom,preCorrelation,labels)

    #plotFeatures(rawAccMom,rawAccBaby,accMomVariance,accBabyVariance,accDifference,selfAccDiffMom,selfPreDiffBaby,accCorrelation,labels)

    
    
    #print np.array(hardCodedPressure)
    #print np.array(accMomVariance)
    #print np.array(accBabyVariance)
    #print np.array(preMomVariance)
    #print np.array(preBabyVariance)
    #print np.array(accCorrelation)
    #print np.array(accDifference)
    #print np.array(preDifference)
    #print np.array(selfAccDiffMom)
    #print np.array(selfAccDiffBaby)
    #print np.array(selfPreDiffMom)
    #print np.array(selfPreDiffBaby).shape
    #print np.array(preCorrelation).shape
    #print np.array(filteredPreMom).shape
    #print np.array(filteredPreBaby).shape
    

    features=np.concatenate(([np.array(hardCodedPressure)[:,1]],[np.array(accMomVariance)[:,1]],[np.array(accBabyVariance)[:,1]],[np.array(preMomVariance)[:,1]],[np.array(preBabyVariance)[:,1]],[np.array(accCorrelation)[:,1]], \
        [np.array(accDifference)[:,1]],[np.array(preDifference)[:,1]],[np.array(selfAccDiffMom)[:,1]],[np.array(selfAccDiffBaby)[:,1]],[np.array(selfPreDiffMom)[:,1]], \
        [np.array(selfPreDiffBaby)[:,1]],[np.array(preCorrelation)[:,1]],[np.array(filteredPreMom)[:,1]],[np.array(filteredPreBaby)[:,1]]),axis=0)

    
    
    
    features=np.transpose(features)
    AC=[]
    RF=[]
    LR=[]
    SVM=[]
    states=[]
    
    for i in range(1,len(labels[0])):
        tAC=[]
        tRF=[]
        tLR=[]
        tSVM=[]

        
        numOfStates=len(np.unique(np.array(labels)[:,i]))
        print np.unique(np.array(labels)[:,i])

        newlabels=np.array(labels)[:,i]
        states.append(numOfStates)
    

        kf = KFold(n_splits=5)

        for train_index, test_index in kf.split(features):
            X_train, X_test = features[train_index], features[test_index]
            Y_train, Y_test = newlabels[train_index], newlabels[test_index]
            #print "test_index",test_index
            if len(np.unique(Y_train))>1:

                ac, svmm, lr, rf=ml(X_train, X_test,Y_train,Y_test)
                tAC.append(ac)
                tSVM.append(svmm)
                tLR.append(lr)
                tRF.append(rf)
        AC.append(np.mean(tAC))
        SVM.append(np.mean(tSVM))
        LR.append(np.mean(tLR))
        RF.append(np.mean(tRF))

        print tAC,tSVM,tLR,tRF

        print "ac variance ",np.std(tAC)
        print "svm variance ",np.std(tSVM)
        print "lr variance ",np.std(tLR)
        print "rf variance ",np.std(tRF)



    plt.plot(states,LR,label="Logistic Regression")
    plt.plot(states,RF,label="Random Forest")
    plt.plot(states,SVM,label="Support Vector Machine")
    plt.plot(states,AC,label="Adaboost Classifier")

    print states
    print LR
    print RF
    print SVM
    print AC


    plt.ylim(ymin=0,ymax=1)


    plt.xlabel("num of states")
    plt.ylabel("Accuracy")
    plt.title("Oct 28")

    plt.legend(loc='upper right')
    plt.show()
    
    

    #X_train,X_test,Y_train,Y_test=train_test_split(features,labels,test_size=0.2,random_state=1)
    #ml(X_train,X_test,Y_train,Y_test)
    
    

