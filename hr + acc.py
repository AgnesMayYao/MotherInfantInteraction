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



data_folder='data/Nov11/'
feature_folder="users/labels/"
graph_title="P1 (Nov. 11)"
result_file_name=feature_folder+graph_title+".csv"
states=[]
states_detail=[]
baroOffset=0
small_font_size=10
large_font_size=20



def find_index(data,small,large):
    output_small=0
    output_large=0
    for i in range(len(data)):
        if data[i][0]==small:
            output_small=i
        if data[i][0]==large:
            output_large=i
    return output_small,output_large


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

    print "row no:",row_no

    return data

def ml(X_train, X_test,Y_train, Y_test):

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
    global real,guess
    real.extend(Y_test)
    guess.extend(prediction)
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
 	new_data=[[data[0][0],0]]
 	for i in range(1,len(data)):
 		temp=[data[i][0]]
 		temp.append(data[i][1]-data[i-1][1])
 		new_data.append(temp)

 	return new_data

def calculateCorrelation(mom,baby):
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
            if len(buffMom)>1 and len(buffBaby)>1:
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
                temp=0
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



    available_states = set(list(np.array(data)[:,1]))

    states = ['holding', 'nearby', 'picking up/ putting down', 'picking up', 'putting down','hovering', 'holding still', 'holding walking']

    #states = ['nearby', 'hovering','touching','picking up', 'holding', 'carrying','bouncing','putting down', 'uncarrying']
    states = [x for x in states if x in available_states]

    global states_detail
    lst1,lst2=[],[]
    for el in data:
        lst1.append(el[0])
        lst2.append(states.index(el[1]))
    
    return lst1,lst2,states

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
    boundaryMom=1.5
    boundaryBaby=1.5
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
    #print baby[0][0], baby[-1][0], len(baby)
    #print mom[0][0], mom[-1][0], len(mom)
    for i in range(len(baby)):
        if baby[i][0]!=mom[i][0]:
           print baby[i][0], mom[i][0]
    for i in range(2,len(mom)):
        if mom[i][0]-1==mom[i-1][0] and mom[i-1][0]-1==mom[i-2][0] :
            momData=[mom[i][1],mom[i-1][1],mom[i-2][1]]
            babyData=[baby[i][1],baby[i-1][1],baby[i-2][1]]
            if abs(max(momData)-min(momData))<=boundaryMom and abs(max(babyData)-min(babyData))<=boundaryBaby:
                last=output[-1][1]
                output.append([mom[i][0],last])
            elif max(babyData)-min(babyData)>boundaryBaby and babyData.index(max(babyData))-babyData.index(min(babyData))>0:
                output.append([mom[i][0],2]) #1 for low to high altitude, picking up
            elif max(babyData)-min(babyData)>boundaryBaby and babyData.index(max(babyData))-babyData.index(min(babyData))<0:
                output.append([mom[i][0],1]) #2 for high to low altitude, 
            elif max(momData)-min(momData)>boundaryMom and momData.index(max(momData))-momData.index(min(momData))>0 and abs(max(babyData)-min(babyData))<=boundaryBaby:
                output.append([mom[i][0],4]) #(mom stand up alone- shift to nearby state)
            elif max(momData)-min(momData)>boundaryMom and momData.index(max(momData))-momData.index(min(momData))<0 and abs(max(babyData)-min(babyData))<=boundaryBaby:
                output.append([mom[i][0],3]) #(mom sits to/ grabs baby -- could be holding or hovering)
        else:
            output.append([mom[i][0],0])

    return output

def plotHardCodeFeatures(filteredMom,filteredBaby,hardCodedPressure,labels):
    fig = plt.figure(figsize=(20, 6.5))

    elan=fig.add_subplot(313)
    elan.plot(forLabels(labels)[0],forLabels(labels)[1])
    elan.set_ylabel("labels",fontsize=small_font_size)
    elan.set_xlabel('time (s)',fontsize=small_font_size)


   

    #ori_labels = [item for item in elan.get_yticklabels()]
    #global states_detail
    #for i in range(len(states_detail)):
    #    ori_labels[i]=states_detail[i] 
    
    #ori_labels[1]='nearby'
    #ori_labels[2]='holding'
    #ori_labels[3]='holding bouncing'
    #ori_labels[-1]='holding'
    #ori_labels[5]='nearby'
    #ori_labels[4]='picking up'
    #ori_labels[5]='putting down'
    #ori_labels[6]='touching'

    elan.yaxis.set_major_locator(ticker.FixedLocator([0,1,2]))
    elan.yaxis.set_major_formatter(ticker.FixedFormatter(['nearby','holding']))
    
    print states_detail
    #elan.set_yticklabels(ori_labels)
    


    prefilter = fig.add_subplot(311,sharex=elan)
    prefilter.plot(forPlot(filteredMom)[0],forPlot(filteredMom)[1],'purple')
    prefilter.plot(forPlot(filteredBaby)[0],forPlot(filteredBaby)[1],'green')
    prefilter.set_ylabel('Altitude',fontsize=small_font_size)
    plt.setp(prefilter.get_xticklabels(),visible=False)


    hard=fig.add_subplot(312,sharex=elan)
    hard.plot(forPlot(hardCodedPressure)[0],forPlot(hardCodedPressure)[1],'purple')
    hard.set_ylabel('Hard Coded Pressure',fontsize=small_font_size)
    plt.setp(hard.get_xticklabels(),visible=False)
    #hard.yaxis.tick_right()
    #hard.yaxis.set_label_position("right")
    '''
    hard_labels = [item for item in hard.get_yticklabels()]
    hard_labels[2]='low->high (both)'
    hard_labels[3]='high->low (both)'
    hard_labels[4]='low->high (mom)'
    hard_labels[5]='high->low (mom)'

    hard.set_yticklabels(hard_labels)
    '''

    hard.yaxis.set_major_locator(ticker.FixedLocator([1,2,3,4]))
    hard.yaxis.set_major_formatter(ticker.FixedFormatter(['low->high (both)','high->low (both)','low->high (mom)','high->low (mom)']))

    prefilter.set_title(graph_title,fontsize=12,loc="center")
    fig.subplots_adjust(hspace=0)


    plt.show()

def plotRawFeatures(rawPreMom,rawPreBaby,rawAccMom,rawAccBaby,hrMom, hrBaby, labels):
    fig = plt.figure(figsize=(25, 7))


    ordered_label = forLabels(labels)[2]
    num_states = len(ordered_label)
    elan=fig.add_subplot(414)
    elan.plot(forLabels(labels)[0],forLabels(labels)[1])
    elan.set_ylabel("labels",fontsize=small_font_size)
    elan.set_xlabel('time (s)',fontsize=small_font_size)
    print states_detail

    #elan.yaxis.set_major_locator(ticker.FixedLocator([0,1]))
    #elan.yaxis.set_major_formatter(ticker.FixedFormatter(['nearby','holding']))

    

    elan.yaxis.set_major_locator(ticker.FixedLocator(xrange(num_states)))
    elan.yaxis.set_major_formatter(ticker.FixedFormatter(ordered_label))





    rawPre = fig.add_subplot(411,sharex=elan)
    rawPre.plot(forPlot(rawPreMom)[0],forPlot(rawPreMom)[1],'purple')
    rawPre.plot(forPlot(rawPreBaby)[0],forPlot(rawPreBaby)[1],'green')
    rawPre.set_ylabel('raw altitude \n feet',fontsize=small_font_size)
    plt.setp(rawPre.get_xticklabels(),visible=False)
    rawPre.yaxis.set_label_position("right")
    rawPre.yaxis.tick_right()


    rawAcc=fig.add_subplot(412,sharex=elan)
    rawAcc.plot(forPlot(rawAccBaby)[0],forPlot(rawAccBaby)[1],'green')
    rawAcc.plot(forPlot(rawAccMom)[0],forPlot(rawAccMom)[1],'purple')
    rawAcc.set_ylabel('raw acc data \n m/s',fontsize=small_font_size)
    plt.setp(rawAcc.get_xticklabels(),visible=False)
    
    

    rawhr=fig.add_subplot(413,sharex=elan)
    rawhr.plot(forPlot(hrBaby)[0],forPlot(hrBaby)[1],'green')
    rawhr.plot(forPlot(hrMom)[0],forPlot(hrMom)[1],'purple')
    rawhr.set_ylabel('heart rate',fontsize=small_font_size)
    plt.setp(rawhr.get_xticklabels(),visible=False)
    rawhr.yaxis.tick_right()
    rawhr.yaxis.set_label_position("right")


    fig.subplots_adjust(hspace=0)
    rawPre.set_title(graph_title,fontsize=12,loc="center")
    #plt.suptitle('P2 (Jan. 17)',fontsize=12)

    plt.xlim([0, 175])

    plt.show()




    
#def plotFeatures(rawPreMom,rawPreBaby,preMomVariance,preBabyVariance,preDifference,selfAccDiffMom,selfPreDiffMom,preCorrelation,labels):
#def plotFeatures(rawAccMom,rawAccBaby,accMomVariance,accBabyVariance,accDifference,selfAccDiffMom,selfPreDiffBaby,accCorrelation,labels):
def plotFeatures(rawPreMom,rawPreBaby,filteredMom,filteredBaby,preMomVariance,preBabyVariance,preDifference,selfPreDiffMom,selfPreDiffBaby,preCorrelation,labels):
    fig = plt.figure(figsize=(20, 10))

    elan=fig.add_subplot(717)
    elan.plot(forLabels(labels)[0],forLabels(labels)[1])
    elan.set_ylabel("labels",fontsize=small_font_size)
    elan.set_xlabel('time (s)',fontsize=small_font_size)

    #ori_labels = [item for item in elan.get_yticklabels()]

    #print ori_labels[0],ori_labels[1],ori_labels[2],ori_labels[3]

    global states_detail
    #for i in range(len(states_detail)):
    #    ori_labels[i]=states_detail[i]
    #ori_labels[0]='carrying'
    #ori_labels[1]='holding'
    #ori_labels[2]='hovering'
   #ori_labels[3]='nearby'

    print states_detail
    #elan.set_yticklabels(ori_labels)


    

    
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
   
    #plt.title('Nov 11',fontsize=small_font_size,loc="center")
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


    plotRawFeatures(rawPreMom,rawPreBaby,rawAccMom,rawAccBaby,hrMomData, hrBabyData, labels)
