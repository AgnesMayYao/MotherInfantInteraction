import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
import pickle

def simplify_acc(temp_list):
        simplified_list=[]
        count_acc=temp_list[0][1]
        cur_time=temp_list[0][0]
        count=1
        for i in range(1,len(temp_list)):
            if temp_list[i][0]==cur_time:
                count_acc+=temp_list[i][1]
                count+=1
            else:
                simplified_list.append([cur_time,count_acc/count])
                cur_time=temp_list[i][0]
                count=1
                count_acc=temp_list[i][1]
        
        
        return simplified_list
    
def combine(mom_list,baby_list):
        result_list=[]
        mom=0
        baby=0
        while mom<len(mom_list) and baby<len(baby_list):
            if mom_list[mom][0]>baby_list[baby][0]:
                baby+=1
            elif mom_list[mom][0]<baby_list[baby][0]:
                mom+=1
            else:
                result_list.append([mom_list[mom][0],mom_list[mom][1],baby_list[baby][1]])
                baby+=1
                mom+=1
        return result_list
    


if __name__ == '__main__':
   
    hold_carry_results=[]
    pupd_results=[]
    nearby_results=[]
    
    timestamp=[]
    timestamp_file=open("timestamp_Oct28.csv","r")
    timestamp_reader=csv.reader(timestamp_file)
    
    mom_simplified=[]
    baby_simplified=[]
    
    #get the timestamp file 
    #start time, end time, behavior 
    #start time and end time have 13 digits (ms)
    for row in timestamp_reader:
        time1=round(float(row[0])/1000)
        time2=round(float(row[1])/1000)
        behaviour=row[2]
        
        timestamp.append([time1,time2,behaviour])
    timestamp_file.close()  
    timestamp=sorted(timestamp, key=lambda t:t[0])
    
    #print timestamp
    
    pre_filename=["Mom_2016-10-28T16.16.08.276_D4D7C843B379_Pressure.csv", \
    "Baby_2016-10-28T16.16.08.276_FB7D0BFC0351_Pressure.csv"]
       
    for i in range(0,2):
        pre_file=open(pre_filename[i],"r")
        pre_reader = csv.reader(pre_file)
        row_no=1
        pre_time=0
        
        for row in pre_reader:
            if row_no>1:
                temp_time=round(float(row[0])/1000)
                
                pre=float(row[3])
                
                if i==0:
                    mom_simplified.append([temp_time,pre])
                else:
                    baby_simplified.append([temp_time,pre])
            row_no+=1
            
    pre_file.close()
            
    mom_pre_simplify=simplify_acc(mom_simplified)
    baby_pre_simplify=simplify_acc(baby_simplified)
    
    combined_mom_baby_pre=combine(mom_pre_simplify,baby_pre_simplify)
    
    mom_simplified=[]
    baby_simplified=[]

    acc_filename=["Mom_2016-10-28T16.16.08.276_D4D7C843B379_Accelerometer.csv", \
    "Baby_2016-10-28T16.16.08.276_FB7D0BFC0351_Accelerometer.csv"]
       
    for i in range(0,2):
        acc_file=open(acc_filename[i],"r")
        acc_reader = csv.reader(acc_file)
        row_no=1
        pre_time=0
        
        for row in acc_reader:
            if row_no>1:
                temp_time=round(float(row[0])/1000)
                
                mean_acc=(abs(float(row[3]))+abs(float(row[4]))+abs(float(row[5])))/3
                #temp_dist=0.5*abs(mean_acc)*(float(row[2])-pre_time)*(float(row[2])-pre_time)
                temp_dist=mean_acc
                pre_time=float(row[2])
                
                if i==0:
                    mom_simplified.append([temp_time,temp_dist])
                else:
                    baby_simplified.append([temp_time,temp_dist])
            row_no+=1
            
    acc_file.close()
            
    mom_acc_simplify=simplify_acc(mom_simplified)
    baby_acc_simplify=simplify_acc(baby_simplified)
    
    combined_mom_baby_acc=combine(mom_acc_simplify,baby_acc_simplify)
    
    
    
    
    
    
    
    
    
    #forJatin=[]
    
    pointer=0
    for i in range(0,len(combined_mom_baby_acc)):
        if combined_mom_baby_acc[i][0]<timestamp[pointer][0]:
            continue
        if combined_mom_baby_acc[i][0]>=timestamp[pointer][0] and combined_mom_baby_acc[i][0]<=timestamp[pointer][1]:
            if timestamp[pointer][2].rstrip().lower()=="hold/carry":
                hold_carry_results.append([combined_mom_baby_acc[i][1],combined_mom_baby_acc[i][2],combined_mom_baby_pre[i][1],combined_mom_baby_pre[i][2]])
                #forJatin.append([combined_mom_baby[i][0],"hold/carry"])
            elif timestamp[pointer][2].rstrip().lower()=="nearby":
                nearby_results.append([combined_mom_baby_acc[i][1],combined_mom_baby_acc[i][2],combined_mom_baby_pre[i][1],combined_mom_baby_pre[i][2]])
                #forJatin.append([combined_mom_baby[i][0],"nearby"])
            elif timestamp[pointer][2].rstrip().lower()=="picking up/ putting down":
                pupd_results.append([combined_mom_baby_acc[i][1],combined_mom_baby_acc[i][2],combined_mom_baby_pre[i][1],combined_mom_baby_pre[i][2]])
        elif combined_mom_baby_acc[i][0]>timestamp[pointer][1]:
            pointer+=1
            i-=1
            
    #result_file = open("result.csv",'wb')
    #wr = csv.writer(result_file, dialect='excel')
    #wr.writerows(forJatin)
    #result_file.close()
    
    """
    h_m, h_b = zip(*hold_results)
    plt.scatter(h_m,h_b,color='b',label="hold")
    c_m, c_b = zip(*carry_results)
    plt.scatter(c_m,c_b,color='r',label="carry")
    n_m,n_b=zip(*nearby_results)
    plt.scatter(n_m,n_b,color="g",label="nearby")
    #plt.xlim(-0.0005,0.0005)
    #plt.ylim(-0.0003,0.0005)
    plt.xlabel("mom's average\nacceleration (m/s)")
    plt.ylabel("baby's average\nacceleration (m/s)")
    plt.title("Hold vs carry vs. nearby")
    plt.legend(loc='upper right')
    """
    
    length_hs=len(hold_carry_results)
    length_n=len(nearby_results)
    length_pupd=len(pupd_results)
    
    
    X=hold_carry_results
    X.extend(nearby_results)
    X.extend(pupd_results)

    temp_list0=[0]*length_hs
    temp_list1=[1]*length_n
    temp_list2=[2]*length_pupd

    Y=temp_list0
    Y.extend(temp_list1)
    Y.extend(temp_list2)

    
    X=np.asarray(X)
    Y=np.asarray(Y)
    
    #X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
    
    
    f_myfile = open('../Nov11/model.pickle', 'rb')
    clf = pickle.load(f_myfile)  # variables come out in the order you put them in
    f_myfile.close()
    
    #clf = GradientBoostingClassifier()
    #clf.fit(X_train, Y_train)
    
    #true_positive=0
    #true_negative=0
    #false_positive=0
    #false_negative=0
    
    
    """
    prediction=clf.predict(X_test)
    for i in range(0,len(prediction)):
        if prediction[i]==Y_test[i]:
            if prediction[i]==1:
                true_positive+=1
            else:
                true_negative+=1
        else:
            if prediction[i]==1:
                false_positive+=1
            else:
                false_negative+=1
                
    print [true_positive,true_negative],[false_positive,false_negative]
        
            

    
    plt.show()
    """
    
    hs_correct=0
    n_correct=0
    pupd_correct=0
    hs_n=0
    hs_pupd=0
    n_hs=0
    n_pupd=0
    pupd_hs=0
    pupd_n=0
    
    prediction=clf.predict(X)
    print prediction
    print Y
    for i in range(0,len(prediction)):
        if prediction[i]==Y[i]:
            if prediction[i]==0:
                hs_correct+=1
            elif prediction[i]==1:
                n_correct+=1
            else:
                pupd_correct+=1
                
        else:
            if prediction[i]==0:
                if Y[i]==1:
                    n_hs+=1
                else:
                    pupd_hs+=1
            elif prediction[i]==1:
                if Y[i]==0:
                    hs_n+=1
                else:
                    pupd_n+=1
            else:
                if Y[i]==0:
                    hs_pupd+=1
                else:
                    n_pupd+=1
     
    print hs_correct,n_correct,pupd_correct,hs_n,hs_pupd,n_hs,n_pupd,pupd_hs,pupd_n
        
    
    
   
