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
    """
    holding_still_results=[]
    holding_walking_results=[]
    hovering_results=[]
    nearby_results=[]
    picking_up_results=[]
    putting_down_results=[]
    """
    holding_results=[]
    nearby_results=[]
    pupd_results=[]

    
    timestamp=[]
    timestamp_file=open("timestamp_Nov11_3_state.csv","r")
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
    

    pre_filename=["Mom_2016-11-11T13.57.58.368_D4D7C843B379_Pressure.csv", \
    "Baby_2016-11-11T13.57.58.368_FB7D0BFC0351_Pressure.csv"]
       
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
    

    acc_filename=["Mom_2016-11-11T13.57.58.368_D4D7C843B379_Accelerometer.csv", \
    "Baby_2016-11-11T13.57.58.368_FB7D0BFC0351_Accelerometer.csv"]
       
    for i in range(0,2):
        acc_file=open(acc_filename[i],"r")
        acc_reader = csv.reader(acc_file)
        row_no=1
        pre_time=0
        
        for row in acc_reader:
            if row_no>1:
                temp_time=round(float(row[0])/1000)
                
                mean_acc=(abs(float(row[3]))+abs(float(row[4]))+abs(float(row[5])))/3
                temp_dist=mean_acc
                
                if i==0:
                    mom_simplified.append([temp_time,temp_dist])
                else:
                    baby_simplified.append([temp_time,temp_dist])
            row_no+=1
            
        acc_file.close()
            
    mom_acc_simplify=simplify_acc(mom_simplified)
    baby_acc_simplify=simplify_acc(baby_simplified)
    
    combined_mom_baby_acc=combine(mom_acc_simplify,baby_acc_simplify)
    
    #print combined_mom_baby_pre
    #print combined_mom_baby_acc
    
    
    
    pointer=0
    for i in range(0,len(combined_mom_baby_acc)):
        if combined_mom_baby_acc[i][0]<timestamp[pointer][0]:
            continue
        if combined_mom_baby_acc[i][0]>=timestamp[pointer][0] and combined_mom_baby_acc[i][0]<=timestamp[pointer][1]:
            """
            if timestamp[pointer][2].rstrip()=="holding still":
                holding_still_results.append([combined_mom_baby[i][1],combined_mom_baby[i][2]])
                #forJatin.append([combined_mom_baby[i][0],"hold/carry"])
            elif timestamp[pointer][2].rstrip()=="holding walking":
                holding_walking_results.append([combined_mom_baby[i][1],combined_mom_baby[i][2]])
                #forJatin.append([combined_mom_baby[i][0],"nearby"])
            elif timestamp[pointer][2].rstrip()=="hovering":
                hovering_results.append([combined_mom_baby[i][1],combined_mom_baby[i][2]])
            elif timestamp[pointer][2].rstrip()=="nearby":
                nearby_results.append([combined_mom_baby[i][1],combined_mom_baby[i][2]])
            elif timestamp[pointer][2].rstrip()=="picking up":
                picking_up_results.append([combined_mom_baby[i][1],combined_mom_baby[i][2]])
            elif timestamp[pointer][2].rstrip()=="putting down":
                putting_down_results.append([combined_mom_baby[i][1],combined_mom_baby[i][2]])
            """
            if timestamp[pointer][2].rstrip()=="holding":
                holding_results.append([combined_mom_baby_acc[i][1],combined_mom_baby_acc[i][2],combined_mom_baby_pre[i][1],combined_mom_baby_pre[i][2]])
                #forJatin.append([combined_mom_baby[i][0],"hold/carry"])
            elif timestamp[pointer][2].rstrip()=="picking up/ putting down":
                pupd_results.append([combined_mom_baby_acc[i][1],combined_mom_baby_acc[i][2],combined_mom_baby_pre[i][1],combined_mom_baby_pre[i][2]])
            elif timestamp[pointer][2].rstrip()=="nearby":
                nearby_results.append([combined_mom_baby_acc[i][1],combined_mom_baby_acc[i][2],combined_mom_baby_pre[i][1],combined_mom_baby_pre[i][2]])
        elif combined_mom_baby_acc[i][0]>timestamp[pointer][1]:
            pointer+=1
            i-=1
            
    #result_file = open("result.csv",'wb')
    #wr = csv.writer(result_file, dialect='excel')
    #wr.writerows(forJatin)
    #result_file.close()
    
    """
    hs_m, hs_b = zip(*holding_still_results)
    plt.scatter(hs_m,hs_b,label="holding still",color="red")
    hw_m, hw_b = zip(*holding_walking_results)
    plt.scatter(hw_m,hw_b,label="holding walking",color="pink")
    n_m,n_b=zip(*nearby_results)
    plt.scatter(n_m,n_b,label="nearby",color="black")
    h_m,h_b=zip(*hovering_results)
    plt.scatter(h_m,h_b,label="hovering",color="green")
    pu_m, pu_b = zip(*picking_up_results)
    plt.scatter(pu_m,pu_b,label="picking up",color="blue")
    pd_m, pd_b = zip(*putting_down_results)
    plt.scatter(pd_m,pd_b,label="putting down",color="purple")
    
    h_m, h_b = zip(*holding_still_results)
    plt.scatter(h_m,h_b,label="holding",color="red")
    pp_m, pp_b = zip(*pupd_results)
    plt.scatter(pp_m,pp_b,label="pp",color="blue")
    n_m,n_b=zip(*nearby_results)
    plt.scatter(n_m,n_b,label="nearby",color="black")
    
    
    
    
    #plt.xlim(-0.0005,0.0005)
    #plt.ylim(-0.0003,0.0005)
    plt.xlabel("mom's average\npressure (Pa)")
    plt.ylabel("baby's average\npressure (Pa)")
    plt.title("Nov 11_3 states_pressure")
    plt.legend(loc='lower right')
    """
    
    length_hs=len(holding_results)
    length_n=len(nearby_results)
    length_pupd=len(pupd_results)
    
    
    X=holding_results
    #X.extend(carry_results)
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
    
    print Y
    #X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
    

    clf = GradientBoostingClassifier()
    clf.fit(X, Y)
    
    _myfile = open('model.pickle', 'wb')
    pickle.dump(clf,_myfile)
    _myfile.close()
    
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
    
    prediction=clf.predict(X_test)
    for i in range(0,len(prediction)):
        if prediction[i]==Y_test[i]:
            if prediction[i]==0:
                hs_correct+=1
            elif prediction[i]==1:
                n_correct+=1
            else:
                pupd_correct+=1
                
        else:
            if prediction[i]==0:
                if Y_test[i]==1:
                    n_hs+=1
                else:
                    pupd_hs+=1
            elif prediction[i]==1:
                if Y_test[i]==0:
                    hs_n+=1
                else:
                    pupd_n+=1
            else:
                if Y_test[i]==0:
                    hs_pupd+=1
                else:
                    n_pupd+=1
     
    print hs_correct,n_correct,pupd_correct,hs_n,hs_pupd,n_hs,n_pupd,pupd_hs,pupd_n
        
            

    
    #plt.show()
            
    """
    
   
