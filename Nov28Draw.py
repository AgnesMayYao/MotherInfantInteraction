import csv
import matplotlib.pyplot as plt





if __name__ == '__main__':
    
    small_font_size=30
    large_font_size=35
    infant_line_width=5.0
    
    offset=1479496140299
    
    fig = plt.figure(figsize=(30, 20))
    acc = fig.add_subplot(414)
    #(acc,pre) = plt.subplots(2, sharex=True,sharey=False)

    #acc files, should be 2 files
    #no_acc = int(raw_input("the number of accelerometer files: "))  
    acc_filename=["Mom_2016-11-18T13.44.09.243_E32C50C8C747_Accelerometer.csv", \
    "Baby_2016-11-18T13.44.09.243_D415A98F6DFC_Accelerometer.csv"]
    
    start_time=0
    
    for i in range(0,2):
        #acc_filename=raw_input("accelerometer file name: ")
        #plot
        acc_file=open(acc_filename[i],"r")
        acc_reader = csv.reader(acc_file)
        
        row_no=1
        
        data_time=[]
        data_acc=[]
        for row in acc_reader:
            if row_no>1:
                if int(row[0])<offset:
                    row_no+=1
                    continue
                if start_time==0:
                    start_time=int(row[0])
                data_time.append((int(row[0])-start_time)*1.0/1000)
                data_acc.append((abs(float(row[3]))+abs(float(row[4]))+abs(float(row[5])))/3)
            row_no+=1
        
        
        if i==0:
            line=acc.plot(data_time,data_acc,'purple')
            acc.text(data_time[len(data_time)-1], data_acc[len(data_acc)-1], "mother", fontsize=small_font_size, color='purple')
        else:
            line=acc.plot(data_time,data_acc,'green')
            acc.text(data_time[len(data_time)-1], data_acc[len(data_acc)-1], "baby", fontsize=small_font_size, color='green')
        
    
    acc.set_xlabel('time (s)',fontsize=small_font_size)
    acc.set_ylabel('Motion data\n(average of 3D axes)',fontsize=small_font_size)    
            
            
    #pre=acc.twinx() 
    pre = fig.add_subplot(413,sharex=acc)
    #pressure files     
    #no_pre = int(raw_input("the number of pressure files: "))  
    pre_filename=["Mom_2016-11-18T13.44.09.243_E32C50C8C747_Pressure.csv", \
    "Baby_2016-11-18T13.44.09.243_D415A98F6DFC_Pressure.csv"]
    
    for i in range(0,2):
        #pre_filename=raw_input("pressure file name: ")
        #plot
        pre_file=open(pre_filename[i],"r")
        pre_reader = csv.reader(pre_file)
        
        row_no=1
        
        data_time=[]
        data_pre=[]
        
        for row in pre_reader:
            if row_no>1:
                if int(row[0])<offset:
                    row_no+=1
                    continue
                data_time.append((int(row[0])-start_time)*1.0/1000)
                data_pre.append(30*(1013 - float(row[3])/100))
                #data_pre.append(float(row[3]))
            row_no+=1
        
        
        
        if i==0:
            line2=pre.plot(data_time,data_pre,'purple')
            pre.text(data_time[len(data_time)-1], data_pre[len(data_pre)-1], "mother", fontsize=small_font_size, color='purple')
        else:
            line2=pre.plot(data_time,data_pre,'g')
            pre.text(data_time[len(data_time)-1], data_pre[len(data_pre)-1], "baby", fontsize=small_font_size, color='green')
            
    pre.set_ylabel('Pressure\nPa',fontsize=small_font_size)
      
    
    
    
    
    
    
    infant = fig.add_subplot(411,sharex=acc)
    #infant file     
    infant_filename="P2_infant.txt"
    
    #infant_filename=raw_input("infant file name: ")
    #plot
    infant_file=open(infant_filename,"r")
    row_no=1
    counter=0
    infant.set_ylim([-1,4])
    labels = [item for item in infant.get_yticklabels()]
    #labels[1]='Nearby'
    #labels[2]='Hold'
    #labels[3]='Carry'
    #label=['Nearby','Hold','Carry']
    label=[]

    for line in infant_file:
        temp=line.split("\t")
        start_point=(float(temp[3])*1000+offset-start_time)*1.0/1000
        end_point=(float(temp[5])*1000+offset-start_time)*1.0/1000
        
        if temp[8]!=None and len(temp[8])!=1:
            label_index = label.index(temp[8]) if temp[8] in label else -1
            if data_time[len(data_time)-1]>=end_point:      
                if label_index==-1:
                    label.append(temp[8])
                    infant.plot([start_point,end_point],[counter,counter],linewidth=infant_line_width,color='black')
                    labels[counter+1]=temp[8]
                    counter+=1
                else:
                    infant.plot([start_point,end_point],[label_index,label_index],linewidth=infant_line_width,color='black')

            elif data_time[len(data_time)-1]>start_point:
                if label_index==-1:
                    label.append(temp[8])
                    infant.plot([start_point,data_time[len(data_time)-1]],[counter,counter],linewidth=infant_line_width,color='black')
                    counter+=1
                    labels[counter]=temp[8]
                    
                else:
                    infant.plot([start_point,data_time[len(data_time)-1]],[label_index,label_index],linewidth=infant_line_width,color='black')


        row_no+=1

    infant.set_ylim([-1,4])
    infant.set_yticklabels(labels)





    proximity = fig.add_subplot(412,sharex=acc)
    #infant file     
    proximity_filename="P2_proximity.txt"
    
    #infant_filename=raw_input("infant file name: ")
    #plot
    proximity_file=open(proximity_filename,"r")
    row_no=1
    counter=0
    proximity.set_ylim([-1,4])
    labels = [item for item in proximity.get_yticklabels()]
    #labels[1]='Nearby'
    #labels[2]='Hold'
    #labels[3]='Carry'
    #label=['Nearby','Hold','Carry']
    label=[]

    for line in proximity_file:
        temp=line.split("\t")
        start_point=(float(temp[3])*1000+offset-start_time)*1.0/1000
        end_point=(float(temp[5])*1000+offset-start_time)*1.0/1000
        
        if temp[10]!=None and len(temp[10])!=1:
            label_index = label.index(temp[10]) if temp[10] in label else -1
            if data_time[len(data_time)-1]>=end_point:      
                if label_index==-1:
                    label.append(temp[10])
                    proximity.plot([start_point,end_point],[counter,counter],linewidth=infant_line_width,color='black')
                    labels[counter+1]=temp[10]
                    counter+=1
                else:
                    proximity.plot([start_point,end_point],[label_index,label_index],linewidth=infant_line_width,color='black')

            elif data_time[len(data_time)-1]>start_point:
                if label_index==-1:
                    label.append(temp[10])
                    proximity.plot([start_point,data_time[len(data_time)-1]],[counter,counter],linewidth=infant_line_width,color='black')
                    labels[counter+1]=temp[10]
                    counter+=1
                else:
                    proximity.plot([start_point,data_time[len(data_time)-1]],[label_index,label_index],linewidth=infant_line_width,color='black')


        row_no+=1

    proximity.set_ylim([-1,4])
    proximity.set_yticklabels(labels)
    
    
 
    acc.set_xlim([0,1600])
   
    pre.yaxis.tick_right()
    pre.yaxis.set_label_position("right")
    proximity.yaxis.set_label_position("right")
    fig.subplots_adjust(hspace=0)
    #acc.legend(loc='upper right')
    #pre.legend(loc="upper right")
    plt.title('Nov 18',fontsize=large_font_size)
    plt.setp(pre.get_xticklabels(), visible=False)
    plt.setp(infant.get_xticklabels(), visible=False)
    plt.setp(proximity.get_xticklabels(), visible=False)
    plt.setp(acc.get_xticklabels(),fontsize=small_font_size)
    plt.setp(acc.get_yticklabels(),fontsize=small_font_size)
    plt.setp(pre.get_yticklabels(),fontsize=small_font_size)
    plt.setp(infant.get_xticklabels(),fontsize=small_font_size)
    plt.setp(infant.get_yticklabels(),fontsize=small_font_size)
    plt.setp(proximity.get_xticklabels(),fontsize=small_font_size)
    plt.setp(proximity.get_yticklabels(),fontsize=small_font_size)
    #acc.grid()
    #pre.grid()
    #infant.grid()
    plt.show()
        
    pre_file.close()    
    acc_file.close()
    infant_file.close()  
    proximity_file.close()
    
    
    #Mom_2016-10-28T16.16.08.276_D4D7C843B379_Accelerometer.csv
    #Baby_2016-10-28T16.16.08.276_FB7D0BFC0351_Accelerometer.csv
    #Mom_2016-10-28T16.16.08.276_D4D7C843B379_Pressure.csv
    #Baby_2016-10-28T16.16.08.276_FB7D0BFC0351_Pressure.csv
