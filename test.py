import csv
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import copy




folder="./data/Jan17/pickup putdown 1 synched data/"
pre_filename=[folder+"synchedMomBarometerJan17part2.csv",folder+"synchedBabyBarometerJan17part2.csv"]
       
for i in range(0,2):
    pre_file=open(pre_filename[i],"r")
    pre_reader = csv.reader(pre_file)
    row_no=1
        
    data_time=[]
    data_pre=[]
    
    for row in pre_reader:
        if row_no>1:
            data_time.append(int(row[0])*1.0/1000)
            data_pre.append(float(row[1]))
            #data_pre.append(30*(1013 - float(row[3])/100))
        row_no+=1

    data_copy=copy.deepcopy(data_pre)

    new_data = savgol_filter(data_pre, window_length=201, polyorder=5,mode='mirror')
    
    
    
    if i==0:
        
        plt.plot(data_time,data_copy,'purple')
        plt.plot(data_time,new_data,'black')
        #pre.text(data_time[len(data_time)-1], data_pre[len(data_pre)-1], "mother", fontsize=small_font_size, color='purple')
    #else:
        #plt.plot(data_time,new_data,'green')
        #plt.plot(data_time,data_pre,'blue')
        #pre.text(data_time[len(data_time)-1], data_pre[len(data_pre)-1], "baby", fontsize=small_font_size, color='green')
        

    pre_file.close()

plt.show()
#plt.set_ylabel('Altitude\n feet',fontsize=small_font_size) 