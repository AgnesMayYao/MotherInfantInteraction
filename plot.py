import pickle
import matplotlib.pyplot as plt





def plotFeatures(predict,real,method):
    
    small_font_size=15
    large_font_size=20
    infant_line_width=5.0
        
    fig = plt.figure(figsize=(30, 20))
    
    f = open("save", "r")
    babyAcc=pickle.load(f)
    momAcc=pickle.load(f)
    babyPre=pickle.load(f)
    momPre=pickle.load(f)
    babyHr=pickle.load(f)
    momHr=pickle.load(f)
    f.close()
    
    acc = fig.add_subplot(515)
    acc.plot(list(range(1,len(babyAcc)+1)),babyAcc,'purple')
    acc.text(len(babyAcc), babyAcc[len(babyAcc)-1], "baby", fontsize=small_font_size, color='green')
    acc.plot(list(range(1,len(momAcc)+1)),momAcc,'green')
    acc.text(len(momAcc), momAcc[len(momAcc)-1], "mother", fontsize=small_font_size, color='purple')
        
    acc.set_xlabel('time (s)',fontsize=small_font_size)
    acc.set_ylabel('Motion data\n(average of 3D axes)',fontsize=small_font_size)    
            
    
    pre = fig.add_subplot(514,sharex=acc)
    pre.plot(list(range(1,len(babyPre)+1)),babyPre,'purple')
    pre.plot(list(range(1,len(momPre)+1)),momPre,'g')
    pre.text(len(babyPre), babyPre[len(babyPre)-1], "baby", fontsize=small_font_size, color='green')
    pre.text(len(momPre), momPre[len(momPre)-1], "mother", fontsize=small_font_size, color='purple')
                 
    pre.set_ylabel('Pressure\nPa',fontsize=small_font_size)


    hr = fig.add_subplot(513,sharex=acc)
    hr.plot(list(range(1,len(babyHr)+1)),babyHr,'purple')
    hr.plot(list(range(1,len(momHr)+1)),momHr,'g')
    hr.text(len(babyHr), babyHr[len(babyHr)-1], "baby", fontsize=small_font_size, color='green')
    hr.text(len(momHr), momHr[len(momHr)-1], "mother", fontsize=small_font_size, color='purple')
                 
    hr.set_ylabel('Heart Rate',fontsize=small_font_size)
      
    
    realDate=fig.add_subplot(512,sharex=acc)
    realDate.plot(list(range(1,len(real)+1)),real,'green')
    realDate.set_ylabel('Truth',fontsize=small_font_size)


    predictData=fig.add_subplot(511,sharex=acc)
    predictData.plot(list(range(1,len(predict)+1)),predict,'red')
    predictData.set_ylabel('Prediction',fontsize=small_font_size)


    #proximity.set_ylim([-1,4])
    #proximity.set_yticklabels(labels)
    
    
 
    acc.set_xlim([0,200])
   
    pre.yaxis.tick_right()
    pre.yaxis.set_label_position("right")
    realDate.yaxis.tick_right()
    realDate.yaxis.set_label_position("right")
    fig.subplots_adjust(hspace=0)
    #acc.legend(loc='upper right')
    #pre.legend(loc="upper right")
    plt.title('Nov 11 using Oct 28 model '+method,fontsize=large_font_size)
    plt.setp(pre.get_xticklabels(), visible=False)
    plt.setp(hr.get_xticklabels(), visible=False)
    plt.setp(realDate.get_xticklabels(), visible=False)
    plt.setp(predictData.get_xticklabels(),visible=False)
    plt.setp(acc.get_xticklabels(),fontsize=small_font_size)
    plt.setp(acc.get_yticklabels(),fontsize=small_font_size)
    plt.setp(hr.get_yticklabels(),fontsize=small_font_size)
    plt.setp(pre.get_yticklabels(),fontsize=small_font_size)
    plt.setp(realDate.get_yticklabels(),fontsize=small_font_size)
    plt.setp(predictData.get_yticklabels(),fontsize=small_font_size)
    #acc.grid()
    #pre.grid()
    #infant.grid()
    plt.show()