import matplotlib.pyplot as plt

fig = plt.figure(figsize=(9, 5))
la=fig.add_subplot(111)

word_size=18

graph_title="P1 (Oct. 28)"
'''



#Jan17
states = [7, 7, 5, 3, 2]
LR=[0.55403280929596721, 0.54876965140123035, 0.59336978810663021, 0.5276828434723172, 0.73701298701298701]
RF=[0.51373889268626116, 0.53212576896787422, 0.60030758714969246, 0.63202323991797682, 0.75003417634996583]
SVM=[0.29330143540669856, 0.29330143540669856, 0.34350649350649348, 0.42255639097744363, 0.51346548188653451]
AC=[0.55058099794941895, 0.54542036910457958, 0.63451811346548193, 0.63458646616541359, 0.71322624743677376]



#Feb13
states =  [8, 7, 5, 3, 2]
LR =  [0.53362584017923831, 0.5336445108289769, 0.55894324122479466, 0.59957057505601197, 0.68467139656460052]
RF =  [0.37320761762509336, 0.41954817027632563, , 0.402165795369678840.51639283047050033, 0.60162434652725916]
SVM =  [0.34800224047796863, 0.35186706497386111, 0.38670649738610902, 0.44288648244958928, 0.5396751306945482]
AC =  [0.28623973114264373, 0.27662434652725915, 0.42156460044809557, 0.51659820761762509, 0.65048543689320393]


#Nov 18
states =  [6, 5, 3, 2]
LR =  [0.12957746478873239, 0.30633802816901412, 0.57112676056338019, 0.59507042253521125]
RF =  [0.42816901408450703, 0.61197183098591545, 0.7274647887323944, 0.73169014084507045]
SVM =  [0.17464788732394365, 0.25704225352112675, 0.55070422535211272, 0.55563380281690145]
AC =  [0.34507042253521131, 0.52605633802816898, 0.58591549295774636, 0.53591549295774643]


#Feb 18
states =  [8, 7, 5, 3, 2]
LR =  [0.48891666666666661, 0.55231349206349201, 0.5906904761904761, 0.61303571428571435, 0.7218174603174603]
RF =  [0.46301587301587305, 0.50944047619047628, 0.55941269841269847, 0.69306746031746036, 0.73496825396825405]
SVM =  [0.32398412698412693, 0.38645634920634919, 0.41150793650793654, 0.45552777777777775, 0.51782936507936506]
AC =  [0.42015476190476192, 0.52104761904761898, 0.53799206349206341, 0.68527380952380945, 0.80990873015873022]



#May 7
states =  [7, 6, 4, 3, 2]
LR =  [0.32872324041275214, 0.4366856614815956, 0.43138764823656245, 0.45258190138830823, 0.69178895953884401]
RF =  [0.46458163736771468, 0.5394226749686476, 0.54407823810257194, 0.56789949616070046, 0.76868495742667942]
SVM =  [0.3426261248377373, 0.45520890629469096, 0.47845591956172584, 0.48446018789465584, 0.70909330927812364]
AC =  [0.43744252051660037, 0.55066335174143588, 0.56263008514664148, 0.57188180678092881, 0.75612857802908628]


#Jan 27
states =  [7, 6, 4, 3, 2]
LR =  [0.2267213114754098, 0.2267213114754098, 0.45846994535519131, 0.7174863387978142, 0.72076502732240444]
RF =  [0.31896174863387977, 0.33234972677595626, 0.37830601092896171, 0.72409836065573774, 0.76349726775956284]
SVM =  [0.17087431693989069, 0.16759562841530054, 0.32234972677595625, 0.7043715846994536, 0.70103825136612019]
AC =  [0.25967213114754101, 0.24972677595628417, 0.33535519125683055, 0.70765027322404372, 0.75027322404371588]


#May 1
states =  [6, 6, 4, 3, 2]
LR =  [0.76825491958845271, 0.76825491958845271, 0.79091998801318553, 0.7938267905304166, 0.8517430826091299]
RF =  [0.68781340525422041, 0.68075117370892024, 0.70620317650584352 , 0.711846968334831630.77561682149635403]
SVM =  [0.37342922784936572, 0.37342922784936572, 0.36921386474877638, 0.38477674557986213, 0.71055838577564678]
AC =  [0.54321246628708431, 0.54462091699130954, 0.59991009889121971, 0.65676755568874223, 0.82643092598142043]


#May 22
states =  [8, 7, 5, 3, 2]
LR =  [0.42737434725848572, 0.57740970409051351, 0.59512891644908616, 0.60656685161009571, 0.71597720844212365]
RF =  [0.51688152741514359, 0.68369642080069626, 0.70140747389033953, 0.72276028067885112, 0.78269963011314181]
SVM =  [0.37884845517841603, 0.60192694734551788, 0.59881826588337694, 0.56027931897301997, 0.69310133812010455]
AC =  [0.43916313098346382, 0.65708904482158403, 0.67640203437771973, 0.65916829852045244, 0.74251931026979989]



#May 26
states =  [8, 7, 5, 3, 2]
LR =  [0.65784313725490196, 0.6588235294117647, 0.67843137254901964, 0.68431372549019609, 0.84313725490196079]
RF =  [0.62450980392156874, 0.65490196078431373, 0.67352941176470593, 0.67549019607843142, 0.86960784313725481]
SVM =  [0.58725490196078434, 0.58529411764705885, 0.61960784313725481, 0.62549019607843137, 0.82156862745098036]
AC =  [0.61470588235294121, 0.63823529411764712, 0.65098039215686276, 0.68627450980392157, 0.86568627450980384]


#Mar 31
states =  [8, 7, 5, 3, 2]
LR =  [0.51533942558746737, 0.60079960835509139, 0.60079960835509139, 0.60132180156657955, 0.75535248041775449]
RF =  [0.4570496083550914, 0.55657093124456058, 0.57635443864229763, 0.60979248259355967, 0.79192096388163624]
SVM =  [0.41530542863359443, 0.50596850522193215, 0.49190192558746737, 0.49977833986074849, 0.64696475195822456]
AC =  [0.47892188859878154, 0.58470409051349004, 0.59719457136640552, 0.61130738685813757, 0.7861537206266318]



#May 11
states =  [8, 7, 5, 3, 2]
LR =  [0.36451797881670378, 0.58920596370505296, 0.59414760844633341, 0.60702624300074215, 0.75350131552317345]
RF =  [0.33979626256493289, 0.51921001146866363, 0.55291101666329356, 0.63656479794913312, 0.71954732510288066]
SVM =  [0.17856371854550362, 0.4032280914794576, 0.40980570734669097, 0.42460702961613705, 0.46897389192471162]
AC =  [0.36036902111583352, 0.55872630371719623, 0.59327733927005322, 0.63492208055049582, 0.6859542602712001]

#Nov 11
states =  [6, 5, 3, 2]
LR =  [0.54147727272727275, 0.64678030303030298, 0.72367424242424239, 0.83465909090909096]
RF =  [0.63219696969696959, 0.69318181818181812, 0.7537878787878789, 0.9015151515151516]
SVM =  [0.4375, 0.48598484848484846, 0.60738636363636356, 0.68125000000000002]
AC =  [0.60132575757575757, 0.60132575757575757, 0.69261363636363638, 0.82045454545454555]
'''

#Oct. 28
states =  [6, 5, 3, 2]
LR =  [0.6742424242424242, 0.82196969696969691, 0.8893939393939394, 0.90113636363636362]
RF =  [0.81022727272727268, 0.85321969696969691, 0.9327651515151516, 0.96344696969696975]
SVM =  [0.66268939393939397, 0.76761363636363633, 0.81647727272727266, 0.88371212121212106]
AC =  [0.69412878787878785, 0.80454545454545467, 0.86609848484848473, 0.96950757575757573]



la.plot(states,LR,label="Logistic Regression")
la.plot(states,RF,label="Random Forest")
la.plot(states,SVM,label="Support Vector Machine")
la.plot(states,AC,label="Adaboost Classifier")


plt.ylim(ymin=0,ymax=1)


plt.xlabel("num of states",fontsize=word_size)
plt.ylabel("Accuracy",fontsize=word_size)
plt.title(graph_title,fontsize=word_size)

la.xaxis.set_major_locator(plt.FixedLocator(states))


for tick in la.xaxis.get_major_ticks():
	tick.label.set_fontsize(word_size) 

for tick in la.yaxis.get_major_ticks():
	tick.label.set_fontsize(word_size) 



la.legend(loc='lower right')
plt.show()