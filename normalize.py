import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


data_folder="users/data/"
data_folder_normalized="users/data_normalized/"


'''

features=np.concatenate(([np.array(hardCodedPressure)[:,1]],[np.array(accMomVariance)[:,1]],[np.array(accBabyVariance)[:,1]],[np.array(preMomVariance)[:,1]],[np.array(preBabyVariance)[:,1]],[np.array(accCorrelation)[:,1]], \
[np.array(accDifference)[:,1]],[np.array(preDifference)[:,1]],[np.array(selfAccDiffMom)[:,1]],[np.array(selfAccDiffBaby)[:,1]],[np.array(selfPreDiffMom)[:,1]], \
[np.array(selfPreDiffBaby)[:,1]],[np.array(preCorrelation)[:,1]],[np.array(filteredPreMom)[:,1]],[np.array(filteredPreBaby)[:,1]]),axis=0)
'''





if __name__ == '__main__':

	

	files = os.listdir(data_folder)

	for f in files:
		_data=[]
		file=open(data_folder+str(f),"r")
		reader = csv.reader(file) 

		for row in reader:
			_data.append(map(float, row))


		file.close()
		
		_data = np.array(_data)

		for i in range(15):
			if i != 0 and i != 5 and i != 12:
				#print _data[:, i]
				if np.max(_data[:, i]) - np.min(_data[:, i]) == 0:
					_data[:, i] = 0.5
				else:
					_data[:, i] = (_data[:, i] - np.min(_data[:, i])) / (np.max(_data[:, i]) - np.min(_data[:, i]))

		result_file = open(data_folder_normalized + str(f), 'wb')
		wr = csv.writer(result_file, dialect='excel')
		wr.writerows(_data)
		result_file.close()

