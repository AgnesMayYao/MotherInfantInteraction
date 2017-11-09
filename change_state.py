import csv
import numpy as np
import os


labels_folder="users/labels/"
new_labels_folder = "users/labels_new/"


dictionary = {}
dictionary['hovering'] = ['hovering', 'nearby', 'nearby']
dictionary['touching'] = ['hovering', 'nearby', 'nearby']
dictionary['nearby'] = ['nearby', 'nearby', 'nearby']
dictionary['holding'] =['holding', 'holding', 'holding']
dictionary['picking up'] = ['carrying', 'carrying', 'holding']
dictionary['putting down'] = ['carrying', 'carrying', 'holding']
dictionary['carrying'] = ['carrying', 'carrying', 'holding']
dictionary['bouncing'] = ['carrying', 'carrying', 'holding']


if __name__ == '__main__':

	files=os.listdir(labels_folder)


	for f in files:

		output = []
		
		file=open(labels_folder + str(f), "r")
		reader = csv.reader(file) 

		for row in reader:
			temp = []
			temp.append(row[0])
			temp.append(row[1])
			temp.extend(dictionary[row[1]])
			if temp[3] == 'carrying':
				temp.append(temp[3])
			else:
				temp.append('uncarrying')
			output.append(temp)

		file.close()

		result_file = open(new_labels_folder + str(f), 'wb')
		wr = csv.writer(result_file, dialect='excel')
		wr.writerows(output)
		result_file.close()