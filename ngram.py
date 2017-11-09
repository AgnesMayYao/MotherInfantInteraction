import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats.stats import pearsonr   
from sklearn import svm
from scipy.signal import freqz
from scipy.fftpack import rfft, irfft, fftfreq
import scipy
import csv
from scipy.stats import entropy
from confusion import drawCM
from personIndependent import giant_plot, getOutput
from operator import itemgetter
from itertools import groupby
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn import svm


from collections import Counter



labels_folder="users/labels_new/"

files_for_testing=['P2 (Jan. 17)', 'P3 (Jan. 27)', 'P5 (Feb. 13)', 'P7 (Feb. 18)', 'P9 (Mar. 31)', \
'P10 (May 1)', 'P11 (May 7)', 'P15 (May 11)', 'P18 (May 22)', 'P19 (May 26)']



 



if __name__ == '__main__':

	for i in range(1, 6):

		files=os.listdir(labels_folder)

		sequences = []

		fileNames = []


		for f in files:

			labels = []

			
			file=open(labels_folder+str(f),"r")
			reader = csv.reader(file) 

			for row in reader:
				labels.append(''.join(row[i].split()))

			file.close()

			fileNames.append(str(f).replace(".csv",''))
			#sequence = ' '.join(map(itemgetter(0), groupby(labels)))
			sequence = ' '.join(labels)
			sequences.append(sequence)


		fileNames.append("Overall")

		vectorizer = CountVectorizer(ngram_range=(2, 2))
		X = vectorizer.fit_transform(sequences)
		states = vectorizer.get_feature_names()
		results = X.toarray()
		results = np.vstack([results, np.sum(results, axis = 0)])
		results = np.array(results, dtype = float)
		#results = np.around(results * 1.0 / results.sum(axis = 1)[:,None], decimals=3)

		u_vectorizer = CountVectorizer(ngram_range=(1, 1))
		u_X = u_vectorizer.fit_transform(sequences)
		u_states = u_vectorizer.get_feature_names()
		u_results = u_X.toarray()
		u_results = np.vstack([u_results, np.sum(u_results, axis = 0)])

		for i, s in enumerate(states):
			a = np.array(results[:, i], dtype = float)
			b =np.array(u_results[:, u_states.index(s.split()[0])], dtype = float)
			results[:, i] = np.divide(a, b, out=np.zeros_like(a), where=b!=0, dtype = float)

		results = np.around(results, decimals = 3)


		df = DataFrame(results)
		df.columns = states
		df.insert(loc=0, column='Session', value=fileNames)
		#df.to_csv("./ngram_results.csv", mode = 'a', header = True)


		print df

		#print sequence



