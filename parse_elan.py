import json
import csv

with open('xuewen_Feb18.json') as json_data:
    d = json.load(json_data)
    json_data.close()

#print d.keys()

dictionary={}
dic=d[u'ANNOTATION_DOCUMENT'][u"TIME_ORDER"][u"TIME_SLOT"]
for each in dic:
	dictionary[str(each[u"_TIME_SLOT_ID"])]=int(each[u'_TIME_VALUE'])

output=[]

data=d[u'ANNOTATION_DOCUMENT'][u"TIER"][2][u"ANNOTATION"]

for each in data:
	t1=dictionary[str(each[u'ALIGNABLE_ANNOTATION'][u'_TIME_SLOT_REF1'])]
	t2=dictionary[str(each[u'ALIGNABLE_ANNOTATION'][u'_TIME_SLOT_REF2'])]
	t3=str(each[u'ALIGNABLE_ANNOTATION'][u'ANNOTATION_VALUE'])
	output.append([t1,t2,t3])

result_file = open("xuewen_Feb18.csv",'wb')
wr = csv.writer(result_file, dialect='excel')
wr.writerows(output)
result_file.close()

print output
