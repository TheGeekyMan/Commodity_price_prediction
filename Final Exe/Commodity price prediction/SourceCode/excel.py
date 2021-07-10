import csv
import pandas as pd
citynamefile = "soyaOilmarket name.csv"
soyafile = "rice.csv"
cities = []
rows = []
count = 0
col_list = ["mkt_name"]
df = pd.read_csv(soyafile,usecols=col_list)
cityname = []
with open(citynamefile,'r') as cityfile:
	csvreader = csv.reader(cityfile)
	cities = next(csvreader)
print(cities)

with open(soyafile,'r') as soya:
	csvreader = csv.reader(soya)
	for row in csvreader:
		rows.append(row)
# for row in rows[:5]:
# 	print(row[5])
# 	print("\n")
		
for el in cities:
	for i in range(1,10070):
		if str(el) == str(df["mkt_name"][i]):
			count = count + 1
	if count < 30 :
		cityname.append(el)

	count = 0

f = open('resultrice.csv','w',encoding='UTF8',newline='')
writer = csv.writer(f)
for row in rows:
    if row[5] not in cityname:
    	writer.writerow(row)
    else:
    	print(row[5])	
f.close()
print(cityname)		
print(len(cityname))
print(len(cities)) 