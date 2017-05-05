import csv, os, urllib, sys, time

failure = []

start = time.time()
N = 3400000
count = 0
count_per_print = N / 25
with open('../data/small_data.csv','r') as csvfile:
# with open('../data/imUrl.csv','r') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		asin, imUrl, category = row
		path = os.path.join('../data/img/',category)
		if not os.path.exists(path):
			os.makedirs(path)
		try:
			urllib.urlretrieve(imUrl, os.path.join(path, asin+'.'+imUrl.split('.')[-1]))
		except:
			failure.append((asin, imUrl, category))
		count += 1
		if (count / 500) * 500 == count:
			num_eq = count / count_per_print
			sys.stdout.write('\r '+str(count)+' / '+str(N)+' [' + '='*num_eq + ' '*(25 - num_eq) + '] - %0.2fs - est. %0.2fs'%(float(time.time()-start), float(time.time()-start) * N / count))
			sys.stdout.flush()
	sys.stdout.write('\r '+str(count)+' / '+str(count)+' [' + '='*25 + '] - %0.2fs \n\nFinished! \n'%(float(time.time()-start)))
	sys.stdout.flush()
with open('../data/download_failure.csv','w') as csvfile:
	writer = csv.writer(csvfile)
	for row in failure:
		writer.writerow(row)