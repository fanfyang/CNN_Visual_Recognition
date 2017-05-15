import csv, os, urllib, sys, time, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--S', help = 'num_skip')
parser.add_argument('-n', '--N', help = 'num_download')
args = parser.parse_args()

num_skip = int(args.S)
num_download = int(args.N)

failure = []

start = time.time()
N = 3400000
count = 0
count_per_print = num_download / 25
with open('../data/imUrl.csv','r') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		count += 1
		if count <= num_skip:
			continue
		if count > num_skip + num_download:
			break
		asin, imUrl, category = row
		path = os.path.join('../data/img/',category)
		if not os.path.exists(path):
			os.makedirs(path)
		try:
			urllib.urlretrieve(imUrl, os.path.join(path, asin+'.'+imUrl.split('.')[-1]))
		except:
			failure.append((asin, imUrl, category))
		if (count / 500) * 500 == count:
			num_eq = (count-num_skip) / count_per_print
			sys.stdout.write('\r '+str(count)+' / '+str(N)+' [' + '='*num_eq + ' '*(25 - num_eq) + '] - %0.2fs - est. %0.2fs'%(float(time.time()-start), float(time.time()-start) * num_download / (count-num_skip)))
			sys.stdout.flush()
	sys.stdout.write('\r '+str(count-1)+' / '+str(count-1)+' [' + '='*25 + '] - %0.2fs \n\nFinished! \n'%(float(time.time()-start)))
	sys.stdout.flush()
with open('../data/download_failure.csv','w') as csvfile:
	writer = csv.writer(csvfile)
	for row in failure:
		writer.writerow(row)