from utils_database import *
import json, sys, time
from db_info import *

db, cursor = connect_database(db_info)

start = time.time()
N = 10000000
count = 0
count_per_print = N / 20
with open('../data/metadata.json', 'r') as f:
	for line in f:
		product = json.loads(json.dumps(eval(line)))
		insert_product(db, cursor, [product])
		count += 1
		if (count / 10000) * 10000 == count:
			num_eq = count / count_per_print
			sys.stdout.write('\r '+str(count)+' / '+str(N)+' [' + '='*num_eq + ' '*(20 - num_eq) + '] - %0.2fs '%(float(time.time()-start)))
			sys.stdout.flush()
sys.stdout.write('\r '+str(count)+' / '+str(count)+' [' + '='*20 + '] - %0.2fs \n\nFinished! '%(float(time.time()-start)))
sys.stdout.flush()