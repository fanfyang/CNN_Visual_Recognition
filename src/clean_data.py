from utils_database import *
import json, time
from db_info import *

db, cursor = connect_database(db_info)
create_Amazon_product_table(db, cursor)

start = time.time()
N = 0
with open('../data/metadata.json', 'r') as f:
	for line in f:
		product = json.loads(json.dumps(eval(line)))
		insert_product(db, cursor, [product])
		N += 1
		if (N / 10000) * 10000 == N:
			print N
			print product['asin']