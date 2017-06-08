import pymysql, json

def connect_database(db_info):
	db = pymysql.connect(**db_info)
	cursor = db.cursor()
	return db, cursor

def create_image_table(db, cursor):
	cursor.execute("""
		CREATE TABLE sImg (
		id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
		imgURL VARCHAR(255),
		tag VARCHAR(255),
		timeStamp TIMESTAMP(6)
		)""")

def insert_sImg(db, cursor, urls, tag):
	for url in urls:
		cursor.execute("""
			INSERT INTO sImg (imgURL, tag)
			VALUES ('%s', '%s')
			""" %(url, tag))
	db.commit()

def create_Amazon_product_table(db, cursor):
	cursor.execute("""
		CREATE TABLE Amazon_Metadata (
		id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
		asin VARCHAR(255),
		title VARCHAR(255),
		price FLOAT,
		imUrl VARCHAR(255),
		brand VARCHAR(255),
		timeStamp TIMESTAMP(6)
		)""")
	cursor.execute("""
		CREATE INDEX Metadata_index
		ON Amazon_Metadata (asin)
		""")
	cursor.execute("""
		CREATE TABLE Amazon_RelatedProduct (
		id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
		asin VARCHAR(255),
		relation VARCHAR(255),
		asin_related VARCHAR(255),
		timeStamp TIMESTAMP(6)
		)""")
	cursor.execute("""
		CREATE INDEX RelatedProduct_index
		ON Amazon_RelatedProduct (asin)
		""")
	cursor.execute("""
		CREATE TABLE Amazon_SalesRank (
		id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
		asin VARCHAR(255),
		category VARCHAR(255),
		rank INT,
		timeStamp TIMESTAMP(6)
		)""")
	cursor.execute("""
		CREATE INDEX SalesRank_index
		ON Amazon_SalesRank (asin)
		""")
	cursor.execute("""
		CREATE TABLE Amazon_Categories (
		id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
		asin VARCHAR(255),
		category_0 VARCHAR(255),
		category_1 VARCHAR(255),
		category_2 VARCHAR(255),
		category_3 VARCHAR(255),
		category_4 VARCHAR(255),
		category_5 VARCHAR(255),
		category_6 VARCHAR(255),
		category_7 VARCHAR(255),
		category_8 VARCHAR(255),
		category_9 VARCHAR(255),
		timeStamp TIMESTAMP(6)
		)""")
	cursor.execute("""
		CREATE INDEX Categories_index
		ON Amazon_Categories (asin)
		""")
	cursor.execute("""
		CREATE TABLE Amazon_Clean (
		id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
		asin VARCHAR(255),
		imUrl VARCHAR(255),
		category VARCHAR(255),
		timeStamp TIMESTAMP(6)
		)""")
	cursor.execute("""
		CREATE INDEX Clean_index
		ON Amazon_Clean (asin)
		""")

def insert_product(db, cursor, products):
	for product in products:
		if 'asin' not in product:
			continue
		succ = 0
		attrs = 'asin'
		values = '\'' + product['asin'] + '\''
		if 'title' in product:
			attrs += ', title'
			values += ',\'' + product['title'].replace('\'','\'\'') + '\''
			# values += ',\'<missing>\''
		if 'price' in product:
			attrs += ', price'
			values += ',' + str(product['price'])
		if 'imUrl' in product:
			attrs += ', imUrl'
			values += ',\'' + product['imUrl'].replace('\'','\'\'') + '\''
		if 'brand' in product:
			attrs += ', brand'
			values += ',\'' + product['brand'].replace('\'','\'\'') + '\''
			# values += ',\'<missing>\''
		try:
			cursor.execute("""
				INSERT INTO Amazon_Metadata (%s)
				VALUES (%s)
				""" %(attrs, values))
		except:
			# print 'Amazon_Metadata Insertion Failed. asin: ' + product['asin']
			succ = 1

		if succ == 0:
			if 'related' in product:
				for key in product['related']:
					for value in product['related'][key]:
						try:
							cursor.execute("""
								INSERT INTO Amazon_RelatedProduct (asin, relation, asin_related)
								VALUES ('%s', '%s', '%s')
								""" % (product['asin'], key, value))
						except:
							# print 'Amazon_RelatedProduct Insertion Failed. asin: ' + product['asin']
							succ = 1

			if 'salesRank' in product:
				for key in product['salesRank']:
					try:
						cursor.execute("""
							INSERT INTO Amazon_SalesRank (asin, category, rank)
							VALUES ('%s', '%s', '%s')
							""" % (product['asin'], key.replace('\'','\'\''), str(product['salesRank'][key])))
					except:
						# print 'Amazon_SalesRank Insertion Failed. asin: ' + product['asin']
						succ = 1

			if 'categories' in product:
				for category in product['categories']:
					attrs = 'asin'
					values = '\'' + product['asin'] + '\''
					for i in range(min(len(category),10)):
						attrs += ',category_' + str(i)
						values += ',\'' + category[i].replace('\'','\'\'') + '\''
					try:
						cursor.execute("""
							INSERT INTO Amazon_Categories (%s)
							VALUES (%s)
							""" %(attrs.rstrip(','), values.rstrip(',')))
					except:
						# print 'Amazon_Categories Insertion Failed. asin: ' + product['asin']
						succ = 1
		else:
			with open('../data/metadata_failure.json','a') as f:
				json.dump(product,f)
				f.write('\n')
	db.commit()

def retrieve_product_img(db, cursor, asins):
	urls = []
	for asin in asins:
		cursor.execute("""
			SELECT imUrl FROM Amazon_Clean
			WHERE asin = '%s'
			""" %(asin))
		url = cursor.fetchall()[0][0]
		urls.append(url)
	return urls

def retrieve_product_info(db, cursor, asins, cols, table = 'Amazon_Metadata'):
	infos = []
	for asin in asins:
		cursor.execute("""
			SELECT %s FROM %s
			WHERE asin = '%s'
			""" % (','.join(cols), table, asin))
		info = cursor.fetchall()[0]
		infos.append(info)
	return infos

def retrieve_product_page(asins):
	return ['https://www.amazon.com/dp/' + asin for asin in asins]