import pymysql

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

def initialize_table(db, cursor):
	cursor.execute("""
		CREATE TABLE product (
		id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
		imgURL VARCHAR(255),
		brand VARCHAR(255),
		name VARCHAR(255),
		tag VARCHAR(255),
		timeStamp TIMESTAMP(6)
		)""")

if __name__ == '__main__':
	create_image_table()