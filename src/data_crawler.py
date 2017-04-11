from bs4 import BeautifulSoup
import os, requests, urllib
from utils_database import *

def collect_image_amazon(db, cursor, product, num = 50):
	page = 0
	count = 0
	urls = []
	while True:
		page += 1
		r = requests.get('https://www.amazon.com/s/?page='+str(page)+'&keywords='+'+'.join(product.split()))
		soup = BeautifulSoup(r.content, 'html.parser')
		ProductCard = soup.findAll('a', {'class':'a-link-normal a-text-normal'})
		for item in ProductCard:
			if item.find('img') != None:
				url = item.find('img').attrs['src']
				urls.append(url)
				count += 1
		if count >= num:
			break
	urls = urls[:num]
	insert_sImg(db, cursor, urls, product)
	return urls

def download_image(url, name, path):
	urllib.urlretrieve(url,os.path.join(path, name))

def download_images(urls, product, path):
	if not os.path.exists(path):
		os.makedirs(path)
	for i in xrange(len(urls)):
		url = urls[i]
		name = product+'_'+str(i)+'.'+url.split('.')[-1]
		download_image(url, name, path)

if __name__ == '__main__':
	db_info = {'host':,'user':,'passwd':,'port':,'db':} # db_info
	db, cursor = connect_database(db_info) # connect database
	product = 'jersey'
	urls = collect_image_amazon(db, cursor, product, 20) # collect image urls
	path = '../data/'+product
	download_images(urls, product, path) # download images