"""
This opens a file called 'items.csv' and saves all the links from the 
wikipedia pages, into a tab delimited csv.

It begins from the start_url, scraps the page, checks for a 'next page' link.
If there is a 'next page' link, it follows the link and scraps the next page.
This carries on until there is no further 'next page' link.

to run: scrapy runspider 1_1_wikispider.py -o items.csv -t csv
"""

from urllib.parse import urljoin
import scrapy
from bs4 import BeautifulSoup

class logospiderSpider(scrapy.Spider):
    name = 'logospider2'
    start_urls = ['https://commons.wikimedia.org/wiki/Category:Unidentified_logos']

    def parse(self, response):
        
        soup = BeautifulSoup(response.text, 'lxml') # html of webpage

        for link in soup('img'): # finds all img tages in html
            yield {'name': link['alt'], 'link': link['src']} # assigns alt content as name, src content at link

        for next_page in response.xpath("(//a[.='next page']/@href)[1]")[0].extract(): # returns relative url of next page
            url = response.urljoin(next_page)
            yield scrapy.Request(url, self.parse)


