# coding:utf-8
# 用字串分割取得網域名
from collections import Counter

def get_domain(email_address):
    """split on '@' and return the last piece"""
    return email_address.lower().split("@")[-1]

with open('email.txt', 'r') as f:
    domain_counts = Counter(get_domain(line.strip())
                            for line in f
                            if "@" in line)
    print domain_counts


import csv
def process(date, symbol, price):
    print date, symbol, price

# tab切割
print "---tab delimited stock prices---"
with open('tab_delimited_stock_prices.txt', 'rb') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        date = row[0]
        symbol = row[1]
        closing_price = float(row[2])
        process(date, symbol, closing_price)

# :分割
print "---colon delimited stock prices---"
with open('colon_delimited_stock_prices.txt', 'rb') as f:
    reader = csv.DictReader(f, delimiter=':')
    for row in reader: # 有給欄位名稱可以用欄位名稱去讀取
        date = row["date"]
        symbol = row["symbol"]
        closing_price = float(row["closing_price"])
        process(date, symbol, closing_price)

# 寫檔用,隔開，也可用tab \t
print "---writing out comma_delimited_stock_prices.txt---"
today_prices = { 'Chinese' : 90.5, 'English' : 41.68, 'Math' : 64.5 }
with open('comma_delimited_stock_prices.txt', 'wb') as f: #檔名
    writer = csv.writer(f, delimiter=',') # 也可用\t
    for stock, price in today_prices.items():
        writer.writerow([stock, price])

# 讀取網址
from bs4 import BeautifulSoup
import requests

print "BeautifulSoup"
html = requests.get("http://www.nfu.edu.tw").text
soup = BeautifulSoup(html)
print soup

# 抓取tag標籤與用正規表達式
import re
def book_info(td):
    """given a BeautifulSoup <td> Tag representing a book,
    extract the book's details and return a dict"""
    title = td.find("div", "thumbheader").a.text
    by_author = td.find('div', 'AuthorName').text
    authors = [x.strip() for x in re.sub("^By ", "", by_author).split(",")]
    isbn_link = td.find("div", "thumbheader").a.get("href")
    isbn = re.match("/product/(.*)\.do", isbn_link).groups()[0]
    date = td.find("span", "directorydate").text.strip()
    return {
        "title": title,
        "authors": authors,
        "isbn": isbn,
        "date": date
    }

# 判斷是否為 video
from time import sleep
def is_video(td):
    """it's a video if it has exactly one pricelabel, and if
    the stripped text inside that pricelabel starts with 'Video'"""
    pricelabels = td('span', 'pricelabel')
    return (len(pricelabels) == 1 and
            pricelabels[0].text.strip().startswith("Video"))

# 爬取網頁，共有31頁，每頁抓完讓他休息3秒
def scrape(num_pages=31):
    base_url = "http://shop.oreilly.com/category/browse-subjects/" + \
               "data.do?sortby=publicationDate&page="
    books = []
    for page_num in range(1, num_pages + 1):
        print "souping page", page_num
        url = base_url + str(page_num)
        soup = BeautifulSoup(requests.get(url).text, 'html5lib')
        for td in soup('td', 'thumbtext'):
            if not is_video(td):
                books.append(book_info(td))
        # now be a good citizen and respect the robots.txt!
        sleep(3)
        print len(books) # 印出有幾本書
    return books
scrape()

# 計算date的年份
def get_year(book):
    """book["date"] looks like 'November 2014' so we need to
    split on the space and then take the second piece"""
    return int(book["date"].split()[1])

# 將年分與books個數用matplotlib畫圖
from matplotlib import pyplot as plt
def plot_years(plt, books):
    # 2014 is the last complete year of data (when I ran this)
    year_counts = Counter(get_year(book) for book in books
                          if get_year(book) <= 2017)
    years = sorted(year_counts)
    book_counts = [year_counts[year] for year in years]
    plt.bar([x - 0.5 for x in years], book_counts)
    plt.xlabel("year")
    plt.ylabel("# of data books")
    plt.title("Data is Big!")
    plt.show()
plot_years(plt,scrape())

# 解析 json
import json
serialized = """{ "title" : "Data Science Book",
                      "author" : "Joel Grus",
                      "publicationYear" : 2014,
                      "topics" : [ "data", "science", "data science"] }"""
# parse the JSON to create a Python object
deserialized = json.loads(serialized)
if "data science" in deserialized["topics"]:
    print deserialized


# 透過 github API撈取，並抓repository
endpoint = "https://api.github.com/users/mauriceHsiao/repos"
repos = json.loads(requests.get(endpoint).text)
for rows in repos:
    data1 = rows["name"] # repository
    print data1


