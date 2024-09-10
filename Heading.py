import requests
from bs4 import BeautifulSoup

def extract_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    heading = soup.find('h1').get_text()  # Modify according to the website structure
    full_article = soup.find('div', class_='article-content').get_text()  # Modify according to the structure
    return heading, full_article