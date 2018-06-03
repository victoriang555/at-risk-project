from tumblr_api import get_client
from selenium import webdriver
import os

if __name__ == '__main__':
    client = get_client()

    os.environ['webdriver.chrome.driver'] = 'chromedriver'
    driver = webdriver.Chrome()
    driver.get('https://google.com')


    print(client.info())