




def get_related_hashtags(initial_hashtag):
    os.environ['webdriver.chrome.driver'] = 'chromedriver'
    driver = webdriver.Chrome()
    driver.get('https://top-hashtags.com/hashtag/{}'.format(initial_hashtag))
    all_hashtags = []
    for idx in range(1,6):
        hashtags = list(set(driver.find_element_by_id('clip-tags-{}'.format(idx)).text.split()))
        all_hashtags.extend(hashtags)
    return list(set(all_hashtags))                     
