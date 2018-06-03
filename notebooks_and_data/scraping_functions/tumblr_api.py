import pytumblr
from selenium import webdriver


def get_client():
    CONSUMER_KEY = 'OOJii0xL1lndypB7OXNALRUOjoh9L4UB9ODnctfIMML9tnBAjj'
    CONSUMER_SECRET = 'jqUWbdwv3RZCq1hOlREXMPzU4k6jWX8WJbM2CK3jltexlv59Kj'
    OAUTH_TOKEN = 'nY8bKKlm6zRhxfF4UxiXq4dECvOklyqmaFIh1IqH6Fb7ENvO7U'
    OAUTH_TOKEN_SECRET = 'k2hoNV78KjEsDuTbeMUWKzV2rXulAU86eUt1K32cgjMfWVC4BP'

    client = pytumblr.TumblrRestClient(
        CONSUMER_KEY,
        CONSUMER_SECRET,
        OAUTH_TOKEN,
        OAUTH_TOKEN_SECRET
    )

    return client

def get_tumblr_usernames_for_hashtags(list_of_hashtags):
    usernames = set()
    for hashtag in list_of_hashtags:
        os.environ['webdriver.chrome.driver'] = 'chromedriver'
        driver = webdriver.Chrome()
        driver.get('https://www.tumblr.com/search/{}'.format(hashtag[1:]))
        usernames = usernames.union(get_tumblr_usernames(driver))
    return usernames

def extract_tumblr_usernames_from_page(driver):
    """
    Get a list of all the usernames from the search results page.
    """

    try:
        username_elements = driver.find_elements_by_class_name('post-info-tumblelog')
        usernames = [element.text for element in username_elements]
    except:
        # default to a single username
        usernames = set(['greasyquotes'])

    return usernames

