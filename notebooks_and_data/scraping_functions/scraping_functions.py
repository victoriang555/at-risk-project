




def get_related_hashtags(initial_hashtag):
    os.environ['webdriver.chrome.driver'] = 'chromedriver'
    driver = webdriver.Chrome()
    driver.get('https://top-hashtags.com/hashtag/{}'.format(initial_hashtag))
    all_hashtags = []
    for idx in range(1,6):
        hashtags = list(set(driver.find_element_by_id('clip-tags-{}'.format(idx)).text.split()))
        all_hashtags.extend(hashtags)
    return list(set(all_hashtags))                     


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



def get_user_posts(username):
    username_json = flatten(client.posts(username))
    all_post_summaries = []
    for idx in range(20):
        post_summary = username_json['posts_{}_summary'.format(idx)]
        all_post_summaries.append([username, post_summary])
    return all_post_summaries



def compile_raw_posts_df(list_of_usernames):
    all_users_posts = []
    for username in list_of_usernames:
        user_posts = get_user_posts(username)
        all_users_posts.extend(user_posts)
    raw_posts_df = pd.DataFrame(all_users_posts)
    return raw_posts_df


client = get_client()
