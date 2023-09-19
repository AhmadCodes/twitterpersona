#%%
# Use a pipeline as a high-level helper
import tweetnlp
try:
    from ..utils.tweet_analysis_percentage import get_percentage
except ImportError:
    import sys
    sys.path.append("..")
    from utils.tweet_analysis_percentage import get_percentage


#%%

# MULTI-LABEL MODEL 
model = tweetnlp.load_model('sentiment')  # Or `model = tweetnlp.TopicClassification()`


#%%

def sentiment_extraction(tweets_list: list[str],
                     debug: bool = False) -> list[str]:
    """
    Extract the sentiments of a list of tweets

    Parameters
    ----------
    tweets_list : list[str]
        A list of tweets
    
    debug : bool, optional
        Whether to print debug information, by default False

    Returns
    -------
    list[str]
        A list of sentiments
    """
    sentiments = []
    # Get the sentiment of each tweet
    for tweet in tweets_list:
        sentiments.append(model.sentiment(tweet, return_probability=True))
    
    
    if debug:
        print(sentiments)
    
    sentiments = [sentiment['label'] for sentiment in sentiments]
    
    sentiment_percentage = get_percentage(sentiments)

    return sentiments, sentiment_percentage

#%% test the topic extraction code

if __name__ == "__main__":
    
    example_tweets = ["I love my dog",
                    "I hate my dog",]
    
    example_tweets = example_tweets * 1000
    
    print(sentiment_extraction(example_tweets, debug=True))

# %%
