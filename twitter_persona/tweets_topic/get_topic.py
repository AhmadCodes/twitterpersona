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


#%%

def topic_extraction(tweets_list: list[str],
                     thresh:float = 0.90,
                     debug: bool = False) -> list[str]:
    """
    Extract the topic of a tweet

    Parameters
    ----------
    tweets_list : list[str]
        A list of tweets
        
    thresh : float, optional
        The threshold for the topic score, by default 0.5
    
    debug : bool, optional
        Whether to print debug information, by default False

    Returns
    -------
    list[str]
        A list of topics
    """
    model = tweetnlp.load_model('topic_classification')  # Or `model = tweetnlp.TopicClassification()`


    topics = []
    
    # Get the topic of each tweet
    for tweet in tweets_list:
        topics.append(model.topic(tweet, return_probability=True))
    
    
    if debug:
        print(topics)
    
    all_topics = [topic['probability'] for topic in topics]
    
    
    # Get the most likely topic
    topics = []
    for topic in all_topics:
        for k in topic.keys():
            if topic[k] > thresh:
                topics.append(str(k))

    topics_percentage = get_percentage(topics)

    return topics, topics_percentage

#%% test the topic extraction code

if __name__ == "__main__":
    
    example_tweets = ["I love my dog",
                    "I love my dog",]
    
    print(topic_extraction(example_tweets,
                           thresh=0.2,
                           debug=True))

# %%
