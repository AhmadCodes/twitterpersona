#%% imports

import tweetnlp
try:
    from ..utils.tweet_analysis_percentage import get_percentage
except ImportError:
    import sys
    sys.path.append("..")
    from utils.tweet_analysis_percentage import get_percentage


#%% load model

model = tweetnlp.load_model('irony')  # Or `model = tweetnlp.TopicClassification()`

#%% 

def irony_extraction(tweets_list: list[str],
                     thresh : float = 0.5,
                     debug: bool = False) -> list[str]:
    """
    Extract the irony of a tweet

    Parameters
    ----------
    tweets_list : list[str]
        A list of tweets
    
    thresh : float, optional
        The threshold for the irony score, by default 0.5
    
    debug : bool, optional
        Show debug info if true, by default False

    Returns
    -------
    list[str]
        A list of ironies
    """
    
    ironies = []
    
    for tweet in tweets_list:
        ironies.append(model.irony(tweet, return_probability=True))
    
    
    if debug:
        print(ironies)
        
    ironies = [irony['label'] for irony in ironies]
    
    ironies_percentage = get_percentage(ironies)
                
    return ironies, ironies_percentage


#%% test the irony extraction code

if __name__ == "__main__":
    
    example_tweets = ["I love my dog",
                    "Is this dog any more lovable?",] *1000
    
    print(irony_extraction(example_tweets, debug=True))    
# %%
