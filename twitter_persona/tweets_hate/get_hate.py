#%%

import tweetnlp
try:
    from ..utils.tweet_analysis_percentage import get_percentage
except ImportError:
    import sys
    sys.path.append("..")
    from utils.tweet_analysis_percentage import get_percentage

#%%



#%%

def hate_extraction(tweets_list: list[str],
                    thresh : float = 0.5,
                    debug: bool = False) -> list[str]:
    """
    Extract the hate of a tweet

    Parameters
    ----------
    tweets_list : list[str]
        A list of tweets
    
    thresh : float, optional
        The threshold for the hate score, by default 0.5
    
    debug : bool, optional
        Show debug info if true, by default False

    Returns
    -------
    list[str]
        A list of hates
    """
    
    model = tweetnlp.load_model('hate')  # Or `model = tweetnlp.TopicClassification()`

    
    hates = model.hate(tweets_list, return_probability=True)
    
    if debug:
        print(hates)
    
    all_hates = [hate['probability'] for hate in hates]
    
    # Get the most likely hate
    hates = []
    for hate in all_hates:
        for k in hate.keys():
            if hate[k] > thresh:
                hates.append(str(k))
    
    hates_percentage = get_percentage(hates)
    
    return hates, hates_percentage


#%% test the hate extraction code

if __name__ == "__main__":
    
    example_tweets = ["I love my dog",
                    "go kill yourself",]
    
    print(hate_extraction(example_tweets, debug=True))
    
# %%
