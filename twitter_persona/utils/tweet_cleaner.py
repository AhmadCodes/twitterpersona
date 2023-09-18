#%% Imports
import re

#%%

def cleanTweet(tweet:str) -> str | None:
    """
    Clean a tweet by removing URLs and mentions

    Parameters
    ----------
    tweet : str
        A tweet

    Returns
    -------
    str | None
        A cleaned tweet or None if the tweet is too short
    """

    # Remove URLs
    tweet = re.sub(r"http\S+", "", tweet)
    
    # Remove mentions
    tweet = re.sub(r"@\S+", "", tweet)
    
    # if word count is less than 5, remove tweet
    
    if len(tweet.split()) < 5:
        return None
    
    return tweet
    
# %% test the tweet cleaner

if __name__ == "__main__":
    
    example_tweet = "@username I love my dog very much, here is a picture: https://www.dog.com/dog.jpg"
    
    print(cleanTweet(example_tweet))

# %%
