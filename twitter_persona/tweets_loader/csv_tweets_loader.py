#%%

import pandas as pd

try:
    from ..utils.tweet_cleaner import cleanTweet
except ImportError:
    import sys
    sys.path.append("..")
    from utils.tweet_cleaner import cleanTweet
import os

csv_default_path = os.path.join(os.path.dirname(__file__), "../../sample_data/TweetsElonMusk.csv")

#%%
def loadCSVTweets(csv_path: str = csv_default_path,
                  cleanTweets: bool = False) -> list[str]:
    """
    Load tweets from a csv file

    Parameters
    ----------
    csv_path : str
        The path to the csv file

    cleanTweets : bool, optional
        Whether to clean the tweets, by default False
    
    Returns
    -------
    list[str]
        A list of tweets
    """
    

    # Load the csv file
    df = pd.read_csv(csv_path)

    # Get the tweets
    tweets = df["tweet"].tolist()
    
    if cleanTweets:
        tweets = [cleanTweet(tweet) for tweet in tweets if cleanTweet(tweet) is not None]

    return tweets


#%% Test the csv loader

if __name__ == "__main__":
        
    tweets = loadCSVTweets("../../sample_data/TweetsElonMusk.csv",
                           cleanTweets=True)
    print(tweets)
# %%
