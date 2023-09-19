#%%
import tweetnlp

try:
    from ..utils.tweet_analysis_percentage import get_percentage
except ImportError:
    import sys
    sys.path.append("..")
    from utils.tweet_analysis_percentage import get_percentage
    
    
#%%

model = tweetnlp.load_model('emotion') 
#%%

def emotion_extraction(tweets_list: list[str],
                       thresh : float = 0.7,
                       top_k : int = 3,
                       debug : bool = False) -> tuple[list[str], list[str]]:
    """
    Extract the emotion from the tweets

    Parameters
    ----------
    tweets_list : list[str]
        The list of tweets
    thresh : float, optional
        _description_, by default 0.7
    top_k : int, optional
        _description_, by default 3
    debug : bool, optional
        _description_, by default False

    Returns
    -------
    tuple[list[str], list[str]]
        _description_
    """
    
    emotions = []
    
    for tweet in tweets_list:
        e = model.emotion(tweet,
                                return_probability=True)
        emotions.append(e)
        
    if debug:
        print(emotions)
    
    
    all_emotions = [emotion['probability'] for emotion in emotions]
    
    
    # Get the most likely topic
    emotions = []
    for emotion in all_emotions:
        for k in emotion.keys():
            if emotion[k] > thresh:
                emotions.append(str(k))

    emotions_percentage = get_percentage(emotions)

    return emotions, emotions_percentage

#%% Test the emotions extractor

if __name__ == "__main__":
    
    tweets_list = ["I am happy", "I am sad", "I am angry", "I am excited", "I am bored"]
    
    print(emotion_extraction(tweets_list, debug=True))
# %%
