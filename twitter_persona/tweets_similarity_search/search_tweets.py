#%%

import tweetnlp

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#%%

class TweetsSimilarity:
    
    """
    A class to get the similarity between loaded tweets and query tweet.
    """
    def __init__(self, 
                 tweets_corpus: list[str],
                 debug = False) -> None:
        
        self.model= tweetnlp.load_model('sentence_embedding')  # Or `model = tweetnlp.SentenceEmbedding()` 

    
        self.vectors = self.model.embedding(tweets_corpus, batch_size=4)
        self.debug = debug
        self.tweets_corpus = tweets_corpus
        
        
    def get_similar_tweets( self, 
                            query_tweet: str,
                            thresh : float = 0.7,
                            top_k : int = 3
                            ) -> list[str]:
        """
        Get the most similar tweets to the query tweet

        Parameters
        ----------
        query_tweet : str
            The query tweet

        thresh : float, optional
            The threshold for the similarity score, by default 0.7

        Returns
        -------
        list[str]
            A list of similar tweets
        """
        
        query_vector = self.model.embedding([query_tweet], batch_size=4)

        # Calculate cosine similarity between the query vector and all embeddings
        similarities = cosine_similarity(query_vector, self.vectors)[0]

        matches_above_threshold = [(self.tweets_corpus[i], similarities[i]) for i in range(len(similarities)) if similarities[i] >= thresh]

        # Sort matches by similarity
        sorted_matches = sorted(matches_above_threshold, key=lambda x: x[1], reverse=True)

        # Get top-k matches
        top_k_matches = [match[0] for match in sorted_matches[:top_k]]

        return top_k_matches

    
    
#%% test the similarity extraction code

if __name__ == "__main__":
        
        example_tweets = ["I love my dog",
                        "Is this dog any more lovable?",
                        "I hate my dog",
                        "I love my cat",
                        "I hate my cat",
                        "I love my dog and my cat",
                        "I hate my dog and my cat",
                        ]
        
        similarity = TweetsSimilarity(example_tweets, debug=True)
        
        print(similarity.get_similar_tweets("who do you love?"))
    
# %%