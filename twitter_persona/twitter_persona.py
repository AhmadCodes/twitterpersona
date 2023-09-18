"""Main module."""
#%%
import LLM
from tweets_topic.get_topic import topic_extraction
from tweets_similarity_search.search_tweets import TweetsSimilarity
from tweets_irony.get_irony import irony_extraction
from tweets_sentiment.get_sentiment import sentiment_extraction
from tweets_emotion.get_emotions import emotion_extraction

from tweets_loader.csv_tweets_loader import loadCSVTweets

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
# from langchain.embeddings import ModelScopeEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from chromadb.config import Settings
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import json
import requests
import json
import time
from langchain.embeddings import HuggingFaceEmbeddings


#%%

class Persona:
    
    def __init__(self, 
                 personality_username: str,
                 debug = False) -> None:
        
        
        self.debug = debug
        self.personality_username = personality_username
        
        self.persona_tweets = self.get_tweets(self.personality_username)[:50]
        
        self.tweets_similarity = TweetsSimilarity(self.persona_tweets,
                                                  debug=self.debug)
        
        
    
    def get_tweets(self, personality_username):
        
        return loadCSVTweets(cleanTweets=True)
    
    def get_emotions_prompt_snippet(self):
        
        
        _, emotions_percentage_dict = emotion_extraction(self.persona_tweets,
                                                     thresh=0.7,
                                                     top_k=3,
                                                     debug=self.debug)
        
    
        
        emotions_percentages = [[k,str(v*100)] for k,v in emotions_percentage_dict.items()]
        
        line_seperated_emotions_percentages = "\n".join([f"{k}: {v}%" for k,v in emotions_percentages])
        
        emotions_prompt_snippet = """
        Their Tweets' Emotion Percentages:
        {}
        """.format(line_seperated_emotions_percentages)
        
        return emotions_prompt_snippet
        
        
    def get_topic_prompt_snippet(self):
        
        _, topic_percentages_dict = topic_extraction(self.persona_tweets,
                                              thresh=0.7,
                                              debug=self.debug)
        
        line_seperated_topic_percentages = "\n".join([f"{k}: {(v*100):.2f}%" for k,v in topic_percentages_dict.items()])
        
        topic_prompt_snippet = """
        Their Tweets' Topic Percentages:
        {}
        """.format(line_seperated_topic_percentages)
        
        return topic_prompt_snippet
    
    
    def get_sentiment_prompt_snippet(self):
        
        _, sentiment_percentages_dict = sentiment_extraction(self.persona_tweets,
                                                      debug=self.debug)
        
        line_seperated_sentiment_percentages = "\n".join([f"{k}: {(v*100):.2f}%" for k,v in sentiment_percentages_dict.items()])
        
        sentiment_prompt_snippet = """
        Their Tweets' Sentiment Percentages:
        {}
        """.format(line_seperated_sentiment_percentages)
        
        return sentiment_prompt_snippet
    
    
    def get_irony_prompt_snippet(self):
        
        _, irony_percentages_dict = irony_extraction(self.persona_tweets,
                                              thresh=0.7,
                                              debug=self.debug)
        
        line_seperated_irony_percentages = "\n".join([f"{k}: {(v*100):.2f}%" for k,v in irony_percentages_dict.items()])
        
        irony_prompt_snippet = """
        Their Tweets' Irony Percentages:
        {}
        """.format(line_seperated_irony_percentages)
        
        return irony_prompt_snippet
    
    def get_similar_tweets(self, query_tweet):
        
        matching_tweets = self.tweets_similarity.get_similar_tweets(query_tweet,
                                                                    thresh=0.7,
                                                                    top_k=5)
        
        line_seperated_tweets = "\n".join(matching_tweets)
        
        similar_tweets_prompt_snippet = """
        Their Similar Tweets to the user query:
        {}
        """.format(line_seperated_tweets)
        


        return similar_tweets_prompt_snippet
    
    
    

#%%

import os

class ConversationBot:
    
    def __init__(self, 
                 personality_username: str,
                 debug = False) -> None:
        
        
        self.debug = debug
        self.persona = Persona(personality_username, debug=self.debug)
        

        
        self.NAME = "Elon Musk"
        
        self.personality_username = personality_username
        
        self.get_all_snippets()
        
        model_kwargs = {'device': 'cpu'}
        self.embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs=model_kwargs,)

        self.persist_directory = '~/.db_instructor_xl'
        
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)
        
        CHROMA_SETTINGS = Settings(
                chroma_db_impl='duckdb+parquet',
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        loader = TextLoader("../sample_data/previous_conversation.txt")
        texts_ = loader.load()
        texts = self.text_splitter.create_documents([texts_[0].page_content])
        self.db = Chroma.from_documents(texts, self.embeddings, persist_directory=self.persist_directory)
        self.db.persist()
        
        self.answers = []
        self.conv_buffer = []

        self.avg_retrival_time = 0
        self.avg_generation_time = 0
        self.avg_store_time = 0
        self.avg_total_time = 0
        self.texts = []
        
        self.count = 0
        

    def get_all_snippets(self):
        
        self.emotions_prompt_snippet = self.persona.get_emotions_prompt_snippet()
        self.topic_prompt_snippet = self.persona.get_topic_prompt_snippet()
        self.sentiment_prompt_snippet = self.persona.get_sentiment_prompt_snippet()
        self.irony_prompt_snippet = self.persona.get_irony_prompt_snippet()
        
        self.all_snippets = "\n".join([self.emotions_prompt_snippet,
                                        self.topic_prompt_snippet,
                                        self.sentiment_prompt_snippet,
                                        self.irony_prompt_snippet])
        

    def reply_user(self, query):
        
        self.count +=1
        
        starting_time = time.time()
        query = str(query)
        print(f"Max: {query}\n")
        self.conv_buffer.append(f"Max: {query}")
        vectordb = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)

        #without similarity score
        context = vectordb.similarity_search_with_score(query)
        retriever = vectordb.as_retriever(search_type="mmr")
        context = retriever.get_relevant_documents(query)
        # context = context[0].page_content

        context = context[0].page_content

        # print(f"""Context Fetched for query: "{query}" :\n""",context)
        retrival_time =  time.time() - starting_time
        
        # print(f"Retrieval Time: {retrival_time}")
        time_after_retrieval = time.time() 
        
        last_convo = self.conv_buffer[-20:]
        last_convo_str = "\n".join(last_convo)



        from base_prompt import base_prompt
        

        system_prompt = f"""
        {base_prompt}
        
        
        {self.all_snippets}
        
        The last matching tweets are 
        
        The memory of the previous conversation between {self.NAME} and Max:
        {context}
        
        Continuing from the last conversation where {self.NAME} greets Max with reference to the last conversation:
        {last_convo_str}
        \n"""

        message = query
        prompt = f"{system_prompt}Max: {message}\n {self.NAME}:"
        
        
        # PROMTING THE LLM HERE
        response = LLM.get_response(prompt)
        
        generation_time = time.time() - time_after_retrieval
        
        person = response.replace(prompt, "").split("/n")[0]
        
        
        if person.rfind(":") > 0:
            person = person[:person.rfind(":")]
            
        
        if person[-1] not in ['.', '?', '!']:
            if person.rfind(".") > person.rfind("?") and person.rfind(".") > person.rfind("!"):
                print(f"Truncated from response: {person[person.rfind('.')+1:]}")
                person = person[:person.rfind(".")] + person[person.rfind(".")]
            elif person.rfind("?") > person.rfind(".") and person.rfind("?") > person.rfind("!"):
                print(f"Truncated from response: {person[person.rfind('.')+1:]}")
                person = person[:person.rfind("?")] + person[person.rfind("?")]
                
            elif person.rfind("!") > person.rfind(".") and person.rfind("!") > person.rfind("?"):
                print(f"Truncated from response: {person[person.rfind('.')+1:]}")
                person = person[:person.rfind("!")] + person[person.rfind("!")]
        
        len_response = len(person.split())
        
        time_after_generation = time.time()
        
        print(f"""{self.NAME}: {person} \n\n""")
        self.conv_buffer.append(f"{self.NAME}: {person}")
        self.answers.append(person)
        
        dialog = f"Max: {query}\n{self.NAME}: {person}\n"
        texts.append(dialog)
        if self.count%10 ==0:
            texts_str = "\n".join(texts)
            
            textss = self.text_splitter.create_documents([texts_str])
            db = Chroma.from_documents(textss, self.embeddings, persist_directory=self.persist_directory)
            db.persist()
            texts = []
        
        store_time = time.time() - time_after_generation
        
        
        return person
# %%


if __name__ == "__main__":
    
    bot = ConversationBot("elonmusk")
# %%