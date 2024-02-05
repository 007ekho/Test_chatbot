import openai
from sentence_transformers import SentenceTransformer
import pinecone
import streamlit as st


model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='db1c8f3b-a76d-4a7a-b476-2937372bb381', environment='gcp-starter')

import os

from pinecone import Pinecone, ServerlessSpec


pc = Pinecone(api_key='db1c8f3b-a76d-4a7a-b476-2937372bb381',environment='gcp-starter')

# Now do stuff

# if 'my_index' not in pc.list_indexes().names():
#     pc.create_index(name='my_index', dimension=1536,metric='euclidean',spec=ServerlessSpec(cloud='aws',region='us-west-2'))


index = pinecone.Index('ai-assistant')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    
    model="gpt-3.5-turbo-instruct",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    # messages=messages,
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']
    

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


