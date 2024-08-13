import ollama
import chromadb
import csv
import gc
import chromadb.utils.embedding_functions as embedding_functions
import os
from operator import itemgetter
import numpy as np
import spacy
import re

print("\n           - - The Airqiv Document Explorer  - -       ")
print(" - - Artificially Intelligent Retrieval Query Interpretive Visualizer - -")
print("                     - - airqiv.com  - -       ")
print("                        - - :-) - -         \n")
print("\nArqiv AI-Assistant Document Explorer")
print("Copyright (c) <2024>, <Paul Bjerk>")
print("All rights reserved.")
print("This source code is licensed under the BSD2-style license found at https://opensource.org/license/bsd-2-clause .\n")
print("The app leverages open-sourced LLMs using the Ollama app. For more information see https://ollama.com ")
print("The app leverages a vector database using ChromaDB. For more information see https://docs.trychroma.com ")
print("The app leverages the spaCy python library, for more information see https://spacy.io ")
print("I am grateful for the excellent chunking algorithm by Solano Todeschini, published in Towards Data Science Jul 20, 2023.\n(See https://towardsdatascience.com/how-to-chunk-text-data-a-comparative-analysis-3858c4a0997a ) ")
print("\n The documents returned and summarized by this Document Explorer are copyright of the authors and archival custodian.\n")
#general variables
#embed_model = "snowflake-arctic-embed:335m"
embed_model = "snowflake-arctic-embed:latest"
quick_embed_model = "snowflake-arctic-embed:latest"
embed_model_author = "Snowflake"
nlp = spacy.load('en_core_web_sm')
#embed_model = "mxbai-embed-large:latest"
#embed_model_author = "Mixed Bread"
embed_model_short, embed_model_detail = embed_model.split(":")
embed_model_dimensions = "1024"
embed_model_layers = "24"
inference_model = "phi3:3.8b-mini-128k-instruct-q5_K_M"
inference_model_author = "Microsoft"
#inference_model_author = "Google"
#phi3_model = "2k"
inference_model_window = "8k tokens"
#inference_model = "gemma-8k:latest"
#inference_model = "phi3-14b-12k:latest"
#inference_model_short, inference_model_detail = inference_model.split(":")
#inference_model_author = "Meta"
#inference_model = "phi3:14b-medium-128k-instruct-q4_K_M"
inference_model_short, inference_model_detail = inference_model.split(":")


conv_context = "response"
prompt = "prompt"
response = "response"
general_prompt = ""
redo_general_prompt = "n"
uniquephotos = []
query_documents = []
all_query_documents = []
number_retrieved = 5
folder_contents = []
folder_docs =[]
clear_context = "/clear"
end_subprocess = "/bye"
metadata_key = ["UNIQUEPHOTO"]
query_chunk_length = 50
currentingest = "foldertitle"
hnsw_space = "ip"
prompt = f"You are a history professor. Answer the following question by designating one or more documents that contain relevant information. The documents are identified by a UNIQUEPHOTO: name."
user_question_1 = "What are the main themes in these documents?"
user_term_1 = ""
names_wanted = ""
countries_wanted = ""
sentencesneeded = "3"
general_prompt = "What is the main theme in these documents?"
open_url = ""
archive_name =""
archive_url = ""
n_results = 30
ranked_results = 8
context_limiter = 1

# a higher context limiter number makes it more likely that the retrieved documents will be chunked and ranked rather than fed in their entirety into the LLM context.
# the context limiter is a divisor to test the number of retrieved documents against the maximium context length of the LLM




ollama_models = os.popen("ollama list").read()
if "phi3-14b-12k" in ollama_models and "phi3-16k" in ollama_models and "phi3-8k" in ollama_models:
    print("There are two language models available, which one do you want to use?")
    model_choice = input("For the larger, but slower phi3-14b-12k, type 1: \nFor the smaller but faster phi3-16k, type 2:\n For the smaller model with a smaller context, type 3\nEnter number here: ")
    if model_choice == "1":
        inference_model = "phi3-14b-12k:latest"
        inference_model_window = "12k tokens"
        n_results = 30
        ranked_results = 12
    elif model_choice == "2":
        inference_model = "phi3-16k:latest"
        inference_model_window = "16k tokens"
        n_results = 40
        ranked_results = 16
    else:
        inference_model = "phi3-8k:latest"
        inference_model_window = "8k tokens"
        n_results = 30
        ranked_results = 8
elif "phi3-2k" in ollama_models:
    inference_model = "phi3-2k:latest"
    inference_model_window = "2k tokens"
    n_results = 10
    ranked_results = 2
elif "phi3-4k" in ollama_models:
    inference_model = "phi3-4k:latest"
    inference_model_window = "4k tokens"
    n_results = 20
    ranked_results = 4
elif "phi3-8k" in ollama_models:
    inference_model = "phi3-8k:latest"
    inference_model_window = "8k tokens"
    n_results = 30
    ranked_results = 8
elif "phi3-14b-12k" in ollama_models:
    inference_model = "phi3-14b-12k:latest"
    inference_model_window = "12k tokens"
    n_results = 40
    ranked_results = 12
elif "phi3-16k" in ollama_models:
    inference_model = "phi3-16k:latest"
    inference_model_window = "16k tokens"
    n_results = 40
    ranked_results = 16
elif "phi3-12k" in ollama_models:
    inference_model = "phi3-12k:latest"
    inference_model_window = "12k tokens"
    n_results = 20
    ranked_results = 12
elif "phi3-14b-16k" in ollama_models:
    inference_model = "phi3-14b-16k:latest"
    inference_model_window = "16k tokens"
    n_results = 80
    ranked_results = 16
else:
    inference_model = "phi3:3.8b-mini-128k-instruct-q5_K_M"
    inference_model_window = "2k tokens"
    n_results = 10
    ranked_results = 2


#inference_model = "phi3-14b-16k:latest"
#inference_model_window = "16k tokens"


inference_model_short, inference_model_detail = inference_model.split(":")


print("\nThe " +embed_model_author+" "+embed_model_short+ " vector embedding model has "+embed_model_dimensions+ " dimensions and "+embed_model_layers+" layers. \n  The chosen hnsw space calculation is Inner Product or ip.\n ")
print ("The "+inference_model_author+" "+inference_model_short+" language model has a context window length of "+inference_model_window+".\n")

#This establishes an Ollama embedding model for Chromadb processes
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=embed_model,
)

#this initiates the Chromadb persistent client
client = chromadb.PersistentClient(path="chromadb/phototextvectors")


#these prompt the user to designate a CSV and loads it for queries
print("\nQuery the assistant to explore your documents!")
desired_collection = input("What collection do you want to explore? \n Enter the archive abbreviation or thematic one-word name only, in lower-case letters. (e.g. nara, lbj, vietnam, tanzania): \n")
currentingest = str("all-"+desired_collection+"-documents")
file_path = currentingest+".csv"
if currentingest in [c.name for c in client.list_collections()]:
    collection = client.get_collection(name=currentingest, embedding_function=ollama_ef)
else:
    print("The requested collection does not exist. Please enter it again.")
    desired_collection = input("What collection do you want to explore? \n Enter the archive abbreviation or thematic one-word name only, in lower-case letters. (e.g. nara, lbj, vietnam, tanzania): \n")
    currentingest = str("all-" + desired_collection + "-documents")
    collection = client.get_collection(name=currentingest, embedding_function=ollama_ef)


#functions used

# first user prompt is to ask question of the initially retrieved documents
def first_query(desired_collection):
    user_question_1 = input("AI-Assistant: What do you want to know about? \nUser: ")
    sentencesneeded = input("AI-Assistant: How many sentences do you want in the answer? \nUser: ")
    prompt = f"You are a history professor able to read documents from a collection related to "+desired_collection+" and answer questions with relevant information. Each document is a JSON with associated key values UNIQUEPHOTO: and PHOTOTEXT: . The PHOTOTEXT: value is the text of the document identified by the UNIQUEPHOTO: value. Please state the UNIQUEPHOTO: value for every statement. Answer the following question by designating documents that contain relevant information. Question: " + user_question_1 + "? Please respond with " + sentencesneeded + " sentences."
    #"Some documents have header material that looks like gibberish at the beginning of the document. Ignore this kind of header material.\n"
    return prompt

# topic query is used to follow up with the retrieved documents or a user-selected folder of relevant documents
def topic_query(desired_collection):
    user_question_1 = input("AI-Assistant: What else do you want to ask about this topic? \nUser: ")
    sentencesneeded = input("AI-Assistant: How many sentences do you want in the answer (1-9)? \nUser: ")
    if sentencesneeded == "1":
        sentencesneeded = "one"
    elif sentencesneeded == "2":
        sentencesneeded = "two"
    elif sentencesneeded == "3":
        sentencesneeded = "three"
    elif sentencesneeded == "4":
        sentencesneeded = "four"
    elif sentencesneeded == "5":
        sentencesneeded = "five"
    elif sentencesneeded == "6":
        sentencesneeded = "six"
    elif sentencesneeded == "7":
        sentencesneeded = "seven"
    elif sentencesneeded == "8":
        sentencesneeded = "eight"
    elif sentencesneeded == "9":
        sentencesneeded = "nine"

    prompt = f"You are a history professor able to read documents from a collection related to "+desired_collection+" and answer questions with relevant information. Each document is a JSON with associated key values UNIQUEPHOTO: and PHOTOTEXT: . The PHOTOTEXT: value is the text of the document identified by the UNIQUEPHOTO: value. Please state the UNIQUEPHOTO: value for every statement. Answer the following question by designating documents that contain relevant information. Question: " + user_question_1 + "? Please respond with " + sentencesneeded + " sentences."
    #"Some documents have header material that looks like gibberish at the beginning of the document. Ignore this kind of header material.\n"
    return prompt

#list metadatas compiles a list (without duplicates) of all uniquephoto identifiers of documents found in previous steps
def list_metadata(metadata_list, metadata_key):
    uniquephotos = []
    raw_list = []
    metadata = {}
    for metadata in metadata_list:
        for idx in metadata_key:
            raw_list.append(metadata[idx])
    for x in raw_list:
        if x not in uniquephotos:
            uniquephotos.append(x)
    return uniquephotos

#retrieve documents brings back relevant full-documents in two steps, first by matching the query embedding second by a simple term search
def retrieve_documents(query_embeddings, user_term_1, names_wanted, countries_wanted, sub_collection):
    #first step retrieves relevant chunks via embeddings that match the query embedding
    all_metadatas = []
    retrieved_chunks =[]
    all_chunks= []

    # these serve as error handling, so that if someone doesn't have a specific search term it doesn't return an error or a huge number of irrelevant documents
    if names_wanted =="":
        names_wanted = "no__name__given"
    elif names_wanted == "NONE":
        names_wanted = "no__name__given"
    elif names_wanted == " ":
        names_wanted = "no__name__given"
    elif names_wanted == "none":
        names_wanted = "no__name__given"
    elif names_wanted == "None":
        names_wanted = "no__name__given"
    elif names_wanted == "n":
        names_wanted = "no__name__given"
    elif names_wanted == None:
        names_wanted = "no__name__given"
    else:
        names_wanted = names_wanted

    if countries_wanted =="":
        countries_wanted = "no__country__given"
    elif countries_wanted == "NONE":
        countries_wanted = "no__country__given"
    elif countries_wanted == " ":
        countries_wanted = "no__country__given"
    elif countries_wanted == "none":
        countries_wanted = "no__country__given"
    elif countries_wanted == "None":
        countries_wanted = "no__country__given"
    elif countries_wanted == "n":
        countries_wanted = "no__country__given"
    elif countries_wanted == None:
        countries_wanted = "no__country__given"
    else:
        countries_wanted = countries_wanted

    if user_term_1 =="":
        user_term_1 = "no__term__given"
    elif user_term_1 == "NONE":
        user_term_1 = "no__term__given"
    elif user_term_1 == " ":
        user_term_1 = "no__term__given"
    elif user_term_1 == "none":
        user_term_1 = "no__term__given"
    elif user_term_1 == "None":
        user_term_1 = "no__term__given"
    elif user_term_1 == "n":
        user_term_1 = "no__term__given"
    elif user_term_1 == None:
        user_term_1 = "no__term__given"
    else:
        user_term_1 = user_term_1

    if sub_collection =="":
        sub_collection = "no__subcollection__given"
    elif sub_collection == "NONE":
        sub_collection = "no__subcollection__given"
    elif sub_collection == " ":
        sub_collection = "no__subcollection__given"
    elif sub_collection == "none":
        sub_collection = "no__subcollection__given"
    elif sub_collection == "None":
        sub_collection = "no__subcollection__given"
    elif sub_collection == "n":
        sub_collection = "no__subcollection__given"
    elif sub_collection == None:
        sub_collection = "no__subcollection__given"
    else:
        sub_collection = sub_collection

    ids_list = []
    chunks_list = []
    metadata_in_list = []
    if names_wanted != "no__name__given":
        if sub_collection != "no__subcollection__given":
            retrieved_chunks = collection.query(query_embeddings=query_embeddings, n_results=n_results, where={"NAMESMENTIONED": names_wanted, "SUBCOLLECTION": sub_collection})
            ids_in_list = retrieved_chunks["ids"]
            metadata_in_list = retrieved_chunks["metadatas"]
            chunks_in_list = retrieved_chunks["documents"]
            if len(ids_in_list) < 1:
                ids_list = retrieved_chunks["ids"]
                chunks_in_list = retrieved_chunks["documents"]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list
            else:
                ids_list = ids_in_list[0]
                chunks_list = chunks_in_list[0]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list[0]
            startround = -1
            for item in ids_list:
                nextround = startround + 1
                startround = nextround
                id = item
                item_doc = chunks_list[startround]
                item_list = id.split("-")
                item_suffix = item_list[-1]
                item_part = str("-part-" + item_suffix)
                photo_id = id.replace(item_part, "")
                doc = {"UNIQUEPHOTO": photo_id, "PHOTOTEXT": item_doc}
                all_chunks.append(doc)
                chunks_metadata_list = metadata_in_list[0]
        else:
            retrieved_chunks = collection.query(query_embeddings=query_embeddings, n_results=n_results, where={"NAMESMENTIONED": names_wanted})
            ids_in_list = retrieved_chunks["ids"]
            chunks_in_list = retrieved_chunks["documents"]
            if len(ids_in_list) < 1:
                ids_list = retrieved_chunks["ids"]
                chunks_in_list = retrieved_chunks["documents"]
                metadata_in_list = retrieved_chunks["metadatas"]
            else:
                ids_list = ids_in_list[0]
                chunks_list = chunks_in_list[0]
                metadata_in_list = retrieved_chunks["metadatas"]
            chunks_metadata_list = metadata_in_list[0]
            startround = -1
            for item in ids_list:
                nextround = startround + 1
                startround = nextround
                id = item
                item_doc = chunks_list[startround]
                item_list = id.split("-")
                item_suffix = item_list[-1]
                item_part = str("-part-" + item_suffix)
                photo_id = id.replace(item_part, "")
                doc = {"UNIQUEPHOTO": photo_id, "PHOTOTEXT": item_doc}
                all_chunks.append(doc)
                chunks_metadata_list = metadata_in_list[0]
    elif countries_wanted != "no__country__given":
        if sub_collection != "no__subcollection__given":
            retrieved_chunks = collection.query(query_embeddings=query_embeddings, n_results=n_results, where={"COUNTRIESMENTIONED": countries_wanted, "SUBCOLLECTION": sub_collection})
            ids_in_list = retrieved_chunks["ids"]
            chunks_in_list = retrieved_chunks["documents"]
            if len(ids_in_list) < 1:
                ids_list = retrieved_chunks["ids"]
                chunks_in_list = retrieved_chunks["documents"]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list
            else:
                ids_list = ids_in_list[0]
                chunks_list = chunks_in_list[0]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list[0]
            startround = -1
            for item in ids_list:
                nextround = startround + 1
                startround = nextround
                id = item
                item_doc = chunks_list[startround]
                item_list = id.split("-")
                item_suffix = item_list[-1]
                item_part = str("-part-" + item_suffix)
                photo_id = id.replace(item_part, "")
                doc = {"UNIQUEPHOTO": photo_id, "PHOTOTEXT": item_doc}
                all_chunks.append(doc)
                chunks_metadata_list = metadata_in_list[0]
        else:
            retrieved_chunks = collection.query(query_embeddings=query_embeddings, n_results=n_results, where={"COUNTRIESMENTIONED": countries_wanted})
            ids_in_list = retrieved_chunks["ids"]
            chunks_in_list = retrieved_chunks["documents"]
            if len(ids_in_list) < 1:
                ids_list = retrieved_chunks["ids"]
                chunks_in_list = retrieved_chunks["documents"]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list
            else:
                ids_list = ids_in_list[0]
                chunks_list = chunks_in_list[0]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list[0]
            startround = -1
            for item in ids_list:
                nextround = startround + 1
                startround = nextround
                id = item
                item_doc = chunks_list[startround]
                item_list = id.split("-")
                item_suffix = item_list[-1]
                item_part = str("-part-" + item_suffix)
                photo_id = id.replace(item_part, "")
                doc = {"UNIQUEPHOTO": photo_id, "PHOTOTEXT": item_doc}
                all_chunks.append(doc)
    elif names_wanted != "no__name__given" and countries_wanted != "no__countries__wanted":
        if sub_collection != "no__subcollection__given":
            retrieved_chunks = collection.query(query_embeddings=query_embeddings, n_results=n_results, where={"NAMESMENTIONED": names_wanted, "COUNTRIESMENTIONED": countries_wanted, "SUBCOLLECTION": sub_collection})
            ids_in_list = retrieved_chunks["ids"]
            chunks_in_list = retrieved_chunks["documents"]
            if len(ids_in_list) < 1:
                ids_list = retrieved_chunks["ids"]
                chunks_in_list = retrieved_chunks["documents"]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list
            else:
                ids_list = ids_in_list[0]
                chunks_list = chunks_in_list[0]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list[0]
            startround = -1
            for item in ids_list:
                nextround = startround + 1
                startround = nextround
                id = item
                item_doc = chunks_list[startround]
                item_list = id.split("-")
                item_suffix = item_list[-1]
                item_part = str("-part-" + item_suffix)
                photo_id = id.replace(item_part, "")
                doc = {"UNIQUEPHOTO": photo_id, "PHOTOTEXT": item_doc}
                all_chunks.append(doc)
                chunks_metadata_list = metadata_in_list[0]
        else:
            retrieved_chunks = collection.query(query_embeddings=query_embeddings, n_results=n_results, where={"NAMESMENTIONED": names_wanted, "COUNTRIESMENTIONED": countries_wanted})
            ids_in_list = retrieved_chunks["ids"]
            chunks_in_list = retrieved_chunks["documents"]
            if len(ids_in_list) < 1:
                ids_list = retrieved_chunks["ids"]
                chunks_in_list = retrieved_chunks["documents"]
                metadata_in_list = retrieved_chunks["metadatas"]
            else:
                ids_list = ids_in_list[0]
                chunks_list = chunks_in_list[0]
                metadata_in_list = retrieved_chunks["metadatas"]
            chunks_metadata_list = metadata_in_list[0]
            startround = -1
            for item in ids_list:
                nextround = startround + 1
                startround = nextround
                id = item
                item_doc = chunks_list[startround]
                item_list = id.split("-")
                item_suffix = item_list[-1]
                item_part = str("-part-" + item_suffix)
                photo_id = id.replace(item_part, "")
                doc = {"UNIQUEPHOTO": photo_id, "PHOTOTEXT": item_doc}
                all_chunks.append(doc)
                chunks_metadata_list = metadata_in_list[0]
    else:
        if sub_collection != "no__subcollection__given":
            retrieved_chunks = collection.query(query_embeddings=query_embeddings, n_results=n_results, where={"SUBCOLLECTION": sub_collection})
            ids_in_list = retrieved_chunks["ids"]
            chunks_in_list = retrieved_chunks["documents"]
            if len(ids_in_list) < 1:
                ids_list = retrieved_chunks["ids"]
                chunks_in_list = retrieved_chunks["documents"]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list
            else:
                ids_list = ids_in_list[0]
                chunks_list = chunks_in_list[0]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list[0]
            startround = -1
            for item in ids_list:
                nextround = startround + 1
                startround = nextround
                id = item
                item_doc = chunks_list[startround]
                item_list = id.split("-")
                item_suffix = item_list[-1]
                item_part = str("-part-" + item_suffix)
                photo_id = id.replace(item_part, "")
                doc = {"UNIQUEPHOTO": photo_id, "PHOTOTEXT": item_doc}
                all_chunks.append(doc)
                chunks_metadata_list = metadata_in_list[0]
        else:
            retrieved_chunks = collection.query(query_embeddings=query_embeddings, n_results=n_results)
            ids_in_list = retrieved_chunks["ids"]
            chunks_in_list = retrieved_chunks["documents"]
            if len(ids_in_list) < 1:
                ids_list = retrieved_chunks["ids"]
                chunks_in_list = retrieved_chunks["documents"]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list
            else:
                ids_list = ids_in_list[0]
                chunks_list = chunks_in_list[0]
                metadata_in_list = retrieved_chunks["metadatas"]
                chunks_metadata_list = metadata_in_list[0]
            startround = -1
            for item in ids_list:
                nextround = startround + 1
                startround = nextround
                id = item
                item_doc = chunks_list[startround]
                item_list = id.split("-")
                item_suffix = item_list[-1]
                item_part = str("-part-" + item_suffix)
                photo_id = id.replace(item_part, "")
                doc = {"UNIQUEPHOTO": photo_id, "PHOTOTEXT": item_doc}
                all_chunks.append(doc)
                chunks_metadata_list = metadata_in_list[0]

    for item in chunks_metadata_list:
        all_metadatas.append(item)

    for item in chunks_in_list:
        all_chunks.append(item)

    #second step uses the user_term as a search term to find more matching chunks
    retrieved_documents = collection.get(ids=[], where_document={"$contains":user_term_1})
    if len(ids_in_list) < 1:
        ids_list = retrieved_documents["ids"]
        metadata_in_list = retrieved_documents["metadatas"]
    else:
        ids_list = ids_in_list[0]
        chunks_list = chunks_in_list[0]
        metadata_in_list = retrieved_documents["metadatas"]
    startround = -1
    for item in ids_list:
        nextround = startround + 1
        startround = nextround
        id = str(item)
        item_doc = chunks_list[startround]
        item_list = id.split("-")
        item_suffix = item_list[-1]
        item_part = str("-part-" + item_suffix)
        photo_id = id.replace(item_part, "")
        doc = {"UNIQUEPHOTO": photo_id, "PHOTOTEXT": item_doc}
        all_chunks.append(doc)

    retrieved_docs_metadata_list = metadata_in_list
    for item in retrieved_docs_metadata_list:
        all_metadatas.append(item)
    if len(all_metadatas) < 1:
        uniquephotos = []
    else:
        uniquephotos = list_metadata(all_metadatas, metadata_key)
    return uniquephotos, all_chunks



#get ranked documents allows users to narrow down a retrived set of documents, to better fit context window low RAM situations
def get_ranked_documents(query_documents, general_prompt, query_chunk_length, ranked_results, file_path, metadata_key):
    client = chromadb.PersistentClient(path="chromadb/phototextvectors")
    embed_model = quick_embed_model
    if "temp_collection" in [c.name for c in client.list_collections()]:
        client.delete_collection(name="temp_collection")
        retrieved_documents = []
        retrieved_metadatas = []
        retrieved_ids = []
    else:
        retrieved_documents = []
        retrieved_metadatas = []
        retrieved_ids = []
    #print(query_documents)
    for i in query_documents:
        #for key in i.keys():
        uniquephoto = i["UNIQUEPHOTO"]
        phototext = i["PHOTOTEXT"]
        metadata = {"UNIQUEPHOTO": uniquephoto}
        doc_chunks = []
        for chunk in chunker(phototext, query_chunk_length):
            doc_chunks.append(chunk)
            retrieved_documents.append(chunk)
            retrieved_metadatas.append(metadata)
        for item in doc_chunks:
            id_item = uniquephoto
            id_index = doc_chunks.index(item) + 1
            id_suffix = str(id_index)
            id = id_item + "-chunk-part-" + id_suffix
            retrieved_ids.append(id)
    #print(retrieved_ids)
    collection = client.create_collection(name="temp_collection", embedding_function=ollama_ef)
    collection.add(
        documents=retrieved_documents,
        metadatas=retrieved_metadatas,
        ids=retrieved_ids
    )
    query_embeddings = ollama_ef(general_prompt)
    ranked_chunks = collection.query(query_embeddings=query_embeddings, n_results=int(ranked_results/context_limiter))
    metadata_in_list = ranked_chunks["metadatas"]
    chunks_metadata_list = metadata_in_list[0]
    all_metadatas = []
    for item in chunks_metadata_list:
        all_metadatas.append(item)
    uniquephotos = list_metadata(all_metadatas, metadata_key)
    ranked_docs, copyright_notice = get_documents(uniquephotos, file_path)
    client.delete_collection(name="temp_collection")
    return ranked_docs, uniquephotos

#using the return from the retrieve documents, get documents returns a set of full-text documents, not just chunks
def get_documents(uniquephotos, file_path):
  with (open(file_path, newline="") as csv_file):
    data = csv.DictReader(csv_file)
    query_documents = []
    for row in data:
        uniquephoto = row["UNIQUEPHOTO"]
        phototext = row["PHOTOTEXT"]
        copyright_notice = row["COPYRIGHT"]
        for item in uniquephotos:
            if row["UNIQUEPHOTO"] == item:
                query_document = {"UNIQUEPHOTO":uniquephoto, "PHOTOTEXT":phototext}
                query_documents.append(query_document)
            #else:
                #print("This item reference is not in the database. Please check the reference")
    #print(query_documents)
    for i in query_documents:
        phototext = i["PHOTOTEXT"]
        text = re.sub('\\s[3][01]\\s|\\s[.][3][01]\\s|[.]\\s[12][0-9]\\s|\\s[12][0-9]\\s|\\s[1-9]\\s|[.]\\s[1-9]\\s',' ', phototext)
        text = re.sub('\\s[U][S]', '%@%@%', text)
        text = re.sub('[U][K]', '@%@%@', text)
        text = re.sub('\\s[A-Z][A-Z]\\s', ' ', text)
        text = re.sub('%@%@%', 'US ', text)
        phototext = re.sub('@%@%@', 'UK ', text)
        i["PHOTOTEXT"] = phototext


    return query_documents, copyright_notice

def get_cited_documents(desired_quote, file_path):
    all_metadatas = []
    retrieved_documents = collection.get(ids=[], where_document={"$contains": desired_quote})
    metadata_in_list = retrieved_documents["metadatas"]
    retrieved_docs_metadata_list = metadata_in_list
    for item in retrieved_docs_metadata_list:
        all_metadatas.append(item)
    uniquephotos = list_metadata(all_metadatas, metadata_key)
    query_documents = get_documents(uniquephotos, file_path)
    return query_documents

def get_desired_doc(file_path):
    view_doc = []
    desired_doc = str(input("To view the original text of one designated document,\n cut and paste the UNIQUEPHOTO name here, or just enter n to continue: "))
    view_doc.append(desired_doc)
    doc_text, copyright_notice = get_documents(view_doc, file_path)
    #clean_doc = re.sub('\\s[3][01]\\s|\\s[.][3][01]\\s|[.]\\s[12][0-9]\\s|\\s[12][0-9]\\s|\\s[1-9]\\s|[.]\\s[1-9]\\s', ' ', doc_text)

    print("\nHere is the full text of document " + desired_doc + "\n--")
    print(doc_text)
    print("--\n")
    view_website = input("Would you like to open the website of this document? Type y or n: ")
    if view_website == "y":
        open_url = open_website(view_doc, file_path)
        if open_url == "" or open_url is None:
            open_url = archive_url
        else:
            open_url = open_url
        print("Now opening the website in your Safari browser, find the page for " + desired_doc)
        open_website_command = str("open '/Applications/Safari.app' '" + open_url + "'")
        os.system(open_website_command)
        #print("Please find your Safari browser window to view " + open_url)
    else:
        print("Okay, you can retrieve this website again later.")



def open_website(uniquephotos, file_path):
    open_url = ""
    page_ref = ""
    with (open(file_path, newline="") as csv_file):
        data = csv.DictReader(csv_file)
        #open_url = ""
        #page_ref = ""
        #query_documents = []
        for row in data:
            #uniquephoto = row["UNIQUEPHOTO"]
            #phototext = row["PHOTOTEXT"]
            page_ref = row["UNIQUEPHOTO"]
            url = row["URL"]
            for item in uniquephotos:
                if row["UNIQUEPHOTO"] == item:
                    open_url = str(url)
        return open_url
def get_namesmentioned(uniquephotos):
  with (open(file_path, newline="") as csv_file):
    data = csv.DictReader(csv_file)
    names_mentioned = []
    for row in data:
        uniquephoto = row["UNIQUEPHOTO"]
        #phototext = row["PHOTOTEXT"]
        names = row["NAMESMENTIONED"]
        for item in uniquephotos:
            if row["UNIQUEPHOTO"] == item:
                name_mentioned = str(uniquephoto+": "+names)
                names_mentioned.append(name_mentioned)
    return names_mentioned

# Ollama's basic query-documents function using selected LLM
# phi3 chat template: <|user|>\nQuestion<|end|>\n<|assistant|>
def response_generation(data,prompt,inference_model):
    #  generate a response combining the prompt and data we retrieved in step 2
    output = ollama.generate(
        model=inference_model,
        prompt=f"<|user|>\nUsing this data: {data}. Respond to this prompt: {prompt}<|end|\n<|assistant|>"
        #gemma prompt=f"<start_of_turn>user\nUsing this data: {data}. Respond to this prompt: {prompt}<end_of_turn>\n"
    )
    conv_context = output["response"]
    return conv_context


#This simple chunker just splits up s text into chunks of n length
# the chunker could be improved with overlapping chunks and some recursive techniques for smarter chunking
#def chunker (s, n):
    #"""Produce `n`-character chunks from `s`."""
    #for start in range(0, len(s), n):
        #yield s[start:start+n]

#Text Processing: Each text chunk is passed to the process function. This function uses the SpaCy library to create sentence embeddings, which are used to represent the semantic meaning of each sentence in the text chunk.
def process(text):
    doc = nlp(text)
    sents = list(doc.sents)
    vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])
    return sents, vecs

def cluster_text(sents, vecs, threshold):
    clusters = [[0]]
    for i in range(1, len(sents)):
        if np.dot(vecs[i], vecs[i - 1]) < threshold:
            clusters.append([])
        clusters[-1].append(i)
    return clusters

def clean_text(text):
    # Add your text cleaning process here
    return text

def average_elements(lst):
    if len(lst) !=0:
        avg = sum(lst) / len(lst)
    else:
        avg = sum(lst) / 1

    return int(avg)

def chunker (phototext,chunk_length):
    """Produce `n`-character chunks from `s`."""
    # Initialize the clusters lengths list and final texts list
    clusters_lens = []
    final_texts = []
    text = phototext

    # If the cosine similarity is less than a specified threshold, a new cluster begins.
    threshold = 0.3

    # Process the chunk
    # This function uses the SpaCy library to create sentence embeddings,
    # which are used to represent the semantic meaning of each sentence in the text chunk.
    sents, vecs = process(text)

    # Cluster the sentences
    # The cluster_text function forms clusters of sentences based on the cosine similarity of their embeddings.
    # If the cosine similarity is less than a specified threshold, a new cluster begins.
    clusters = cluster_text(sents, vecs, threshold)

    for cluster in clusters:
        cluster_txt = clean_text(' '.join([sents[i].text for i in cluster]))
        cluster_len = len(cluster_txt)

        # Length Check: The code then checks the length of each cluster.
        # If a cluster is too short (less than 60 characters) or too long (more than 3000 characters),
        # the threshold is adjusted and the process repeats for that particular cluster until an acceptable length is achieved.
        # Check if the cluster is too short
        if cluster_len < chunk_length:
            continue

        # Check if the cluster is too long
        elif cluster_len > 1000:
            threshold = 0.4
            sents_div, vecs_div = process(cluster_txt)
            reclusters = cluster_text(sents_div, vecs_div, threshold)

            for subcluster in reclusters:
                div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
                div_len = len(div_txt)

                if div_len < chunk_length or div_len > 1000:
                    continue

                clusters_lens.append(div_len)
                final_texts.append(div_txt)

        else:
            clusters_lens.append(cluster_len)
            final_texts.append(cluster_txt)

    avg_cluster_len = int(average_elements(clusters_lens))
    number_of_clusters = int(len(final_texts))
    total_chars = avg_cluster_len * number_of_clusters

    return final_texts


#the get folder function allows the user to request the entire enclosing collection of a single document
def get_folder(folder_contents):
    sorted_query_documents = []
    with (open(file_path, newline="") as csv_file):
        data = csv.DictReader(csv_file)
        query_documents = []
        for row in data:
            uniquephoto = row["UNIQUEPHOTO"]
            phototext = row["PHOTOTEXT"]
            for item in folder_contents:
                if row["FOLDERNAME"] == item:
                    query_document = {"UNIQUEPHOTO": uniquephoto, "PHOTOTEXT": phototext}
                    query_documents.append(query_document)
        sorted_query_documents = sorted(query_documents, key=itemgetter('UNIQUEPHOTO'), reverse=True)
        return sorted_query_documents

def get_general_prompt(file_path):
    print("\n")
    initial_prompt = input("AI-Assistant: What is the general topic you want to know about? \nUser: ")
    general_prompt = ("Represent this sentence for searching relevant passages: "+initial_prompt)
    user_term_1 = input("To EXPAND the number of retrieved documents, please provide ONE specific term (name, organization event) relevant to your question.\n If you don't want to specify a search term, type - NONE. \n Or enter a one-word search term here. \nUser: ")
    names_wanted = input("To LIMIT the number of retrieved documents to those authored by (or associated with) a single name, provide ONE name. \n If you don't want to limit your retrieved documents type - NONE. \n Or enter the full indexed name delimiter here. \nUser: ")
    countries_wanted = input("To LIMIT the number of retrieved documents to those associated with a single country, provide ONE country. \n If you don't want to limit your retrieved documents type - NONE. \n Or enter the full indexed country delimiter here. \nUser: ")
    sub_collection = input("To LIMIT the number of retrieved documents to those associated with a single sub-collection (e.g oh: oral histories or RG59: NARA Record Group 59), provide ONE abbreviation. \n If you don't want to limit your retrieved documents type - NONE. \n Or enter the abbreviated sub-collection delimiter here. \nUser: ")


    #countries_wanted = countries_wanted.lower()
    sub_collection = sub_collection.lower()
    #embed the query and find matching chunks
    query_embeddings = ollama_ef(general_prompt)
    uniquephotos, all_chunks = retrieve_documents(query_embeddings, user_term_1, names_wanted, countries_wanted, sub_collection)

    print(uniquephotos)
    number_retrieved = len(uniquephotos)
    print("The " + str(number_retrieved) + " documents listed above have content matching your query.\n If no documents are listed, please enter a different query term below.\n")

    see_names = input("Would you like to see the names associated with these documents? y or n: ")
    if see_names == "y":
        print("\nSee names associated with the retrieved documents with page references:")
        names_mentioned = get_namesmentioned(uniquephotos)
        for i in names_mentioned:
            print(i)
        #view_desired_doc = "n"
        view_desired_doc = input("\nWould you like to view any of the referenced documents? y/n: ")
        while view_desired_doc != "n":
            get_desired_doc(file_path)
            view_desired_doc = input("Would you like to view another of the referenced documents? y/n: ")
    else:
        print("Okay, you can ask for names again later.")


    all_query_documents, copyright_notice = get_documents(uniquephotos, file_path)
    return number_retrieved, all_query_documents, general_prompt, copyright_notice, uniquephotos, all_chunks

# Here is where the main program starts

userresponse = "c"
conv_continue = "y"
copyright_notice = str("Copyright of " + desired_collection + ". Fair use criteria of Section 107 of the Copyright Act of 1976 must be followed. The following materials can be used for educational and other noncommercial purposes without the written permission of " + desired_collection + ". These materials are not to be used for resale or commercial purposes without written authorization from " + desired_collection + ". This text extraction and summarization process and software is designed and copyrighted by Paul Bjerk. All materials cited must be attributed to the " + desired_collection + ", and The Airqiv Document Explorer at airqiv.com ")

# The first while loop prompts the user for a query, if the query returns too many or too few documents, the user can re-phrase it in an inner while loop
#it seems helpful allow the user to clear the context window of the LLM from time to time

while userresponse != "q":
    gc.collect()
    print("\nIf you've been working for a while, it is helpful at this point to clear the LLM context window, otherwise it gets confused.\n But if you are just starting a new session, you don't need to do this.")
    user_clear = input("Would you like to clear the context window? ")
    if user_clear == "y":
        print("Wait a moment, and when you see the >>> prompt, type: "+clear_context+"\n When the >>> appears again, type: " +end_subprocess+ "\n Typing these two entries will clear the context window of the LLM")
        #os.system("cd")
        os.system("ollama run " + inference_model)

    #retrieve full document texts from CSV and string them together into a list of strings that can be entered into LLM context window
    number_retrieved, all_query_documents, general_prompt, copyright_notice, uniquephotos, all_chunks = get_general_prompt(file_path)

    view_docs = input("Would you like to view all the documents matching your general query? Type: y or n: ")
    if view_docs != "n":
        print("\n--")
        print(all_query_documents)
        print("--\n")
        print("See the retrieved documents above. \nNow enter a more specific question below or enter a new query.")
    else:
        print("Great. Now enter a more specific question below or enter a new query.")
    print("If more than " + str(int(ranked_results / context_limiter)) + " documents are listed, \nthe AI-Assistant will re-read them and retrieve only the most relevant documents.\n You may wish to enter a new query to retrieve a smaller number of documents.\n")
    redo_general_prompt = input("\nWould you like to enter a different query? Type: y or n: ")


#this inner loop allows the user to request a different set of documents
    while redo_general_prompt == "y":
        number_retrieved, all_query_documents, general_prompt, copyright_notice, uniquephotos, all_chunks = get_general_prompt(file_path)

        view_docs = input("\nWould you like to view all the documents matching your general query? Type: y or n: ")
        if view_docs != "n":
            print("View the full set of documents matching your query below: \n--")
            print(all_query_documents)
            print("--\n")
            print("See the retrieved documents above.")
            print("You can choose to input a new query term below, by typing y or query this set of documents by typing n below.")
        else:
            print("You can choose to input a new query term below, by typing y or query this set of documents by typing n below.")
        print("\nIf more than " + str(int(ranked_results / context_limiter)) + " documents are listed, \nthe AI-Assistant will re-read them and retrieve only the most relevant documents.\n You may wish to enter a new query to retrieve a smaller number of documents.\n")
        redo_general_prompt = input("Would you like to enter a different query term? Type: y or n: ")

    userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
    if userresponse == "q":
        #print(copyright_notice)
        exit()
    else:
        print("Let's continue.\n")

    gc.collect()
    if desired_collection == "nara":
        archive_name = "The United States National Archives"
        archive_url = "archives.gov"
    elif desired_collection == "jfk":
        archive_name = "The John F. Kennedy Presidential Library"
        archive_url = "jfklibrary.org"
    elif desired_collection == "lbj":
        archive_name = "The Lyndon Baines Johnson Presidential Library"
        archive_url = "lbjlibrary.org"
    elif desired_collection == "rmn":
        archive_name = "The Richard M. Nixon Presidential Library"
        archive_url = "nixonlibrary.gov"
    elif desired_collection == "grf":
        archive_name = "The Gerald R. Ford Presidential Library"
        archive_url = "fordlibrarymuseum.gov"
    elif desired_collection == "jec":
        archive_name = "The Jimmy Carter Presidential Library"
        archive_url = "jimmycarterlibrary.gov"
    elif desired_collection == "rwb":
        archive_name = "The Ronald Reagan Presidential Library"
        archive_url = "reaganlibrary.gov"
    elif desired_collection == "ghwb":
        archive_name = "The George H.W. Bush Library"
        archive_url = "bush41.org"
    elif desired_collection == "imf":
        archive_name = "The International Monetary Fund (IMF)"
        archive_url = "imf.org"
    elif desired_collection == "pro":
        archive_name = "The United Kingdom National Archives, formerly the Public Record Office (PRO)"
        archive_url = "nationalarchives.gov.uk"
    elif desired_collection == "tna":
        archive_name = "The Tanzania National Archives"
        archive_url = "nyaraka.go.tz"
    elif desired_collection == "kna":
        archive_name = "The Kenya National Archives"
        archive_url = "archives.go.ke"
    elif desired_collection == "ttuva":
        archive_name = "The Vietnam Archive at Texas Tech University"
        archive_url = "vva.vietnam.ttu.edu"
    else:
        archive_name = desired_collection
        archive_url = str("For more information search for the archives of " + desired_collection)

    conv_context = "response"
    prompt = first_query(desired_collection)

    if number_retrieved > int(ranked_results/context_limiter):
        #query_documents, uniquephotos = get_ranked_documents(all_query_documents, general_prompt, query_chunk_length, ranked_results, file_path, metadata_key)
        query_documents = all_chunks
        #print(query_documents)
    else:
        query_documents = all_query_documents
        #query_documents = all_chunks

    #print(query_documents)
    conv_context = response_generation(query_documents, prompt, inference_model)
    print("\nHere is the AI-Assistant's summary of relevant material from the documents: \n--")
    print(conv_context)
    print("--\n")
    view_folder = "n"
    view_desired_doc = "n"
    view_desired_doc = input("\nWould you like to view any of the referenced documents? y/n: ")
    while view_desired_doc != "n":
        get_desired_doc(file_path)
        view_desired_doc = input("Would you like to view another of the referenced documents? y/n: ")

    view_folder = input("Would you like to retrieve all documents in this folder? Type y or n: ")
    folder_contents = []
    if view_folder == "y":
        desired_folder = str(input("Copy the folder name here: \n (i.e. paste the preliminary part of the document name, prior to the final -IMG_ suffix)"))
        folder_contents.append(desired_folder)
        folder_docs = get_folder(folder_contents)
        folder_length = len(folder_docs)
        print("\nHere is the full contents of the folder " + desired_folder + "\n--")
        print(folder_docs)
        print("--\n")
        print("See contents of folder above. There are " + str(folder_length) + " pages in this folder.\n")
        conv_continue = input("\nWould you like to ask questions about the contents of this folder? y/n: ")
        if folder_length > ranked_results / context_limiter:
            folder_docs, uniquephotos = get_ranked_documents(folder_docs, general_prompt, query_chunk_length, ranked_results, file_path, metadata_key)
        else:
            folder_docs = folder_docs

    else:
        folder_docs = query_documents
        print("We'll continue asking questions of the originally retrieved documents.\n")

    userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
    if userresponse == "q":
        print("")
        print(copyright_notice)
        print("")
        exit()
    else:
        print("Let's continue.\n")

    while conv_continue == "y":
        gc.collect()
        print("\nIt is helpful at this point to clear the LLM context window, otherwise it gets confused.")
        user_clear = input("Would you like to clear the context window? ")
        if user_clear == "y":
            print("Wait a moment, and when you see the >>> prompt, type: " + clear_context + "\n When the >>> appears again, type: " + end_subprocess + "\n Typing these two entries will clear the context window of the LLM")
            os.system("cd")
            os.system("ollama run " + inference_model)


        conv_context = "response"
        prompt = "prompt"
        query_documents = folder_docs
        view_docs = input("Would you like to view all the documents matching your general query? Type: y or n: ")
        if view_docs == "y":
            print("\nHere is the full text of documents matching your query: \n--")
            print(query_documents)
            print("--\n")
            print("See the full set of retrieved documents above.")
            view_desired_doc = "n"
            view_desired_doc = input("\nWould you like to view any of the referenced documents? y/n: ")
            while view_desired_doc != "n":
                get_desired_doc(file_path)
                view_desired_doc = input("Would you like to view another of the referenced documents? y/n: ")
        else:
            print("Great. Now enter a new question below")
        prompt = topic_query(desired_collection)
        conv_context = response_generation(query_documents, prompt, inference_model)
        print("\nHere is the AI-Assistant's summary of relevant material from the documents: \n--")
        print(conv_context)
        print("--\n")
        view_desired_doc = "n"
        view_desired_doc = input("\nWould you like to view any of the referenced documents? y/n: ")
        while view_desired_doc != "n":
            get_desired_doc(file_path)
            view_desired_doc = input("Would you like to view another of the referenced documents? y/n: ")

        view_folder = input("Would you like to retrieve all documents in this folder? Type y or n: ")
        folder_contents = []
        if view_folder == "y":
            desired_folder = str(input("Copy the folder name here: \n (i.e. paste the preliminary part of the document name, prior to the final -IMG_ suffix)"))
            folder_contents.append(desired_folder)
            folder_docs = get_folder(folder_contents)
            folder_length = len(folder_docs) / 2
            print("\nHere is the full contents of the folder " + desired_folder + "\n--")
            print(folder_docs)
            print("--\n")
            print("See contents of folder above. There are " + str(folder_length) + " pages in this folder.\n")
            conv_continue = input("\nWould you like to ask questions about the contents of this folder? y/n: ")
            if folder_length > ranked_results / context_limiter:
                folder_docs, uniquephotos = get_ranked_documents(folder_docs, general_prompt, query_chunk_length, ranked_results, file_path, metadata_key)
            else:
                folder_docs = folder_docs
        else:
            folder_docs = query_documents
            print("We'll continue asking questions of the originally retrieved documents.")

        conv_continue = input("\nWould you like to ask questions about this material? y/n: ")

    print("\nWe'll close this topic, but you can inquire about a different topic.")
    gc.collect()
    userresponse = input("\nWhat do you want to do? \n type q to quit or c to continue: ")

print("\nThank you. I hope you found what you were looking for!\n")
print(copyright_notice)
print("")
gc.collect()
exit()

