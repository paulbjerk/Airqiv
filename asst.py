import ollama
import chromadb
import csv
import gc
import chromadb.utils.embedding_functions as embedding_functions
import os
from operator import itemgetter


print("\nCopyright (c) <2024>, <Paul Bjerk>")
print("All rights reserved.")
print("\nThis source code is licensed under the BSD2-style license found at https://opensource.org/license/bsd-2-clause .\n")

#general variables
#embed_model = "snowflake-arctic-embed:335m"
embed_model = "mxbai-embed-large:latest"
embed_model_dimensions = "1024"
embed_model_layers = "24"
inference_model = "phi3:3.8b-mini-128k-instruct-q5_K_M"
inference_model_short, inference_model_detail = inference_model.split(":")
phi3_model = "2k"
inference_model_window = "2k tokens"
conv_context = "response"
prompt = "prompt"
response = "response"
general_prompt = ""
redo_general_prompt = "n"
uniquephotos = []
query_documents = []
folder_contents = []
folder_docs =[]
clear_context = "/clear"
end_subprocess = "/bye"
metadata_key = ["UNIQUEPHOTO"]
query_chunk_length = 300
currentingest = "foldertitle"
hnsw_space = "ip"
prompt = f"You are a history professor. Answer the following question by designating one or more documents that contain relevant information. The documents are identified by a UNIQUEPHOTO: name."
user_question_1 = "What are the main themes in these documents?"
user_term_1 = ""
sentencesneeded = "3"
general_prompt = "What is the main theme in these documents?"

# a higher context limiter number makes it more likely that the retrieved documents will be chunked and ranked rather than fed in their entirety into the LLM context.
# the context limiter is a divisor to test the number of retrieved documents against the maximium context length of the LLM
context_limiter = 3


# This step looks at the phi3 model installed in setup and sets a few parameters in the asst app to best use the user's system capabilities (RAM)
ollama_models = os.popen("ollama list").read()

if "phi3-2k" in ollama_models:
    inference_model = "phi3-2k:latest"
    inference_model_window = "2k tokens"
    ranked_results = 20
elif "phi3-4k" in ollama_models:
    inference_model = "phi3-4k:latest"
    inference_model_window = "4k tokens"
    ranked_results = 40
elif "phi3-8k" in ollama_models:
    inference_model = "phi3-8k:latest"
    inference_model_window = "8k tokens"
    ranked_results = 80
elif "phi3-12k" in ollama_models:
    inference_model = "phi3-12k:latest"
    inference_model_window = "12k tokens"
    ranked_results = 120
elif "llama3-16k" in ollama_models:
    inference_model = "llama3-16k:latest"
    inference_model_window = "16k tokens"
    ranked_results = 160
else:
    inference_model = "phi3:3.8b-mini-128k-instruct-q5_K_M"
    inference_model_window = "2k tokens"
    ranked_results = 20

inference_model_short, inference_model_latest = inference_model.split(":")

print("\nThe " +embed_model+ " vector embedding model has "+embed_model_dimensions+ " dimensions and "+embed_model_layers+" layers. \n  The chosen hnsw space calculation is Inner Product or ip.\n ")
print ("The "+inference_model_short+" language model has a context window length of "+inference_model_window+".\n")
#os.system("ollama show --modelfile "+inference_model)
#print("The modelfile information above shows the parameters of the LLM model.\n")

#This establishes an Ollama embedding model for Chromadb processes
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=embed_model,
)

#this initiates the Chromadb persistent client
client = chromadb.PersistentClient(path="chromadb/phototextvectors")


#these prompt the user to designate a CSV and loads it for queries
print("Query the assistant to explore your documents!")
currentingest = input("What collection do you want to explore? \n Enter the filename only, without the .csv suffix. (e.g. all-nara-documents)\n It's best to copy-paste to avoid typos: ")
file_path = currentingest+".csv"
collection = client.get_collection(name=currentingest, embedding_function=ollama_ef)


#functions used

# first user prompt is to ask question of the initially retrieved documents
def first_query():
    user_question_1 = input("What do you want to know about? ")
    sentencesneeded = input("How many sentences do you want in the answer? ")
    prompt = f"You are a history professor. Each document is a JSON with two associated key terms UNIQUEPHOTO and PHOTOTEXT. PHOTOTEXT is the text of the document identified by the UNIQUEPHOTO value. Please state the UNIQUEPHOTO value for every statement. These are documents based on diplomatic reporting. Some documents have header material that looks like gibberish at the beginning of the document. Ignore this kind of header material.\n Answer the following question by designating one or more UNIQUEPHOTO documents that contain relevant information. Question: " + user_question_1 + "? Please respond with " + sentencesneeded + " sentences."
    return prompt

# topic query is used to follow up with the retrieved documents or a user-selected folder of relevant documents
def topic_query():
    user_question_1 = input("What else do you want to ask about this topic? ")
    sentencesneeded = input("How many sentences do you want in the answer (1-9)? ")
    prompt = f"Please revisit the full set of relevant documents. Each document is a JSON with two associated key terms UNIQUEPHOTO and PHOTOTEXT. PHOTOTEXT is the text of the document identified by the UNIQUEPHOTO value. Please state the UNIQUEPHOTO value for every statement. These are documents based on diplomatic reporting. Some documents have header material that can be ignored.\n Using the context of the full set of given documents. Answer this new question: " + user_question_1 + "? Please respond with " + sentencesneeded + " sentences."
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
def retrieve_documents(query_embeddings, user_term_1):
    #first step retrieves relevant chunks via embeddings that match the query embedding
    all_metadatas = []
    retrieved_chunks = collection.query(query_embeddings=query_embeddings, include=["metadatas"], n_results=10)
    metadata_in_list = retrieved_chunks["metadatas"]
    chunks_metadata_list = metadata_in_list[0]
    #print("chunks_metadata_list is: ")
    #print(chunks_metadata_list)
    #chunks_references = list_metadata(chunks_metadata_list, metadata_key)
    #print("chunks_references list is: ")
    #print(chunks_references)
    for item in chunks_metadata_list:
        all_metadatas.append(item)
    #print("all_metadatas after adding chunks is this: ")
    #print(all_metadatas)

    #second step uses the user_term as a search term to find more matching chunks
    retrieved_documents = collection.get(ids=[], where_document={"$contains":user_term_1})
    metadata_in_list = retrieved_documents["metadatas"]
    retrieved_docs_metadata_list = metadata_in_list
    for item in retrieved_docs_metadata_list:
        all_metadatas.append(item)
    uniquephotos = list_metadata(all_metadatas, metadata_key)
    #print(uniquephotos)
    return uniquephotos



#get ranked documents allows users to narrow down a retrived set of documents, to better fit context window low RAM situations
def get_ranked_documents(query_documents, general_prompt, query_chunk_length, ranked_results):
    client = chromadb.PersistentClient(path="chromadb/phototextvectors")
    if "temp_collection" in [c.name for c in client.list_collections()]:
        client.delete_collection(name="temp_collection")
        retrieved_documents = []
        retrieved_metadatas = []
        retrieved_ids = []
    else:
        retrieved_documents = []
        retrieved_metadatas = []
        retrieved_ids = []
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
    ranked_chunks = collection.query(query_embeddings=query_embeddings, n_results=ranked_results)
    metadata_in_list = ranked_chunks["metadatas"]
    chunks_metadata_list = metadata_in_list[0]
    chunks_in_list = ranked_chunks["documents"]
    chunks_chunks_list = chunks_in_list[0]
    ranked_docs = []
    #print(chunks_metadata_list)
    #print(chunks_chunks_list)
    for i in chunks_metadata_list:
        chunk_index = chunks_metadata_list.index(i)
        chunk_text = chunks_chunks_list[chunk_index]
        chunk_image_ref = i["UNIQUEPHOTO"]
        ranked_doc = {"UNIQUEPHOTO": chunk_image_ref, "PHOTOTEXT": chunk_text}
        ranked_docs.append(ranked_doc)
    return ranked_docs

#using the return from the retrieve documents, get documents returns a set of full-text documents, not just chunks
def get_documents(uniquephotos):
  with (open(file_path, newline="") as csv_file):
    data = csv.DictReader(csv_file)
    query_documents = []
    for row in data:
        uniquephoto = row["UNIQUEPHOTO"]
        phototext = row["PHOTOTEXT"]
        for item in uniquephotos:
            if row["UNIQUEPHOTO"] == item:
                query_document = {"UNIQUEPHOTO":uniquephoto, "PHOTOTEXT":phototext}
                query_documents.append(query_document)
    return query_documents

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
def response_generation(data,prompt):
    #  generate a response combining the prompt and data we retrieved in step 2
    output = ollama.generate(
        model=inference_model,
        prompt=f"<|user|>\nUsing this data: {data}. Respond to this prompt: {prompt}<|end|\n<|assistant|>"
    )
    conv_context = output["response"]
    return conv_context


#This simple chunker just splits up s text into chunks of n length
# the chunker could be improved with overlapping chunks and some recursive techniques for smarter chunking
def chunker (s, n):
    """Produce `n`-character chunks from `s`."""
    for start in range(0, len(s), n):
        yield s[start:start+n]


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


# Here is where the main program starts
userresponse = "c"
conv_continue = "y"

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
    conv_context = "response"
    general_prompt = input("What is the general topic you want to know about? ")
    user_term_1 = input("Please provide ONE specific term (name, organization event) relevant to your question.\n If you don't want to specify a search term, type - NONE. \n Enter a one-word search term here: ")
    # these serve as error handling, so that if someone doesn't have a specific search term it doesn't return an error or a huge number of irrelevant documents

    if user_term_1 =="":
        user_term_1 = "asdfkjaosidfjap"
    elif user_term_1 == "NONE":
        user_term_1 = "asdfkjaosidfjap"
    elif user_term_1 == " ":
        user_term_1 = "asdfkjaosidfjap"
    elif user_term_1 == "none":
        user_term_1 = "asdfkjaosidfjap"
    elif user_term_1 == "None":
        user_term_1 = "asdfkjaosidfjap"
    elif user_term_1 == "n":
        user_term_1 = "asdfkjaosidfjap"
    else:
        user_term_1 = user_term_1



    #embed the query and find matching chunks
    query_embeddings = ollama_ef(general_prompt)
    uniquephotos = retrieve_documents(query_embeddings, user_term_1)
    #print(query_embeddings)
    user_term_1 = ""

    #retrieve full document texts from CSV and string them together into a list of strings that can be entered into LLM context window
    all_query_documents = get_documents(uniquephotos)
    #print(query_documents)
    print(uniquephotos)
    number_retrieved = len(uniquephotos)
    print("The " + str(number_retrieved) + " documents listed above have content matching your query.\n If no documents are listed, please enter a different query term below.\n")

    see_names = input("Would you like to see the names associated with these documents? y or n: ")
    print("\nSee names associated with the retrieved documents with page references (add 1 to get right page number)")
    if see_names == "y":
        names_mentioned = get_namesmentioned(uniquephotos)
        for i in names_mentioned:
            print(i)
    else:
        print("Okay, you can ask for names again later.")

    redo_general_prompt = input("Would you like to enter a different query term? Type: y or n: ")
    if redo_general_prompt == "n":
        view_docs = input("Would you like to view the documents matching your general query? Type: y or n: ")
        if view_docs != "n":
            print(all_query_documents)
            print("See the retrieved documents above. Now enter a more specific question below")
        else:
            print("Great. Now enter a more specific question below")
    else:
        print("Okay. Enter a new general topic below.")

#this inner loop allows the user to request a different set of documents
    while redo_general_prompt == "y":
        general_prompt = input("What is the general topic you want to know about? ")
        user_term_1 = input("Please provide ONE specific term (name, organization event) relevant to your question: ")
        query_embeddings = ollama_ef(general_prompt)
        uniquephotos = retrieve_documents(query_embeddings, user_term_1)
        user_term_1 = ""
        all_query_documents = get_documents(uniquephotos)
        print(uniquephotos)
        number_retrieved = len(uniquephotos)
        print("The " + str(number_retrieved) + " documents listed above have content matching your query.\n If no documents are listed, please enter a different query term.")

        see_names = input("Would you like to see the names associated with these documents? y or n: ")
        if see_names == "y":
            names_mentioned = get_namesmentioned(uniquephotos)
            print("\nSee names and document IDs with page reference (add 1 to get right page number)")
            for i in names_mentioned:
                print(i)
        else:
            print("Okay, you can ask for names again later.")

        view_docs = input("Would you like to view the documents matching your general query? Type: y or n: ")
        if view_docs != "n":
            print(all_query_documents)
            print("See the retrieved documents above.")
            print("You can choose to input a new query term below, by typing y or query this set of documents by typing n below.")
        else:
            print("You can choose to input a new query term below, by typing y or query this set of documents by typing n below.")
        redo_general_prompt = input("Would you like to enter a different query term? Type: y or n: ")

    userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
    if userresponse == "q":
        exit()
    else:
        print("Let's continue.\n")

    gc.collect()

    conv_context = "response"
    prompt = first_query()

    if number_retrieved > ranked_results/context_limiter:
        query_documents = get_ranked_documents(all_query_documents, general_prompt, query_chunk_length, ranked_results)
    else:
        query_documents = all_query_documents


    conv_context = response_generation(query_documents,prompt)
    print(conv_context)
    view_folder = "n"
    view_docs = "n"
    view_docs = input("\nWould you like to view any of the referenced documents? y/n: ")
    while view_docs == "y":
        view_doc =[]
        desired_doc = str(input("To view the original text of one designated document,\n cut and paste the UNIQUEPHOTO name here, or just enter n to continue: "))
    #print(desired_doc)
        view_doc.append(desired_doc)
    #print(view_doc)
        doc_text = get_documents(view_doc)
        print(doc_text)
        view_docs = input("Would you like to view another of the referenced documents? y/n: ")

    view_folder = input("Would you like to retrieve all documents in this folder? Type y or n: ")
    folder_contents = []
    if view_folder == "y":
        desired_folder = str(input("Copy the folder name here: \n (i.e. paste the preliminary part of the document name, prior to the final -IMG_ suffix)"))
        folder_contents.append(desired_folder)
        folder_docs = get_folder(folder_contents)
        folder_length = len(folder_docs)
        print(folder_docs)
        print("See contents of folder above. There are " + str(folder_length) + " pages in this folder.\n")
        conv_continue = input("\nWould you like to ask questions about the contents of this folder? y/n: ")
        if folder_length > ranked_results / context_limiter:
            folder_docs = get_ranked_documents(folder_docs, general_prompt, query_chunk_length, ranked_results)
        else:
            folder_docs = folder_docs
    else:
        folder_docs = query_documents
        print("We'll continue asking questions of the originally retrieved documents.")

    #print(conv_continue)

    userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
    if userresponse == "q":
        exit()
    else:
        print("Let's continue.")

    while conv_continue == "y":
        gc.collect()
        print("\nIt is helpful at this point to clear the LLM context window, otherwise it gets confused.")
        user_clear = input("Would you like to clear the context window? ")
        if user_clear == "y":
            print(
                "Wait a moment, and when you see the >>> prompt, type: " + clear_context + "\n When the >>> appears again, type: " + end_subprocess + "\n Typing these two entries will clear the context window of the LLM")
            os.system("cd")
            os.system("ollama run " + inference_model)


        conv_context = "response"
        prompt = "prompt"
        query_documents = folder_docs
        prompt = topic_query()
        conv_context = response_generation(query_documents, prompt)
        print(conv_context)
        view_docs = input("Would you like to view any of the referenced documents? y/n: ")
        while view_docs == "y":
            view_doc = []
            desired_doc = str(input(
                "To view the original text of one designated document,\n cut and paste the UNIQUEPHOTO name here, or just enter n to continue: "))
            # print(desired_doc)
            view_doc.append(desired_doc)
            # print(view_doc)
            doc_text = get_documents(view_doc)
            print(doc_text)
            view_docs = input("Would you like to view another of the referenced documents? y/n: ")

        view_folder = input("Would you like to retrieve all documents in this folder? Type y or n: ")
        folder_contents = []
        if view_folder == "y":
            desired_folder = str(input(
                "Copy the folder name here: \n (i.e. paste the preliminary part of the document name, prior to the final -IMG_ suffix)"))
            folder_contents.append(desired_folder)
            folder_docs = get_folder(folder_contents)
            folder_length = len(folder_docs) / 2
            print(folder_docs)
            print("See contents of folder above. There are " + str(folder_length) + " pages in this folder.\n")
            conv_continue = input("\nWould you like to ask questions about the contents of this folder? y/n: ")
            if folder_length > ranked_results / context_limiter:
                folder_docs = get_ranked_documents(folder_docs, general_prompt, query_chunk_length, ranked_results)
            else:
                folder_docs = folder_docs
        else:
            folder_docs = query_documents
            print("We'll continue asking questions of the originally retrieved documents.")

        conv_continue = input("\nWould you like to ask questions about this material? y/n: ")

    print("We'll close this topic, but you can inquire about a different topic.")
    gc.collect()
    userresponse = input("What do you want to do? \n type q to quit or c to continue: ")

print("Thank you. I hope you found what you were looking for!")
gc.collect()
exit()
