import ollama
import chromadb
import csv
import gc
import chromadb.utils.embedding_functions as embedding_functions
import os
import subprocess


#"note different dimensions of this small model: " embed_model = "nomic-embed-text"
#embed_model = "snowflake-arctic-embed:335m"
embed_model = "mxbai-embed-large"
embed_model_dimensions = "1024"
embed_model_layers = "24"
inference_model = "phi3:3.8b-mini-128k-instruct-q5_K_M"
inference_model_short = "phi3"
inference_model_window = "128k tokens"
#inference_model = "llama3:instruct"
conv_context = "response"
prompt = "prompt"
response = "response"
general_prompt = ""
uniquephotos = []
query_documents = []
folder_contents = []
folder_docs =[]
clear_context = "/clear"
end_subprocess = "/bye"


#embed_model = "nomic-embed-text"
#embed_model = "snowflake-arctic-embed:335m"
#To improve performance see scripts for snowflake model at https://huggingface.co/Snowflake/snowflake-arctic-embed-l

print("The " +embed_model+ " vector embedding model has "+embed_model_dimensions+ " dimensions and "+embed_model_layers+" layers. \n  The chosen hnsw space calculation is Inner Product or ip.\n ")
print ("The "+inference_model_short+" has a context window length of "+inference_model_window+".\n")
#client.delete_collection(name="current_query")
#collection = client.create_collection(name="current_query", metadata=hnsw_space)

prompt = f"You are a history professor. Answer the following question by designating one or more documents that contain relevant information. The documents are identified by a UNIQUEPHOTO: name."
prompt = f"You are a history professor. Each document is a JSON with two associated key terms UNIQUEPHOTO and PHOTOTEXT. PHOTOTEXT is the text of the document identified by the UNIQUEPHOTO value. Please state the UNIQUEPHOTO value for every statement. These are documents about Africa based on diplomatic reporting. Some documents have header material that looks like gibberish at the beginning of the document. Ignore this kind of header material.\n Answer the following question by designating one or more UNIQUEPHOTO documents that contain relevant information. Question: What is the main theme in these documents? Please respond with 3 sentences."
user_question_1 = "What are the main themes in these documents?"
#user_question_2 = "Can you be more specific? (if not leave blank): "
user_term_1 = ""
#user_term_2 = "other important themes"
sentencesneeded = "3"
general_prompt = "What is the main theme in these documents?"

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=embed_model,
)

client = chromadb.PersistentClient(path="chromadb/phototextvectors")

#these prompt the user to designate a CSV to load
print("Query the assistant to explore your documents!")
currentingest = input("What collection do you want to explore? \n Enter the filename only, without the .csv suffix. \n It's best to copy-paste to avoid typos: ")
#currentingest = "NARA-RG59-67-69-Box2513-fx"
collection = client.get_collection(name=currentingest, embedding_function=ollama_ef)
#currentingest_count = str(collection.count())
#collection_preview = collection.peek()
#print("The collection " + currentingest + "contains " + currentingest_count + " UNIQUEPHOTO records.\n")
#print(collection_preview)
#print("Does the above preview of the first ten records look like the right collection?\n")


#hnsw_space = input("Use the same HNSW Space for your query as the document collection ingest.\n" +
                   #"You can choose one of the following: \n" +
                   #"Enter: l2 (for Squared L2) \n" +
                   #"Enter: ip (for Inner Product) \n" +
                   #"Enter: cosine (for Cosine Similarity\n" +
                   #"Enter your choice here: " )
hnsw_space = "ip"
file_path = currentingest+".csv"




#https://docs.trychroma.com/guides
#collection.peek() # returns a list of the first 10 items in the collection
#collection.count() # returns the number of items in the collection
#collection.modify(name="new_name") # Rename the collection





#functions used

# first user prompt

def first_query():
    user_question_1 = input("What do you want to know about? ")
    sentencesneeded = input("How many sentences do you want in the answer? ")
    prompt = f"You are a history professor. Each document is a JSON with two associated key terms UNIQUEPHOTO and PHOTOTEXT. PHOTOTEXT is the text of the document identified by the UNIQUEPHOTO value. Please state the UNIQUEPHOTO value for every statement. These are documents about Africa based on diplomatic reporting. Some documents have header material that looks like gibberish at the beginning of the document. Ignore this kind of header material.\n Answer the following question by designating one or more UNIQUEPHOTO documents that contain relevant information. Question: " + user_question_1 + "? Specifically, relating to " + user_term_1 + "? Please respond with " + sentencesneeded + " sentences."
    return prompt
def topic_query():
    user_question_1 = input("What else do you want to ask about this topic? ")
    sentencesneeded = input("How many sentences do you want in the answer (1-9)? ")
    prompt = f"Please revisit the full set of relevant documents. Each document is a JSON with two associated key terms UNIQUEPHOTO and PHOTOTEXT. PHOTOTEXT is the text of the document identified by the UNIQUEPHOTO value. Please state the UNIQUEPHOTO value for every statement. These are documents about Africa based on diplomatic reporting. Some documents have header material that can be ignored.\n Using the context of the full set of given documents. Answer this new question: " + user_question_1 + "? Please respond with " + sentencesneeded + " sentences."
    return prompt

# Ollama's basic query-documents function using selected LLM
def response_generation(data,prompt):
    #  generate a response combining the prompt and data we retrieved in step 2
    output = ollama.generate(
        model=inference_model,
        prompt=f"<|user|>\nUsing this data: {data}. Respond to this prompt: {prompt}<|end|\n<|assistant|>"
    )
    #phi3 chat template: <|user|>\nQuestion<|end|>\n<|assistant|>

    conv_context = output["response"]
    return conv_context

#query = first_query()

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

def chunker (s, n):
    """Produce `n`-character chunks from `s`."""
    for start in range(0, len(s), n):
        yield s[start:start+n]


#def collection_query(query_embeddings):
    #collection.query(
    #query_embeddings=query_embeddings,
    #include=["documents"],
    #n_results=5,
    #where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":user_term_1}
    #)
#def response_generation():

#def query_embedding(prompt):
    #collection.query(
        #query_texts=[prompt],
        #n_results=5,
        #where={"metadata_field": "is_equal_to_this"},
        #where_document={"$contains":user_term_1}
    #)

def query_chunker(general_prompt, chunk_length):
    #chunk_length = 300
    #query = query
    documents = []
    embeddings = []
    metadatas = []
    ids = []
    metadata = {"general_prompt": general_prompt}
    query = general_prompt
    doc_chunks = []
    for chunk in chunker(query, chunk_length):
        doc_chunks.append(chunk)
        documents.append(chunk)
        metadatas.append(metadata)
    for item in doc_chunks:
        id_item = "general prompt"
        id_index = doc_chunks.index(item) + 1
        id_suffix = str(id_index)
        id = id_item + "-part-" + id_suffix
        ids.append(id)
    embeddings = ollama_ef(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    query_embeddings = collection.get(include=["embeddings"])
    return query_embeddings

def get_documents(uniquephotos):
    #https://www.squash.io/processing-csv-files-in-python/
    #https://teamtreehouse.com/community/how-to-filter-csv-rows-by-keywords-from-another-csv-file
  with (open(file_path, newline="") as csv_file):
    data = csv.DictReader(csv_file)
    query_documents = []
    for row in data:
        uniquephoto = row["UNIQUEPHOTO"]
        #metadata = {"UNIQUEPHOTO" : uniquephoto}
        phototext = row["PHOTOTEXT"]
        for item in uniquephotos:
            if row["UNIQUEPHOTO"] == item:
                query_document = {"UNIQUEPHOTO":uniquephoto, "PHOTOTEXT":phototext}
                query_documents.append(query_document)
    return query_documents

def get_folder(folder_contents):
    with (open(file_path, newline="") as csv_file):
        data = csv.DictReader(csv_file)
        query_documents = []
        for row in data:
            uniquephoto = row["UNIQUEPHOTO"]
            #foldername = row("FOLDERNAME")
            # metadata = {"UNIQUEPHOTO" : uniquephoto}
            phototext = row["PHOTOTEXT"]
            for item in folder_contents:
                if row["FOLDERNAME"] == item:
                    query_document = {"UNIQUEPHOTO": uniquephoto, "PHOTOTEXT": phototext}
                    query_documents.append(query_document)
        return query_documents

def retrieve_documents(query_embeddings):
    uniquephotos = []
    retrieved_chunks = collection.query(query_embeddings=query_embeddings, include=["metadatas"], n_results=10)
    metadata_list_in_list = retrieved_chunks["metadatas"]
    metadata_list = metadata_list_in_list[0]
    #print(metadata_list)
    uniquephotos = list_metadata(metadata_list, metadata_key)

    #use the user_term as a search term and find matching chunks
    retrieved_documents = collection.get(ids=[], where_document={"$contains":user_term_1})
    metadata_list = retrieved_documents["metadatas"]
    #print(metadata_list)

    #uniquephotos compiles a list (without duplicates) of all uniquephoto identifiers of documents found in previous steps
    uniquephotos = list_metadata(metadata_list, metadata_key)
    return uniquephotos



userresponse = "c"
conv_continue = "y"

while userresponse != "q":
    gc.collect()
    print("\nIf you've been working for a while, it is helpful at this point to clear the LLM context window, otherwise it gets confused.\n But if you are just starting a new session, you don't need to do this.")
    user_clear = input("Would you like to clear the context window? ")
    if user_clear == "y":
        print("Wait a moment, and when you see the >>> prompt, type: "+clear_context+"\n When the >>> appears again, type: " +end_subprocess+ "\n Typing these two entries will clear the context window of the LLM")
        os.system("cd")
        os.system("ollama run " + inference_model)

    metadata_key = ["UNIQUEPHOTO"]
    #uniquephotos = []
    conv_context = "response"
    general_prompt = input("What is the general topic you want to know about? ")
    user_term_1 = input("Please provide ONE specific term (name, organization event) relevant to your question: ")

    #embed the query and find matching chunks
    query_embeddings = ollama_ef(general_prompt)
    uniquephotos = retrieve_documents(query_embeddings)
    #print(query_embeddings)

    #retrieve full document texts from CSV and string them together into a list of strings that can be entered into LLM context window
    query_documents = get_documents(uniquephotos)
    #print(query_documents)
    print(uniquephotos)
    print("The above document references have content matching your query.\n If no references are listed, please enter a different query term below.")
    redo_general_prompt = input("Would you like to enter a different query term? Type: y or n: ")
    if redo_general_prompt != "y":
        view_docs = input("Would you like to view the documents matching your general query? Type: y or n: ")
        if view_docs != "n":
            print(query_documents)
            print("See the retrieved documents above. Now enter a more specific question below")
        else:
            print("Great. Now enter a more specific question below")
    else:
        print("Okay. Enter a new general topic below.")

    while redo_general_prompt != "n":
        general_prompt = input("What is the general topic you want to know about? ")
        user_term_1 = input("Please provide ONE specific term (name, organization event) relevant to your question: ")
        query_embeddings = ollama_ef(general_prompt)
        uniquephotos = retrieve_documents(query_embeddings)
        query_documents = get_documents(uniquephotos)
        print(uniquephotos)
        view_docs = input("Would you like to view the documents matching your general query? Type: y or n: ")
        if view_docs != "n":
            print(query_documents)
            print("See the retrieved documents above. Now enter a more specific question below")
        else:
            print("Great. Now enter a more specific question below")
        #print(query_documents)
        number_retrieved = len(uniquephotos)
        print("The "+str(number_retrieved)+" documents listed above have content matching your query.\n If no documents are listed, please enter a different query term.")
        redo_general_prompt = input("Would you like to enter a different query term? Type: y or n: ")

    userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
    if userresponse == "q":
        exit()
    else:
        print("Let's continue.")

    gc.collect()

    conv_context = "response"
    prompt = first_query()
    conv_context = response_generation(query_documents,prompt)
    print(conv_context)
    view_doc =[]
    desired_doc = str(input("If you would like to see the original text of the designated document,\n cut and paste the UNIQEPHOTO name here, or just enter n to continue: "))
    #print(desired_doc)
    view_doc.append(desired_doc)
    #print(view_doc)
    doc_text = get_documents(view_doc)
    print(doc_text)
    view_doc = []
    desired_doc = str(input("If you would like to see the text of another document,\n cut and paste the UNIQEPHOTO name here, or just enter n to continue: "))
    #print(desired_doc)
    view_doc.append(desired_doc)
    #print(view_doc)
    doc_text = get_documents(view_doc)
    print(doc_text)
    folder_contents = []
    view_folder = input("Would you like to retrieve all documents in this folder? Type y or n: ")
    if view_folder != "n":
        desired_folder = str(input("Copy the folder name here: \n (i.e. paste the preliminary part of the document name, prior to the final -IMG_ suffix)"))
        folder_contents.append(desired_folder)
        folder_docs = get_folder(folder_contents)
        folder_length = len(folder_docs)/2
        print(folder_docs)
        print("See contents of folder above. There are " + str(folder_length) + " pages in this folder.\n")
        conv_continue = input("\nWould you like to ask questions about the contents of this folder? y/n: ")
    else:
        folder_docs = query_documents
        print("We'll continue asking questions of the originally retrieved documents.")

    #print(conv_continue)

    #userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
    #if userresponse == "q":
        #exit()
    #else:
        #print("Let's continue.")

    while conv_continue != "n":
        gc.collect()
        print("\nIt is helpful at this point to clear the LLM context window, otherwise it gets confused.")
        user_clear = input("Would you like to clear the context window? ")
        if user_clear == "y":
            print(
                "Wait a moment, and when you see the >>> prompt, type: " + clear_context + "\n When the >>> appears again, type: " + end_subprocess + "\n Typing these two entries will clear the context window of the LLM")
            os.system("cd")
            os.system("ollama run " + inference_model)

        print("See the current active documents above.")
        conv_context = "response"
        prompt = "prompt"
        query_documents = folder_docs
        prompt = topic_query()
        conv_context = response_generation(query_documents, prompt)
        print(conv_context)
        view_doc = []
        desired_doc = str(input("If you would like to see the original text of the designated document,\n cut and paste the UNIQEPHOTO name here, or just enter n to continue: "))
        #print(desired_doc)
        view_doc.append(desired_doc)
        #print(view_doc)
        doc_text = get_documents(view_doc)
        print(doc_text)
        view_doc = []
        desired_doc = str(input("If you would like to see the text of another document,\n cut and paste the UNIQEPHOTO name here, or just enter n to continue: "))
        # print(desired_doc)
        view_doc.append(desired_doc)
        # print(view_doc)
        doc_text = get_documents(view_doc)
        print(doc_text)
        view_folder = input("Would you like to retrieve all documents in this folder? Type y or n: ")
        if view_folder != "n":
            desired_folder = input(
                "Copy the folder name here: \n (i.e. paste the preliminary part of the document name, prior to the final -IMG_ suffix)")
            folder_contents.append(desired_folder)
            folder_docs = get_folder(folder_contents)
            folder_length = len(folder_docs)/2
            print(folder_docs)
            print("See contents of folder above. There are " + str(folder_length) + " pages in this folder.\n")
            conv_continue = input("\nWould you like to ask questions about the contents of this folder? y/n: ")
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



def response_generation():
 # generate an embedding for the prompt and retrieve the most relevant doc
    response = ollama.embeddings(
        prompt=prompt,
        model=embed_model
    )
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=3
    )
    data = results["documents"][0][0]

    #  generate a response combining the prompt and data we retrieved in step 2
    output = ollama.generate(
        model=inference_model,
        prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
    )
    #print(data)
    #print(output["response"])
    conv_context = output["response"]
    return conv_context


