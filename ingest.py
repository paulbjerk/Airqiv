import ollama
import chromadb
import csv
import gc
import chromadb.utils.embedding_functions as embedding_functions
import os
import statistics
from statistics import mode
import numpy as np
import spacy
import re

# see documentation of spaCy Python library: https://spacy.io/usage/spacy-101
#See this post on how to use spaCy for chunking:
#https://towardsdatascience.com/how-to-chunk-text-data-a-comparative-analysis-3858c4a0997a
#see code sample from above post at bottom, under exit

print("\nArqiv AI-Assistant Document Explorer")
print("This source code is licensed under the BSD2-style license found at https://opensource.org/license/bsd-2-clause .\n")
print("The app leverages open-sourced LLMs using the Ollama app. For more information see https://ollama.com ")
print("The app leverages a vector database using ChromaDB. For more information see https://docs.trychroma.com ")
print("The app leverages the spaCy python library, for more information see https://spacy.io ")
print("I am grateful for the excellent chunking algorithm by Solano Todeschini, published in Towards Data Science Jul 20, 2023.\n(See https://towardsdatascience.com/how-to-chunk-text-data-a-comparative-analysis-3858c4a0997a ) ")
print("\n The documents returned and summarized by this Document Explorer are copyright of the authors and archival custodian.\n")

print("Copyright (c) <2024>, <Paul Bjerk>")
print("All rights reserved.")
print("This source code is licensed under the BSD2-style license found at https://opensource.org/license/bsd-2-clause .\n")
print("\n           - - The Airqiv Document Explorer  - -       ")
print(" - - Artificially Intelligent Retrieval Query Interpretive Visualizer - -")
print("                     - - airqiv.com  - -       ")
print("                        - - :-) - -         \n")

# HNSW space affects the discovery of documents in chromadb
hnsw_space = "ip"
#embed_model = "mxbai-embed-large:latest"
embed_model = "snowflake-arctic-embed:latest"
#embed_model = "rjmalagon/gte-qwen2-1.5b-instruct-embed-f16:latest"
nlp = spacy.load('en_core_web_sm')
embed_model_dimensions = "1024"
#embed_model_dimensions = "1536"
embed_model_layers = "24"
#embed_model = "snowflake-arctic-embed:335m"
#To improve performance see scripts for snowflake model at https://huggingface.co/Snowflake/snowflake-arctic-embed-l
phototext = ""
currentingest = "foldertitle"
chunk = ""
existing_subfolders = []
configured_collections =[]
documents = []
metadatas = []
ids = []
clean_list =[]
doc_chunks = []
chunk_length = 50
archive_collection = ""
topic_collection = ""
sub_collecction = ""
user_choice = ""
batch_ingest = ""
csv_suffix = ".csv"



#This establishes an Ollama embedding model for Chromadb processes
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=embed_model,
)

#this initiates the Chromadb persistent client
client = chromadb.PersistentClient(path="chromadb/phototextvectors")

# add new documents adds raw documents to a collection and returns a new CSV
def add_new_documents(file_path, collection, archive_collection, topic_collection, sub_collection):
    fieldnames = ["FOLDERNAME", "LANGUAGE", "PHOTONAME", "UNIQUEPHOTO", "PHOTOTEXT", "NAMESMENTIONED","COUNTRIESMENTIONED","URL","COPYRIGHT","ARCHIVE","TOPIC","SUBCOLLECTION",]
    with open(file_path, mode="r") as old_file, open(str("all-"+collection+"-documents.csv"), mode="a") as new_file:
        current = csv.DictReader(old_file, fieldnames=fieldnames)
        if archive_collection == "nara":
            archive_name = "The United States National Archives"
            archive_url = "archives.gov"
        elif archive_collection == "jfk":
            archive_name = "The John F. Kennedy Presidential Library"
            archive_url = "jfklibrary.org"
        elif archive_collection == "lbj":
            archive_name = "The Lyndon Baines Johnson Presidential Library"
            archive_url = "lbjlibrary.org"
        elif archive_collection == "rmn":
            archive_name = "The Richard M. Nixon Presidential Library"
            archive_url = "nixonlibrary.gov"
        elif archive_collection == "grf":
            archive_name = "The Gerald R. Ford Presidential Library"
            archive_url = "fordlibrarymuseum.gov"
        elif archive_collection == "jec":
            archive_name = "The Jimmy Carter Presidential Library"
            archive_url = "jimmycarterlibrary.gov"
        elif archive_collection == "rwb":
            archive_name = "The Ronald Reagan Presidential Library"
            archive_url = "reaganlibrary.gov"
        elif archive_collection == "ghwb":
            archive_name = "The George H.W. Bush Library"
            archive_url = "bush41.org"
        elif archive_collection == "imf":
            archive_name = "The International Monetary Fund (IMF)"
            archive_url = "imf.org"
        elif archive_collection == "pro":
            archive_name = "The United Kingdom National Archives, formerly the Public Record Office (PRO)"
            archive_url = "nationalarchives.gov.uk"
        elif archive_collection == "tna":
            archive_name = "The Tanzania National Archives"
            archive_url = "nyaraka.go.tz"
        elif archive_collection == "kna":
            archive_name = "The Kenya National Archives"
            archive_url = "archives.go.ke"
        elif archive_collection == "ttuva":
            archive_name = "The Vietnam Archive at Texas Tech University"
            archive_url = "vva.vietnam.ttu.edu"
        else:
            archive_name = archive_collection
            archive_url = str("For more information search for the archives of "+archive_collection)


        copyright = str("Copyright of "+archive_name+". Fair use criteria of Section 107 of the Copyright Act of 1976 must be followed. The following materials can be used for educational and other noncommercial purposes without the written permission of "+archive_name+". These materials are not to be used for resale or commercial purposes without written authorization from "+archive_name+". This text extraction and summarization process and software is designed and copyrighted by Paul Bjerk. All materials cited must be attributed to the "+archive_name+" at "+archive_url+" and The Airqiv Document Explorer at airqiv.com ")

        #added_values = [archive_collection, topic_collection, sub_collection]
        #this next step skips adding the header, presuming the header already exists
        #but we need an if statement to add a header if it does not exist, i.e. if the csv was not pre-created in the setup 
        next(current, None)
        all_files = csv.DictWriter(new_file, fieldnames=fieldnames)
        for row in current:
            phototext = row["PHOTOTEXT"]
            text = re.sub('\\s[3][01]\\s|\\s[.][3][01]\\s|[.]\\s[12][0-9]\\s|\\s[12][0-9]\\s|\\s[1-9]\\s|[.]\\s[1-9]\\s', ' ', phototext)
            row["PHOTOTEXT"] = text
            row["COPYRIGHT"] = copyright
            #row["URL"] = archive_url
            row["ARCHIVE"] = archive_collection
            row["TOPIC"] = topic_collection
            row["SUBCOLLECTION"] = sub_collection
            all_files.writerow(row)

# create CSV creates a CSV with standard column headings
def create_csv(collection):
    fieldnames = ["FOLDERNAME", "LANGUAGE", "PHOTONAME", "UNIQUEPHOTO", "PHOTOTEXT", "NAMESMENTIONED","COUNTRIESMENTIONED","URL","COPYRIGHT","ARCHIVE","TOPIC","SUBCOLLECTION",]
    with open(str("all-"+collection+"-documents.csv"), mode="w", newline="") as new_file:
        all_files = csv.DictWriter(new_file, fieldnames=fieldnames)
        all_files.writeheader()

def create_folder(archive_collection):
    os.system("mkdir "+archive_collection)

def create_sub_folder(archive_collection, topic_collection):
    os.system("mkdir "+archive_collection+"/"+topic_collection)

#this is a very simple chunker s is the text to chunk and n is the number of characters per chunk
# this could be improved with overlapping chunks and some recursive techniques for better semantic chunks
#def chunker (phototext, chunk_length):
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
    #text = re.sub('\\s[3][01]\\s|\\s[.][3][01]\\s|[.]\\s[12][0-9]\\s|\\s[12][0-9]\\s|\\s[1-9]\\s|[.]\\s[1-9]\\s', ' ', phototext)
    text=phototext
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
        elif cluster_len > 2000:
            threshold = 0.5
            sents_div, vecs_div = process(cluster_txt)
            reclusters = cluster_text(sents_div, vecs_div, threshold)

            for subcluster in reclusters:
                div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
                div_len = len(div_txt)

                if div_len < chunk_length or div_len > 2000:
                    continue

                clusters_lens.append(div_len)
                final_texts.append(div_txt)

        else:
            clusters_lens.append(cluster_len)
            final_texts.append(cluster_txt)

    avg_cluster_len = int(average_elements(clusters_lens))
    number_of_clusters = int(len(final_texts))
    total_chars = avg_cluster_len * number_of_clusters
    #print(final_texts)
    #print(clusters_lens)
    #print("There are "+str(total_chars)+" characters split into " + str(number_of_clusters) + " clusters in this PHOTOTEXT, averaging "+str(avg_cluster_len)+" characters per cluster.")

    return final_texts

def most_frequent(incoming_string):
    clean_string = incoming_string.replace(";", ",")
    countriesmentioned = list(clean_string.split(","))
    # if countriesmentioned == "":
    if len(countriesmentioned) < 1:
        most_common_element = "no__items__mentioned"
    else:
        most_common_element = (mode(countriesmentioned))
        #most_common_element = most_common_element.lower()
    return most_common_element

#in get documents the CSV is called, and each row of data is chunked and labeled with ids
def get_documents(file_path):
    with (open(file_path, mode="r") as csv_file):
        data = csv.DictReader(csv_file)
        documents = []
        metadatas =[]
        ids = []
        for row in data:
            uniquephoto = row["UNIQUEPHOTO"]
            foldername = row["FOLDERNAME"]
            phototext = row["PHOTOTEXT"]
            text = re.sub('\\s[3][01]\\s|\\s[.][3][01]\\s|[.]\\s[12][0-9]\\s|\\s[12][0-9]\\s|\\s[1-9]\\s|[.]\\s[1-9]\\s', ' ',phototext)
            phototext = text
            namesmentioned = str(row["NAMESMENTIONED"])
            countriesmentioned = str(row["COUNTRIESMENTIONED"])
            #url = str(row["URL"])
            #copyright = str(row["COPYRIGHT"])
            url = ""
            copyright = ""
            country_mentioned = str(most_frequent(countriesmentioned))
            name_mentioned = str(most_frequent(namesmentioned))
            archive = archive_collection
            #print("archive metadata field added:"+archive)
            topic = topic_collection
            #print("topic metadata field added:" + topic)
            subcollection = sub_collection
            #print("subcollection metadata field added:" + subcollection)
            metadata = {"FOLDERNAME" : foldername, "UNIQUEPHOTO": uniquephoto, "NAMESMENTIONED": name_mentioned, "COUNTRIESMENTIONED": country_mentioned, "URL": url, "COPYRIGHT": copyright, "ARCHIVE": archive, "TOPIC": topic, "SUBCOLLECTION": subcollection}
            doc_chunks = []
            for chunk in chunker(phototext, chunk_length):
                doc_chunks.append(chunk)
                documents.append(chunk)
                metadatas.append(metadata)
            for item in doc_chunks:
                id_item = uniquephoto
                id_index = doc_chunks.index(item) + 1
                id_suffix = str(id_index)
                id = id_item + "-part-" + id_suffix
                ids.append(id)
    return(documents, metadatas, ids)


# this can be used to count lines in one of the collection CSV files produced at the end
def count_lines(collection):
    with open(str("all-"+collection+"-documents.csv"), mode="r") as new_file:
        num_docs = sum(1 for line in new_file)
        num_docs_str = str(num_docs)
        print("There are " + num_docs_str + " documents in the all-"+collection+"-documents collection")

def count_lines_currentingest(file_path):
    with (open(file_path, newline="") as csv_file):
        num_docs = sum(1 for row in csv_file)
        num_docs_str = str(num_docs)
        print("There are "+num_docs_str+" documents in this ingest folder.\n")

def ingest_csv (currentingest, archive_collection, topic_collection):
    file_path = archive_collection+"/"+topic_collection+"/"+currentingest+csv_suffix
    print("The file_path is: "+file_path)
    collection = client.create_collection(name=currentingest, metadata={"hnsw:space": hnsw_space})

    print("We will now ingest "+currentingest)
    count_lines_currentingest(file_path)

    documents, metadatas, ids = get_documents(file_path)
    embeddings = ollama_ef(documents)
    #print(embeddings)
    #print(documents)

    #https://docs.trychroma.com/guides
    #This adds the chunked documents to the chromadb database under the title of the selected currentingest CSV file
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    #these next steps create two parallel sub-collections
    collection = client.get_or_create_collection(name=str("all-"+topic_collection + "-documents"), metadata={"hnsw:space": hnsw_space})

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    add_new_documents(file_path, topic_collection, archive_collection, topic_collection, sub_collection)

    collection = client.get_or_create_collection(name=str("all-"+archive_collection+"-documents"), metadata={"hnsw:space": hnsw_space})

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    add_new_documents(file_path, archive_collection, archive_collection, topic_collection, sub_collection)

def enter_ingestfile():
    user_choice = input("\nWould you like to ingest a SINGLE CSV file or a whole FOLDER full of files?\n Type: 1 for SINGLE CSV file\n Type 2 for FOLDER of multiple CSV files\n Type 1 or 2 here: ")
    if user_choice == "1":
        batch_ingest = "file"
        print("Good. We'll ingest a single CSV file from a single folder of scanned documents.")
        currentingest = input("What CSV file do you want to ingest? \n Enter the filename only, without the .csv suffix. It's best to copy-paste to avoid typos: ")

    elif user_choice == "2":
        batch_ingest = "folder"
        currentingest = "currentingest"
        print("Good. We'll ingest a folder filled with multiple CSV files derived from from a multiple folders of scanned documents.")
    else:
        print("Please type python3 ingest again, and type 1 for a folder of multiple CSV files, or type 2 for a single CSV file.")
        #user_choice = input("Would you like to ingest a SINGLE CSV file or a whole FOLDER full of files?\n Type: 1 for SINGLE CSV file\n Type 2 for FOLDER of multiple CSV files\n Type 1 or 2 here: ")
        exit()

    archive_collection = input("What one-word name did you give for the overall collection during setup? \n (e.g. archive name like nara): ")
    topic_collection = input("What one-word topic did you give for your collection during setup? \n (e.g. a country, an individual, or a theme: ")
    sub_collection = input("What one-word name did you give for the sub-collection during setup? \n (e.g. a record group or an oral history collection): ")

    archive_collection = archive_collection.lower()
    topic_collection = topic_collection.lower()
    sub_collection = sub_collection.lower()

    return batch_ingest, archive_collection, topic_collection, sub_collection, currentingest


print("\nThis ingest process is slow if you have a lot of documents. It  turns them into machine-readable vectors.\n You should plan on about 150 kilobytes of hard drive storage per page of ingested documents.\n")
print("The " +embed_model+ " vector embedding model has "+embed_model_dimensions+ " dimensions and "+embed_model_layers+" layers. \n  The chosen hnsw space calculation is Inner Product or ip.\n ")

#batch_ingest, archive_collection, topic_collection, sub_collection, currentingest = enter_ingestfile()

file_entry = "n"
while file_entry != "y":
    batch_ingest, archive_collection, topic_collection, sub_collection, currentingest = enter_ingestfile()
    file_entry = input("\nIs  the above information correct? Type y or n: ")


# this instantiates the chromadb database
#client = chromadb.PersistentClient(path="chromadb/phototextvectors")

# This step looks at the CSV files created in setup in order to use them or create new ones for the ingest
#ollama_models = os.popen("ollama list").read()
configured_folders=os.popen("ls -R").read()
configured_collections = os.popen("ls").read()
if str("./"+archive_collection+":") in configured_folders:
    print("We will begin to create collections in this archive folder.")
    #list_command = str(archive_collection + " ls")
    #existing_subfolders = os.popen(list_command).read()
else:
    print("We will create a folder for this archive.")
    #list_command = str(archive_collection + " ls")
    create_folder(archive_collection)
    #existing_subfolders = os.popen(list_command).read()

if str("./"+archive_collection+"/"+topic_collection+":") in configured_folders:
    print("We will add this CSV to the sub-folder titled " + topic_collection + " in the "+archive_collection+" folder.")
else:
    print("We will create a new sub-folder titled " + topic_collection + " in the " + archive_collection + " folder.")
    create_sub_folder(archive_collection, topic_collection)

if "all-"+archive_collection+"-documents.csv" in configured_collections:
    print("We will add the documents in this CSV file to all-"+archive_collection+"-documents")
else:
    # this creates a starting point for a new collection CSV
    print("We will add the documents in this CSV file to all-" + archive_collection + "-documents")
    create_csv(archive_collection)

if "all-"+topic_collection+"-documents.csv" in configured_collections:
    print("We will add the documents in this CSV file to all-" + topic_collection + "-documents")
else:
    print("We will add the documents in this CSV file to all-" + topic_collection + "-documents")
    create_csv(topic_collection)


if batch_ingest == "file":
    #this step moves the CSV from the user's Pictures folder to the topic ingest subfolder
    ingest_folder = str(archive_collection + "/" + topic_collection)
    os.system("mv " + currentingest+csv_suffix + " " + ingest_folder + "/" + currentingest+csv_suffix)
    if currentingest in [c.name for c in client.list_collections()]:
        print("This CSV file has already been ingested!")
        reingest = input("Do you want to re-ingest this collection? type y/n: ")
        if reingest == "n":
            print("Now type - python3 asst.py - and designate - " + currentingest +" - as the collection you want to query.")
            exit()
        else:
            # we need a loop here to delete the relevant rows of the relevant compiled CSVs and replace them with the re-ingested rows
            client.delete_collection(name=currentingest)
        print("You'll see a message when the ingest is done. But it may take a while.\n Leave the Terminal window open.\n")
    else:
        print("You'll see a message when the ingest is done. But it may take a while.\n Leave the Terminal window open.\n")


    ingest_csv(currentingest, archive_collection, topic_collection)
    gc.collect()
else:
    ingest_folder = str(archive_collection + "/" + topic_collection)
    #we need to move all csv files from Pictures that have the archive collection in the first position of filename to the topic collection sub-folder
    folder_content = os.popen("ls").read()
    #print(folder_content)
    csvs_to_ingest = re.findall(archive_collection+"_"+topic_collection+"_.*"+csv_suffix, folder_content)
    print("We will ingest the following CSV files: ")
    print(csvs_to_ingest)
    print("You'll see a message when the ingest is done. But it may take a while.\n Leave the Terminal window open.\n")
    for i in csvs_to_ingest:
        os.system("mv "+i+" "+ingest_folder+"/"+i)
        currentingest, ext = i.split(".")
        #print("currentingest is: " + currentingest)
        for file in os.listdir(ingest_folder):
            if file == currentingest+csv_suffix:
                #currentingest, ext = file.split(".")
                #print("currentingest is: "+currentingest)
                if currentingest in [c.name for c in client.list_collections()]:
                    #we need a loop here to delete the relevant rows of the relevant compiled CSVs and replace them with the re-ingested rows
                    #client.delete_collection(name=currentingest)
                    #ingest_csv(currentingest, archive_collection, topic_collection)
                    os.system("mv "+ingest_folder+"/"+i+" "+i)
                    print(i+" has already been ingested. Please delete the collection and relevant lines in CSV files.")
                else:
                    #valid_oh = "y"
                    oh_split = currentingest.split("_")
                    oh_last_term = oh_split[-1]
                    print(oh_last_term)
                    match = re.search("[O][H][0-9]{4}", oh_last_term)
                    if match:
                        os.system("mv " + ingest_folder + "/" + i + " ttuva-error-log/" + i)
                        print(i + " has no content and needs to be re-obtained.")
                    else:
                        ingest_csv(currentingest, archive_collection, topic_collection)
    gc.collect()


print("Ingest is done!\n")
print("\nTo add more documents to this sub-collection and overall collection, \ntype - python3 ingest7.py - again and enter a new csv or folder name.")
print( "\nTo explore the newly loaded documents, type - python3 asst.py - \n and designate - " + currentingest + " - \n or - all-"+topic_collection+"-documents, or all-"+archive_collection+"-documents - as the collection you want to query.")


exit()

"""
import numpy as np
import spacy

# Load the Spacy model
nlp = spacy.load('en_core_web_sm')


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


# Initialize the clusters lengths list and final texts list
clusters_lens = []
final_texts = []

# Process the chunk
threshold = 0.3
sents, vecs = process(text)

# Cluster the sentences
clusters = cluster_text(sents, vecs, threshold)

for cluster in clusters:
    cluster_txt = clean_text(' '.join([sents[i].text for i in cluster]))
    cluster_len = len(cluster_txt)

    # Check if the cluster is too short
    if cluster_len < 60:
        continue

    # Check if the cluster is too long
    elif cluster_len > 3000:
        threshold = 0.6
        sents_div, vecs_div = process(cluster_txt)
        reclusters = cluster_text(sents_div, vecs_div, threshold)

        for subcluster in reclusters:
            div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
            div_len = len(div_txt)

            if div_len < 60 or div_len > 3000:
                continue

            clusters_lens.append(div_len)
            final_texts.append(div_txt)

    else:
        clusters_lens.append(cluster_len)
        final_texts.append(cluster_txt)

exit()

"""

def create_embed_template ():
    #os.system("ollama pull phi3:14b-medium-128k-instruct-q5_K_M")
    with open("model-template.txt", "w") as file:
        file.write("""FROM mxbai-embed-large:latest
TEMPLATE "{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>"
PARAMETER stop <|end|>
PARAMETER stop <|user|>
PARAMETER stop <|assistant|>
PARAMETER num_ctx 1024
PARAMETER repeat_last_n 2048""")

"""
if os.path.exists("model-template.txt"):
    os.remove("model-template.txt")
    create_large_model_template()
    inference_model_window = "16k tokens"
    os.system("ollama create phi3-16k -f model-template.txt")
    os.system("ollama show --modelfile phi3-16k")
else:
    create_large_model_template()
    inference_model_window = "16k tokens"
    os.system("ollama create phi3-16k -f model-template.txt")
    os.system("ollama show --modelfile phi3-16k")

os.remove("model-template.txt")

"""
