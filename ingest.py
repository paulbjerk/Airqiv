import ollama
import chromadb
import csv
import gc
import chromadb.utils.embedding_functions as embedding_functions
import os
import re
import statistics
from statistics import mode

print("\n   - - The Airqiv Document Explorer  - -       ")
print("\n          - - www.airqiv.com  - -       ")
print("\nAI-Assistant Document Explorer")
print("Copyright (c) <2024>, <Paul Bjerk>")
print("All rights reserved.")
print("\nThis source code is licensed under the BSD2-style license found at https://opensource.org/license/bsd-2-clause .\n")
print("The app leverages open-sourced LLMs using the Ollama app and a vector database using ChromaDB")


# HNSW space affects the discovery of documents in chromadb
hnsw_space = "ip"
embed_model = "mxbai-embed-large"
embed_model_dimensions = "1024"
embed_model_layers = "24"
#embed_model = "snowflake-arctic-embed:335m"
#To improve performance see scripts for snowflake model at https://huggingface.co/Snowflake/snowflake-arctic-embed-l
phototext = ""
currentingest = "foldertitle"
chunk = ""
documents = []
metadatas = []
ids = []
clean_list =[]
doc_chunks = []
chunk_length = 500
archive_collection = ""
topic_collection = ""
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
def add_new_documents(file_path, collection):
    fieldnames = ["FOLDERNAME", "LANGUAGE", "PHOTONAME", "UNIQUEPHOTO", "PHOTOTEXT", "NAMESMENTIONED","COUNTRIESMENTIONED","INSTRUCTION","CONTEXT","RESPONSE",]
    with open(file_path, mode="r") as old_file, open(str("all-"+collection+"-documents.csv"), mode="a") as new_file:
        current = csv.DictReader(old_file, fieldnames=fieldnames)
        #this next step skips adding the header, presuming the header already exists
        #but we need an if statement to add a header if it does not exist, i.e. if the csv was not pre-created in the setup 
        next(current, None)
        all_files = csv.DictWriter(new_file, fieldnames=fieldnames)
        for row in current:
            all_files.writerow(row)

# create CSV creates a CSV with standard column headings
def create_csv(collection):
    fieldnames = ["FOLDERNAME", "LANGUAGE", "PHOTONAME", "UNIQUEPHOTO", "PHOTOTEXT", "NAMESMENTIONED","COUNTRIESMENTIONED","INSTRUCTION","CONTEXT","RESPONSE",]
    with open(str("all-"+collection+"-documents.csv"), mode="w", newline="") as new_file:
        all_files = csv.DictWriter(new_file, fieldnames=fieldnames)
        all_files.writeheader()

def create_folder(archive_collection):
    os.system("mkdir "+archive_collection)

def create_sub_folder(archive_collection, topic_collection):
    os.system("mkdir "+archive_collection+"/"+topic_collection)

#this is a very simple chunker s is the text to chunk and n is the number of characters per chunk
# this could be improved with overlapping chunks and some recursive techniques for better semantic chunks
def chunker (s, n):
    """Produce `n`-character chunks from `s`."""
    for start in range(0, len(s), n):
        yield s[start:start+n]

def most_frequent(incoming_string):
    clean_string = incoming_string.replace(";", ",")
    countriesmentioned = list(clean_string.split(","))
    # if countriesmentioned == "":
    if len(countriesmentioned) < 1:
        most_common_element = "no__items__mentioned"
    else:
        most_common_element = (mode(countriesmentioned))
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
            namesmentioned = str(row["NAMESMENTIONED"])
            countriesmentioned = str(row["COUNTRIESMENTIONED"])
            country_mentioned = str(most_frequent(countriesmentioned))
            name_mentioned = str(most_frequent(namesmentioned))
            metadata = {"FOLDERNAME" : foldername, "UNIQUEPHOTO": uniquephoto, "NAMESMENTIONED": name_mentioned, "COUNTRIESMENTIONED": country_mentioned}
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
    add_new_documents(file_path, topic_collection)

    collection = client.get_or_create_collection(name=str("all-"+archive_collection+"-documents"), metadata={"hnsw:space": hnsw_space})

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents
    )
    add_new_documents(file_path, archive_collection)

def enter_ingestfile():
    user_choice = input("\nWould you like to ingest a SINGLE CSV file or a whole FOLDER full of files?\n Type: 1 for SINGLE CSV file\n Type 2 for FOLDER of multiple CSV files\n Type 1 or 2 here: ")
    if user_choice == "1":
        batch_ingest = "file"
        print("Good. We'll ingest a single CSV file from a single folder of scanned documents.")
    elif user_choice == "2":
        batch_ingest = "folder"
        print("Good. We'll ingest a folder filled with multiple CSV files derived from from a multiple folders of scanned documents.")
    else:
        print("PLease type 1 for a folder of multiple CSV files, or type 2 for a single CSV file.")
        user_choice = input("Would you like to ingest a SINGLE CSV file or a whole FOLDER full of files?\n Type: 1 for SINGLE CSV file\n Type 2 for FOLDER of multiple CSV files\n Type 1 or 2 here: ")

    archive_collection = input("What one-word name did you give for the overall collection during setup? \n (e.g. archive name like nara): ")
    topic_collection = input("What one-word name did you give for your sub-collection during setup? \n (e.g. a country, an individual, a theme, an archive or sub-section: ")
    return batch_ingest, archive_collection, topic_collection

print("\nThis ingest process is slow if you have a lot of documents. It  turns them into machine-readable vectors.\n You should plan on about 150 kilobytes of hard drive storage per page of ingested documents.\n")
print("The " +embed_model+ " vector embedding model has "+embed_model_dimensions+ " dimensions and "+embed_model_layers+" layers. \n  The chosen hnsw space calculation is Inner Product or ip.\n ")

batch_ingest, archive_collection, topic_collection = enter_ingestfile()

file_entry = input("\nIs  the above information correct? Type y or n: ")
while file_entry != "y":
    batch_ingest, archive_collection, topic_collection = enter_ingestfile()


# this instantiates the chromadb database
#client = chromadb.PersistentClient(path="chromadb/phototextvectors")

# This step looks at the CSV files created in setup in order to use them or create new ones for the ingest
#ollama_models = os.popen("ollama list").read()
configured_collections=os.popen("ls").read()


if "all-"+archive_collection+"-documents.csv" in configured_collections:
    print("We will add the documents in this CSV file to all-"+archive_collection+"-documents")
else:
    # this creates a starting point for a new collection CSV
    print("We will add the documents in this CSV file to all-" + archive_collection + "-documents")
    create_csv(archive_collection)
    create_folder(archive_collection)

if "all-" + topic_collection + "-documents.csv" in configured_collections:
    print("We will add the documents in this CSV file to all-" + topic_collection + "-documents")
else:
    # this creates a starting point for a new collection CSV
    print("We will add the documents in this CSV file to all-" + topic_collection + "-documents")
    create_csv(topic_collection)
    create_sub_folder(archive_collection, topic_collection)


if batch_ingest == "file":
    currentingest = input("What CSV file do you want to ingest? \n Enter the filename only, without the .csv suffix. It's best to copy-paste to avoid typos: ")
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
            print("You'll see a message when the ingest is done. But it may take a while.\n Leave the Terminal window open.")
    else:
        print("You'll see a message when the ingest is done. But it may take a while.\n Leave the Terminal window open.")


    ingest_csv(currentingest, archive_collection, topic_collection)
    print("\nTo add more documents to this sub-collection and overall collection, \ntype - python3 ingest7.py - again and enter a new csv or folder name.")
    print( "\nTo explore the newly loaded documents, type - python3 asst.py - \n and designate - " + currentingest + " - \n or - all-"+topic_collection+"-documents, or all-"+archive_collection+"-documents - as the collection you want to query.")
    gc.collect()
else:
    ingest_folder = str(archive_collection + "/" + topic_collection)
    #we need to move all csv files from Pictures that have the archive collection in the first position of filename to the topic collection sub-folder
    folder_content = os.popen("ls").read()
    #print(folder_content)
    csvs_to_ingest = re.findall(archive_collection+"_"+topic_collection+"_.*"+csv_suffix, folder_content)
    print("We will ingest the following CSV files: ")
    print(csvs_to_ingest)
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
                    client.delete_collection(name=currentingest)
                    ingest_csv(currentingest, archive_collection, topic_collection)
                else:
                    ingest_csv(currentingest, archive_collection, topic_collection)
    gc.collect()


    print("Ingest is done!\n")
    print("\nTo add more documents to this sub-collection and overall collection, \ntype - python3 ingest7.py - again and enter a new csv or folder name.")
    print( "\nTo explore the newly loaded documents, type - python3 asst.py - \n and designate - " + currentingest + " - \n or - all-"+topic_collection+"-documents, or all-"+archive_collection+"-documents - as the collection you want to query.")


exit()
