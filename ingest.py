import ollama
import chromadb
import csv
import gc
import chromadb.utils.embedding_functions as embedding_functions
import os

print("\nCopyright (c) <2024>, <Paul Bjerk>")
print("All rights reserved.")
print("\nThis source code is licensed under the BSD2-style license found at https://opensource.org/license/bsd-2-clause .\n")


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
            metadata = {"FOLDERNAME" : foldername, "UNIQUEPHOTO" : uniquephoto}
            phototext = row["PHOTOTEXT"]
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
    file_path = "./"+archive_collection+"/"+topic_collection+"/"+currentingest+".csv"
    collection = client.create_collection(name=currentingest, metadata={"hnsw:space": hnsw_space})

    print("We will now ingest "+currentingest)
    count_lines_currentingest(file_path)

    documents, metadatas, ids = get_documents(file_path)
    embeddings = ollama_ef(documents)
    print(embeddings)
    print(documents)

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


print("This ingest process is slow if you have a lot of documents. It  turns them into machine-readable vectors.")
print("The " +embed_model+ " vector embedding model has "+embed_model_dimensions+ " dimensions and "+embed_model_layers+" layers. \n  The chosen hnsw space calculation is Inner Product or ip.\n ")

user_choice = input("Would you like to ingest a SINGLE CSV file or a whole FOLDER full of files?\n Type: 1 for SINGLE CSV file\n Type 2 for FOLDER of multiple CSV files\n Type 1 or 2 here: ")
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

print("\nIs  the above information correct? If not you can quit and start over.\n")
userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
if userresponse == "q":
    exit()
else:
    print("Let's continue.\n")



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
    #we need to move the CSV from Pictures folder to the topic ingest subfolder
    os.system("mv ~/Pictures/"+currentingest+".csv ~/ai-assistant/"+archive_collection+"/"+topic_collection+"/"+currentingest+".csv")
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
    ingest_folder = str("./" + archive_collection + "/" + topic_collection)
    #we need to move all csv files from pictures that have the archive collection in the first position of filename to the topic collection sub-folder
    for file in os.listdir(ingest_folder):
        if file.endswith(".csv"):
            currentingest, ext = file.split(".")
            print(currentingest)
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
