import ollama
import chromadb
import csv
import gc
#from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import chromadb.utils.embedding_functions as embedding_functions
from time import process_time

process_start = process_time()

currentingest = "foldertitle"
#these prompt the user to designate a CSV to load

hnsw_space = "ip"
#hnsw_space = input("What HNSW space do you want to use (this affects the discovery of documents.\n" +
                   #"You can choose one of the following: \n" +
                   #"Enter: l2 (for Squared L2) \n" +
                   #"Enter: ip (for Inner Product) \n" +
                   #"Enter: cosine (for Cosine Similarity\n" +
                   #"Enter your choice here: "
#)

embed_model = "mxbai-embed-large"
embed_model_dimensions = "1024"
embed_model_layers = "24"
#embed_model = "nomic-embed-text"
#embed_model = "snowflake-arctic-embed:335m"
#To improve performance see scripts for snowflake model at https://huggingface.co/Snowflake/snowflake-arctic-embed-l


client = chromadb.PersistentClient(path="chromadb/phototextvectors")

ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=embed_model,
)
print("This ingest process is slow if you have a lot of documents. It  turns them into machine-readable vectors.")
print("The " +embed_model+ " vector embedding model has "+embed_model_dimensions+ " dimensions and "+embed_model_layers+" layers. \n  The chosen hnsw space calculation is Inner Product or ip.\n ")

#item_or_folder = input ("Would you like to process a single CSV or a folder full of several CSVs? Type CSV or FOLDER:  ")
#if item_or_folder == CSV:
currentingest = input("What CSV file do you want to ingest? (enter the filename only, without the .csv suffix) It's best to copy-paste to avoid typos ")
#print(currentingest)
topic_collection = input("What one-word name would you give for your sub-collection? \n (e.g. a country, an individual, a theme, an archive or sub-section: ")
archive_collection = input("What is the one-word name of the overall collection? \n (e.g. archive name or research project: ")

phototext = ""
file_path = currentingest+".csv"
chunk = ""
documents = []
metadatas = []
ids = []
clean_list =[]
doc_chunks = []
chunk_length = 500

#this step exits program if collection has already been ingested
# or collects all the appended CSV lines into a single list that for access by the vector database
if currentingest in [c.name for c in client.list_collections()]:
  print("This collection has already been ingested!")
  reingest = input("Do you want to re-ingest this collection? type y/n: ")
  if reingest != "y":
    collection = client.get_collection(name=currentingest)
    print(collection.peek())
    print("Does the above preview of the first ten records look like the right collection?\n")
    #add_new_documents()
    process_stop = round(process_time(), 0)
    ingest_length = len(metadatas)
    ingest_time = round(((process_stop - process_start) / 60), 1)
    print("This ingest of " + str(ingest_length) + " chunks of " + str(chunk_length) + "-character chunks of documents took " + str(ingest_time) + "minutes.\n")
    print("Now type - python3 asst.py - and designate - " + currentingest +" - as the collection you want to query.")
    exit()
  else:
    collection = client.get_collection(name=currentingest)
    client.delete_collection(name=currentingest)
    collection = client.create_collection(name=currentingest, metadata={"hnsw:space": hnsw_space})
    #all_documents = get_documents(documents)
    print("You'll see a message when the ingest is done. But it may take a while.\n Leave the Terminal window open.")
else:
  print("You'll see a message when the ingest is done. But it may take a while.\n Leave the Terminal window open.")
  collection = client.create_collection(name=currentingest, metadata={"hnsw:space": hnsw_space})



def add_new_documents(collection):
    fieldnames = ["FOLDERNAME", "LANGUAGE", "PHOTONAME", "UNIQUEPHOTO", "PHOTOTEXT", "NAMESMENTIONED","COUNTRIESMENTIONED","INSTRUCTION","CONTEXT","RESPONSE"]
    with open(file_path, mode="r") as old_file, open(str("all-"+collection+"-documents.csv"), mode="a") as new_file:
        current = csv.DictReader(old_file, fieldnames=fieldnames)
        next(current, None)
        all_files = csv.DictWriter(new_file, fieldnames=fieldnames)
        for row in current:
            all_files.writerow(row)

def create_csv(collection):
    fieldnames = ["FOLDERNAME", "LANGUAGE", "PHOTONAME", "UNIQUEPHOTO", "PHOTOTEXT", "NAMESMENTIONED","COUNTRIESMENTIONED","INSTRUCTION","CONTEXT","RESPONSE"]
    with open(str("all-"+collection+"-documents.csv"), mode="w") as new_file:
        all_files = csv.DictWriter(new_file, fieldnames=fieldnames)
        all_files.writerow(fieldnames)

#this is a very simple chunker s is the text to chunk and n is the number of characters per chunk
def chunker (s, n):
    """Produce `n`-character chunks from `s`."""
    for start in range(0, len(s), n):
        yield s[start:start+n]


#this is the main process, the CSV is called, and each row of data is chunked and labeled with ids

def get_documents():
  with (open(file_path, newline="") as csv_file):
    data = csv.DictReader(csv_file)
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
    return documents



#This instantiates the chromadb


documents = get_documents()

#print("This is going to take a while, so please be patient. I'll let you know when I'm done.")
#print(documents)
#print(metadatas)
#print(ids)

embeddings = ollama_ef(documents)

#https://docs.trychroma.com/guides
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(collection.peek())
print("Ingest is done! See first 10 chunks above.")
process_stop = process_time()
ingest_length = len(metadatas)
ingest_time_float = (process_stop-process_start)/60
ingest_time = round(ingest_time_float, 1)
#print("This ingest of "+ str(ingest_length) + " chunks of "+ str(chunk_length) + "-character chunks of documents took " + str(ingest_time) + "minutes.")

collection = client.get_or_create_collection(name=str("all-"+topic_collection + "-documents"), metadata={"hnsw:space": hnsw_space})

collection.upsert(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)
add_new_documents(topic_collection)

collection = client.get_or_create_collection(name=str("all-"+archive_collection+"-documents"), metadata={"hnsw:space": hnsw_space})

collection.upsert(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents
)

add_new_documents(archive_collection)


print("To add more documents to this sub-collection and overall collection, type - python3 ingest7.py - again \n and enter a new csv name.")
print( "To explore the newly loaded documents, type - python3 asst.py - \n and designate - " + currentingest + " - \n or - all-"+topic_collection+"-documents, or all-"+archive_collection+"-documents - as the collection you want to query.")
gc.collect()

exit()

