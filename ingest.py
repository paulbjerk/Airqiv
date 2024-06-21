#import ollama
import chromadb
import csv
import gc
import chromadb.utils.embedding_functions as embedding_functions

print("\nCopyright (c) <2024>, <Paul Bjerk>")
print("All rights reserved.")
print("\nThis source code is licensed under the BSD2-style license found at https://opensource.org/license/bsd-2-clause .")


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


# this instantiates the chromadb database
client = chromadb.PersistentClient(path="chromadb/phototextvectors")

# this establishes the chosen ollama embedding model
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url="http://localhost:11434/api/embeddings",
    model_name=embed_model,
)

print("This ingest process is slow if you have a lot of documents. It  turns them into machine-readable vectors.")
print("The " +embed_model+ " vector embedding model has "+embed_model_dimensions+ " dimensions and "+embed_model_layers+" layers. \n  The chosen hnsw space calculation is Inner Product or ip.\n ")

#these prompt the user to designate a CSV to process
currentingest = input("What CSV file do you want to ingest? \n Enter the filename only, without the .csv suffix. It's best to copy-paste to avoid typos: ")
archive_collection = input("What one-word name did you give for the overall collection during setup? \n (e.g. archive name or research project: ")
topic_collection = input("What one-word name did you give for your sub-collection during setup? \n (e.g. a country, an individual, a theme, an archive or sub-section: ")
file_path = currentingest+".csv"
print("\nIs  the above information correct? If not you can quit and start over.\n")

userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
if userresponse == "q":
    exit()
else:
    print("Let's continue.\n")

#this step exits program if collection has already been ingested, or prompts a user to re-ingest it
if currentingest in [c.name for c in client.list_collections()]:
  print("This collection has already been ingested!")
  reingest = input("Do you want to re-ingest this collection? type y/n: ")
  if reingest == "n":
    collection = client.get_collection(name=currentingest)
    print(collection.peek())
    right_docs = input("Does the above preview of the first ten records look like the right collection? ")
    reingest = input("Do you want to re-ingest this collection? type y/n: ")
    #add_new_documents()
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


# add new documents adds raw documents to a collection and returns a new CSV
def add_new_documents(collection):
    fieldnames = ["FOLDERNAME", "LANGUAGE", "PHOTONAME", "UNIQUEPHOTO", "PHOTOTEXT", "NAMESMENTIONED","COUNTRIESMENTIONED","INSTRUCTION","CONTEXT","RESPONSE",]
    with open(file_path, mode="r") as old_file, open(str("all-"+collection+"-documents.csv"), mode="a") as new_file:
        current = csv.DictReader(old_file, fieldnames=fieldnames)
        next(current, None)
        all_files = csv.DictWriter(new_file, fieldnames=fieldnames)
        for row in current:
            all_files.writerow(row)

# create CSV creates a CSV with standard column headings
def create_csv(collection):
    fieldnames = ["FOLDERNAME", "LANGUAGE", "PHOTONAME", "UNIQUEPHOTO", "PHOTOTEXT", "NAMESMENTIONED","COUNTRIESMENTIONED","INSTRUCTION","CONTEXT","RESPONSE",]
    with open(str("all-"+collection+"-documents.csv"), mode="w") as new_file:
        all_files = csv.DictWriter(new_file, fieldnames=fieldnames)
        all_files.writerow(fieldnames)

#this is a very simple chunker s is the text to chunk and n is the number of characters per chunk
# this could be improved with overlapping chunks and some recursive techniques for better semantic chunks
def chunker (s, n):
    """Produce `n`-character chunks from `s`."""
    for start in range(0, len(s), n):
        yield s[start:start+n]


#in get documents the CSV is called, and each row of data is chunked and labeled with ids
def get_documents(file_path):
  with (open(file_path, newline="") as csv_file):
    data = csv.DictReader(csv_file)
    documents = []
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

count_lines_currentingest(file_path)

userresponse = input("If you would like to exit the program here, press q: \n or press c to continue.  ")
if userresponse == "q":
    exit()
else:
    print("Let's continue.\n")

documents = get_documents(file_path)
embeddings = ollama_ef(documents)

#https://docs.trychroma.com/guides
#This adds the chunked documents to the chromadb database under the title of the selected currentingest CSV file
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(collection.peek())
print("Ingest is done! See first 10 chunks above.\n")

#these next steps create two parallel sub-collections
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

print("\nTo add more documents to this sub-collection and overall collection, \ntype - python3 ingest7.py - again and enter a new csv name.")
print( "\nTo explore the newly loaded documents, type - python3 asst.py - \n and designate - " + currentingest + " - \n or - all-"+topic_collection+"-documents, or all-"+archive_collection+"-documents - as the collection you want to query.")
gc.collect()

exit()

