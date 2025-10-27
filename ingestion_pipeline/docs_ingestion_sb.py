import os
import json
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

# === Config ===
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
INDEX_NAME = os.getenv("AZURE_SEARCH_DOC_INDEX")
API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

credential = AzureKeyCredential(API_KEY)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY")
)

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=INDEX_NAME,
    credential=credential
)

# === Helpers ===
def parse_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text(separator="\n", strip=True)

def process_folder(folder_path):
    print(f"Starting processing folder: {folder_path}")
    batch_docs = []
    batch_size = 100
    total_uploaded = 0

    for root, _, files in os.walk(folder_path):   # recursive
        for file in files:
            if file.endswith(".html"):
                print(f"Processing HTML file: {os.path.join(root, file)}")
                page_id = file.split("_")[-1].replace(".html", "")
                json_file = file.replace(".html", ".json")
                json_path = os.path.join(root, json_file)
                if not os.path.exists(json_path):
                    continue

                with open(json_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                print(f"Loaded metadata for {file}: {metadata.get('title', '')}")

                text = parse_html(os.path.join(root, file))
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=0
                )
                chunks = splitter.split_text(text)
                print(f"Created {len(chunks)} chunks for {file}")

                seen = set()
                unique_chunks = []
                skipped_count = 0
                for chunk in chunks:
                    if chunk in seen:
                        skipped_count += 1
                        continue
                    seen.add(chunk)
                    unique_chunks.append(chunk)
                print(f"Skipped {skipped_count} duplicate chunks for {file}")

                vectors = embeddings.embed_documents(unique_chunks)
                print(f"Generated embeddings for {len(vectors)} chunks of {file}")

                for i, (chunk, vector) in enumerate(zip(unique_chunks, vectors)):
                    batch_docs.append({
                        "id": f"{page_id}_{i}",
                        "content": chunk,
                        "embedding": vector,
                        "title": metadata["title"],
                        "url": metadata["url"],
                        "lastModified": metadata["lastModified"],
                        "parentId": metadata["parentId"]
                    })

                    if len(batch_docs) >= batch_size:
                        print(f"Batch size reached ({len(batch_docs)}). Uploading batch...")
                        result = search_client.upload_documents(batch_docs)
                        print(f"Uploaded {len(result)} docs in current batch to Azure AI Search.")
                        total_uploaded += len(result)
                        batch_docs.clear()

    # Upload any remaining docs in the buffer
    if batch_docs:
        print(f"Uploading remaining {len(batch_docs)} docs in final batch...")
        result = search_client.upload_documents(batch_docs)
        print(f"Uploaded {len(result)} docs in final batch to Azure AI Search.")
        total_uploaded += len(result)
        batch_docs.clear()

    print(f"Finished processing folder. Total documents uploaded: {total_uploaded}")

# === Upload ===
try:
    count = search_client.get_document_count()
    print(f"Connected to Azure AI Search index '{INDEX_NAME}'. Current document count: {count}")
except Exception as e:
    print(f"Failed to connect to Azure AI Search index '{INDEX_NAME}': {e}")
    exit(1)

folder = "path_to_exported_confluence_docs"
process_folder(folder)