from tree_sitter import Language, Parser
import os
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from azure.search.documents import SearchClient
from azure.identity import ClientSecretCredential
from dotenv import load_dotenv
import hashlib

load_dotenv()

CPP_LANGUAGE = Language("/Users/kumarn/workspace/build/my-languages.so", "cpp")

parser = Parser()
parser.set_language(CPP_LANGUAGE)

AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = "code-index-v4"
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID")
AZURE_SEARCH_CLIENT_ID = os.getenv("AZURE_SEARCH_CLIENT_ID")
AZURE_SEARCH_CLIENT_SECRET = os.getenv("AZURE_SEARCH_CLIENT_SECRET")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
REPO_NAME = os.getenv("REPO_NAME", "XXXX")

# Use your actual OpenGrok base URL here
OPENGROK_BASE_URL = "https://opengrok.cicd.rnd.ORG_NAME.net/xref/PROJECT_NAME/REPO_NAME/"

DRY_RUN = False  # Set True to skip actual uploading and just output to file

credential = ClientSecretCredential(
    tenant_id=AZURE_TENANT_ID,
    client_id=AZURE_SEARCH_CLIENT_ID,
    client_secret=AZURE_SEARCH_CLIENT_SECRET
)

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    model="text-embedding-3-large",
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=credential,
)

def get_node_text(node, code_bytes):
    return code_bytes[node.start_byte:node.end_byte].decode('utf-8')

def get_node_start_line(node):
    return node.start_point[0] + 1

def get_node_end_line(node):
    return node.end_point[0] + 1

def extract_functions_and_macros(root_node, code_bytes):
    results = []

    def add_symbol(symbol_type, node, signature_node=None, body_node=None):
        signature_text = get_node_text(signature_node if signature_node else node, code_bytes)
        body_text = get_node_text(body_node, code_bytes) if body_node else None


        symbol_name = None
        if symbol_type in ['function', 'prototype']:
            # Try to extract function name from declarator or identifier child
            if signature_node:
                # For function_definition, signature_node is function_declarator or similar
                for c in signature_node.children:
                    if c.type == 'identifier':
                        symbol_name = get_node_text(c, code_bytes)
                        break
            else:
                # fallback: search in node children
                for c in node.children:
                    if c.type == 'identifier':
                        symbol_name = get_node_text(c, code_bytes)
                        break
            if not symbol_name:
                # fallback: entire signature first word
                symbol_name = signature_text.split('(')[0].strip().split()[-1]
        elif symbol_type in ['class', 'struct']:
            # class_specifier or struct_specifier: name is identifier child
            for c in node.children:
                if c.type == 'type_identifier' or c.type == 'identifier':
                    symbol_name = get_node_text(c, code_bytes)
                    break
        elif symbol_type == 'macro':
            # preproc_def or preproc_function_def: name is first child after '#define'
            first_child = node.child_by_field_name('name')
            if first_child:
                symbol_name = get_node_text(first_child, code_bytes)
            else:
                # fallback: first token after '#define'
                tokens = signature_text.strip().split()
                if len(tokens) > 1:
                    symbol_name = tokens[1]
        elif symbol_type == 'enum':
            # enum_specifier: name is type_identifier child if exists
            for c in node.children:
                if c.type == 'type_identifier':
                    symbol_name = get_node_text(c, code_bytes)
                    break
            if not symbol_name:
                symbol_name = 'enum'
        elif symbol_type == 'macro_call':
            # call_expression with uppercase callee
            callee = None
            for c in node.children:
                if c.type == 'identifier':
                    callee = get_node_text(c, code_bytes)
                    break
            symbol_name = callee

        if not symbol_name:
            symbol_name = "<unknown>"

        # Get last modified time of file if available (passed externally)
        # We'll fill it later in the caller.

        symbol_info = {
            "symbol": symbol_name,
            "type": symbol_type,
            "signature": signature_text,
        }
        if body_text:
            symbol_info["body"] = body_text
        return symbol_info

    def traverse(node):
        # We look for:
        # function_definition (signature + body, with comments attached to signature)
        # class_specifier and struct_specifier (signature + body separately)
        # preproc_def, preproc_function_def (macros, but we skip adding them to results)
        # enum_specifier and enumerator_list
        # declaration (function prototypes)
        # call_expression with uppercase callee (macro-like calls, skip adding to results)

        if node.type == 'function_definition':
            # function_definition children: type, declarator, body
            declarator = None
            body = None
            for c in node.children:
                if c.type == 'function_declarator':
                    declarator = c
                elif c.type == 'compound_statement':
                    body = c
            symbol = add_symbol('function', node, signature_node=declarator, body_node=body)
            results.append(symbol)

        elif node.type == 'class_specifier' or node.type == 'struct_specifier':
            # signature: from start to body start (type + identifier + base classes)
            # body: compound_statement or field_declaration_list child
            # Usually, the body is a 'field_declaration_list' child
            signature_end_byte = None
            body_node = None
            for c in node.children:
                if c.type == 'field_declaration_list':
                    body_node = c
                    signature_end_byte = c.start_byte
                    break
            if body_node:
                signature_text = code_bytes[node.start_byte:signature_end_byte].decode('utf-8')
                body_text = get_node_text(body_node, code_bytes)
                symbol_name = None
                for c in node.children:
                    if c.type == 'type_identifier' or c.type == 'identifier':
                        symbol_name = get_node_text(c, code_bytes)
                        break
                if not symbol_name:
                    symbol_name = "<unknown>"

                results.append({
                    "symbol": symbol_name,
                    "type": node.type.replace('_specifier',''),
                    "signature": signature_text,
                    "body": body_text
                })
            else:
                # No body found, treat whole node as signature
                symbol = add_symbol(node.type.replace('_specifier',''), node)
                results.append(symbol)

        elif node.type == 'preproc_def' or node.type == 'preproc_function_def':
            # Macro nodes: do not append to results, but parse for completeness if needed
            _ = add_symbol('macro', node)
            # Do not append to results

        elif node.type == 'enum_specifier':
            # signature up to enumerator_list
            enumerator_list_node = None
            for c in node.children:
                if c.type == 'enumerator_list':
                    enumerator_list_node = c
                    break
            if enumerator_list_node:
                signature_text = code_bytes[node.start_byte:enumerator_list_node.start_byte].decode('utf-8')
                body_text = get_node_text(enumerator_list_node, code_bytes)
                symbol_name = None
                for c in node.children:
                    if c.type == 'type_identifier':
                        symbol_name = get_node_text(c, code_bytes)
                        break
                if not symbol_name:
                    symbol_name = "enum"
                results.append({
                    "symbol": symbol_name,
                    "type": "enum",
                    "signature": signature_text,
                    "body": body_text
                })
            else:
                symbol = add_symbol('enum', node)
                results.append(symbol)

        elif node.type == 'declaration':
            # Could be function prototype if it has a function_declarator child
            has_func_decl = False
            func_decl_node = None
            for c in node.children:
                if c.type == 'function_declarator':
                    has_func_decl = True
                    func_decl_node = c
                    break
            if has_func_decl:
                symbol = add_symbol('prototype', node, signature_node=func_decl_node)
                results.append(symbol)

        elif node.type == 'call_expression':
            # Check if callee is uppercase (macro-like call)
            callee = None
            for c in node.children:
                if c.type == 'identifier':
                    callee = get_node_text(c, code_bytes)
                    break
            if callee and callee.isupper():
                # Macro-like call: do not append to results, but parse for completeness if needed
                _ = add_symbol('macro_call', node)
                # Do not append to results

        # Recurse children
        for child in node.children:
            traverse(child)

    traverse(root_node)
    return results

def process_file(filepath):
    print(f"Starting processing file: {filepath}")
    start_time = time.time()
    with open(filepath, "rb") as f:
        code_bytes = f.read()
    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    symbols = extract_functions_and_macros(root_node, code_bytes)
    print(f"Extracted {len(symbols)} symbols from file: {filepath}")
    last_modified = os.path.getmtime(filepath)
    last_modified_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_modified))
    for sym in symbols:
        sym["file_name"] = os.path.basename(filepath)
        sym["file_path"] = filepath
        sym["lastModified"] = last_modified_str
    print(f"Finished processing file: {filepath}")
    elapsed = time.time() - start_time
    print(f"Time taken for file {filepath}: {elapsed:.2f} seconds")
    return symbols

def process_folder(folder_path):
    print(f"Starting processing folder: {folder_path}")
    exts = (".cpp", ".hpp", ".h")
    all_symbols = []
    for root, dirs, files in os.walk(folder_path):
        for fname in files:
            if fname.endswith(exts):
                filepath = os.path.join(root, fname)
                print(f"Processing file: {filepath}")
                try:
                    syms = process_file(filepath)
                    all_symbols.extend(syms)
                except Exception as e:
                    print(f"Error processing file {filepath}: {e}")
                    all_symbols.append({
                        "file_name": fname,
                        "file_path": filepath,
                        "error": str(e)
                    })
                print(f"Completed file: {filepath}")
    print(f"Completed processing folder: {folder_path} with total symbols: {len(all_symbols)}")
    return all_symbols

def chunk_symbol(symbol):
    print(f"Chunking symbol: {symbol.get('symbol', '<unknown>')} of type {symbol.get('type', '')}")
    # Compose the text to chunk: comments (if any), then signature, then body
    comments = symbol.get("comments", "")
    text_to_chunk = ""
    if comments:
        text_to_chunk += comments.strip() + "\n"
    text_to_chunk += symbol.get("signature", "")
    if symbol.get("body"):
        text_to_chunk += "\n" + symbol["body"]

    # For macros and enums: do not chunk, just return as a single chunk
    if symbol.get("type") in ("macro", "enum"):
        return [text_to_chunk]
    # For other types: chunk as usual, with no overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_text(text_to_chunk)
    return chunks

def generate_document(chunk, symbol):
    print(f"Generating document for chunk of symbol: {symbol.get('symbol', '<unknown>')}")
    h = hashlib.sha256()
    h.update(chunk.encode('utf-8'))
    doc_id = h.hexdigest()

    # Compute rel_path relative to REPO_NAME root folder
    # Find "REPO_NAME/" in the file_path and slice after it
    file_path = symbol.get("file_path", "")
    rel_path = file_path.split("REPO_NAME/")[-1] if "REPO_NAME/" in file_path else os.path.basename(file_path)
    opengrok_url = f"{OPENGROK_BASE_URL}{rel_path}?defs={symbol['symbol']}"

    doc = {
        "id": doc_id,
        "code": chunk,  # store only the chunk (no full body)
        "symbol": symbol.get("symbol", ""),
        "type": symbol.get("type", ""),
        "signature": symbol.get("signature", ""),
        # "body": symbol.get("body", ""),  # Removed: do not include full body, just chunk
        "filePath": rel_path,
        "lastModified": symbol.get("lastModified", ""),
        "repo": REPO_NAME,
        "opengrokUrl": opengrok_url,
    }
    return doc

def upload_documents(docs):
    print(f"[UPLOAD] Would upload {len(docs)} docs (DRY_RUN={DRY_RUN})")
    if DRY_RUN:
        with open("dryrun_output.json", "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=4)
        print(f"Dry run: wrote {len(docs)} documents to dryrun_output.json")
        return

    # Check if index exists and is accessible
    try:
        search_client.get_document_count()
    except Exception as e:
        print(f"Failed to connect to search index: {e}")
        return

    batch_size = 500
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        try:
            result = search_client.upload_documents(documents=batch)
            print(f"Uploaded batch of {len(batch)} documents, status: {result[0].status_code}")
        except Exception as e:
            print(f"Failed to upload batch: {e}")

if __name__ == "__main__":
    TARGET_DIR = "/FULL_PATH_TO_REPO/src"
    results = process_folder(TARGET_DIR)

    batch_size = 20
    from collections import defaultdict
    file_to_syms = defaultdict(list)
    for sym in results:
        file_to_syms[sym.get("file_path", "<unknown>")].append(sym)

    doc_buffer = []
    for file_path, syms in file_to_syms.items():
        file_start_time = time.time()
        symbol_chunk_info = []
        for sym in syms:
            chunks = chunk_symbol(sym)
            for chunk in chunks:
                symbol_chunk_info.append((sym, chunk))
        all_chunks = [chunk for (_, chunk) in symbol_chunk_info]
        if not all_chunks:
            continue
        print(f"Generating embeddings for {len(all_chunks)} chunks from file: {file_path}")
        embeddings_for_chunks = []
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            print(f"Embedding batch of {len(batch)} chunks for file: {file_path}")
            batch_embeddings = embeddings.embed_documents(batch)
            print(f"Received {len(batch_embeddings)} embeddings for file: {file_path}")
            embeddings_for_chunks.extend(batch_embeddings)
        assert len(embeddings_for_chunks) == len(symbol_chunk_info)
        docs_for_file = []
        for (sym, chunk), embedding in zip(symbol_chunk_info, embeddings_for_chunks):
            doc = generate_document(chunk, sym)
            doc["embedding"] = embedding
            docs_for_file.append(doc)
        doc_buffer.extend(docs_for_file)
        print(f"Prepared {len(symbol_chunk_info)} documents for file: {file_path}")
        file_elapsed = time.time() - file_start_time
        print(f"Total time for embeddings + docs for file {file_path}: {file_elapsed:.2f} seconds")
        if len(doc_buffer) >= 1000:
            upload_documents(doc_buffer)
            doc_buffer.clear()
    # After all files, flush any remaining docs
    if doc_buffer:
        upload_documents(doc_buffer)