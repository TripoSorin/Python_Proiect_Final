from pathlib import Path

import chromadb
import pymupdf4llm
from sentence_transformers import SentenceTransformer

#configurari
DOCUMENTS_DIR = Path("documents")
COLLECTION_NAME = "my_collection"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
BATCH_SIZE = 100


#impartire text in chunk-uri de CHUNK_SIZE caractere, cu suprapunere CHUNK_OVERLAP
def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        # OVERLAP logic
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


#citire fisier si returnare lista de tupluri/perechi nr_pagina, text
def read_pages(file_path: Path) -> list[tuple[int, str]]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":

        pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
        result = []
        for index, page in enumerate(pages):
            #nr pagina vine din metadata, in lipsa ei folosim index+1
            page_number = page["metadata"].get("page_number", index + 1)
            result.append((page_number, page["text"]))
        return result

    if suffix == ".txt":
        #.txt e fara pagini deci o sa consideram ca avem o singura pagina
        return [(1, file_path.read_text(encoding="utf-8-sig"))]

    return []


#indexare fisier, citire -> chunk-uri -> metadata -> stergere versiune veche -> upsert
def index_file(collection, model, file_path: Path) -> int:
    #trei liste paralele: elementul de pe pozitia i din fiecare apartine aceluiasi chunk
    ids = []
    documents = []
    metadatas = []
    chunk_number = 1  #numaram chunk-uri de la 1 la n, pe tot documentul

    for page_number, page_text in read_pages(file_path):
        for chunk in chunk_text(page_text):
            # id unic, ca upsert sa poata actualiza chunk-ul la reindexare
            ids.append(f"{file_path.name}:{chunk_number}")
            documents.append(chunk)
            #metadata: source- fisierul, page - nr. pagina, chunk - numarul chunk-ului
            metadatas.append(
                {
                    "source": file_path.name,
                    "page": page_number,
                    "chunk": chunk_number,
                }
            )
            chunk_number += 1

    # fisier fara text -> nu avem ce indexa
    if not documents:
        return 0

    # stergem versiunea veche a fisierului
    collection.delete(where={"source": file_path.name})

    #inseram in loturi de BATCH_SIZE chunk-uri, ca sa nu incarcam prea mult memoria
    for start in range(0, len(documents), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_docs = documents[start:end]
        collection.upsert(
            ids=ids[start:end],
            documents=batch_docs,
            metadatas=metadatas[start:end],
            # embeddings calculate de noi, normalizate (potrivit pentru distanta cosine)
            embeddings=model.encode(batch_docs, normalize_embeddings=True).tolist(),
        )

    return len(documents)



def main():
    if not DOCUMENTS_DIR.is_dir():
        raise SystemExit(f"Folderul '{DOCUMENTS_DIR}' nu exista. Creeaza-l si pune documente in el.")

    #conectare server
    try:
        client = chromadb.HttpClient(host="localhost", port=8000)
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,  # embeddings le calculam noi
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as error:
        raise SystemExit(
            f"Nu ma pot conecta la Chroma ({error}). "
            "Porneste serverul: uv run chroma run --path chroma_data --port 8000"
        )

    print("Incarc modelul de embeddings...")
    # modelul de embeddings
    model = SentenceTransformer(MODEL_NAME)

    total = 0
    #parcurgere fisiere din folder, acceptam doar .txt si .pdf
    for file_path in sorted(DOCUMENTS_DIR.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in (".txt", ".pdf"):
            print(f"Ignor {file_path.name} (tip neacceptat)")
            continue

        count = index_file(collection, model, file_path)
        print(f"{file_path.name}: {count} chunk-uri indexate")
        total += count

    print(f"Gata. Total: {total} chunk-uri in colectia '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
