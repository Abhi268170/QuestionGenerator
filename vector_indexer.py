import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document

class VectorIndexer:
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2', hybrid_weight: float = 0.5):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )  # Proper LangChain embedding object
        self.hybrid_weight = hybrid_weight
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None

    def load_chunks(self, chunk_dir: str) -> Tuple[List[str], List[Dict]]:
        docs, meta = [], []
        sorted_files = sorted(Path(chunk_dir).glob("chunk_*.txt"), 
                            key=lambda x: int(x.stem.split('_')[-1]))
        for idx, path in enumerate(sorted_files):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        docs.append(content)
                        meta.append({
                            "source": path.name,
                            "chunk_id": idx,
                            "file_path": str(path)
                        })
            except Exception as e:
                print(f"Error loading {path.name}: {str(e)}")
        return docs, meta

    def build_indices(self, documents: List[str]):
        if not documents:
            raise ValueError("No documents provided for indexing")
            
        print("Building FAISS index...")
        embeddings = self.embedding_model.embed_documents(documents)  # Updated embedding call
        self.vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(documents, embeddings)),
            embedding=self.embedding_model
        )
        
        print("Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_texts(
            texts=documents,
            metadatas=[{"source": f"chunk_{i}"} for i in range(len(documents))]
        )
        
        print("Creating hybrid retriever...")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_store.as_retriever()],
            weights=[self.hybrid_weight, 1-self.hybrid_weight]
        )

    def save_index(self, save_dir: str):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        print("Saving FAISS index...")
        self.vector_store.save_local(Path(save_dir)/"faiss_index")
        
        print("Saving BM25 index...")
        bm25_data = {
            "docs": [doc.page_content for doc in self.bm25_retriever.docs],
            "metadatas": [dict(doc.metadata) for doc in self.bm25_retriever.docs]
        }
        with open(Path(save_dir)/"bm25_index.json", "w", encoding='utf-8') as f:
            json.dump(bm25_data, f, ensure_ascii=False, indent=2)

    def load_index(self, save_dir: str):
        print("Loading FAISS index...")
        self.vector_store = FAISS.load_local(
            folder_path=Path(save_dir)/"faiss_index",
            embeddings=self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        print("Loading BM25 index...")
        with open(Path(save_dir)/"bm25_index.json", "r", encoding='utf-8') as f:
            bm25_data = json.load(f)
            
        self.bm25_retriever = BM25Retriever.from_texts(
            texts=bm25_data["docs"],
            metadatas=bm25_data["metadatas"]
        )
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_store.as_retriever()],
            weights=[self.hybrid_weight, 1-self.hybrid_weight]
        )