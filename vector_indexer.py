import json
import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

class VectorIndexer:
    def __init__(self, embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 hybrid_weight: float = 0.5, content_threshold: float = 0.4):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.hybrid_weight = hybrid_weight
        self.content_threshold = content_threshold
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.content_richness_scores = None

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

    def _calculate_content_richness(self, documents: List[str]) -> np.ndarray:
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
        sentence_lens = np.array([len(doc.split()) for doc in documents])
        tfidf_scores = np.asarray(tfidf_matrix.mean(axis=1)).flatten()
        return (tfidf_scores + 0.5 * (sentence_lens / sentence_lens.max()))

    def build_indices(self, documents: List[str], metadata: List[Dict]):
        if not documents:
            raise ValueError("No documents provided for indexing")
            
        print("Building content richness scores...")
        self.content_richness_scores = self._calculate_content_richness(documents)
            
        print("Building FAISS index...")
        embeddings = self.embedding_model.embed_documents(documents)
        self.vector_store = FAISS.from_embeddings(
            text_embeddings=list(zip(documents, embeddings)),
            embedding=self.embedding_model
        )
        
        print("Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_texts(
            texts=documents,
            metadatas=metadata
        )
        
        print("Creating hybrid retriever...")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_store.as_retriever()],
            weights=[self.hybrid_weight, 1-self.hybrid_weight]
        )

    def hierarchical_retrieve(self, query: str, 
                            broad_k: int = 20, 
                            refine_k: int = 5) -> List[Document]:
        # Broad initial retrieval
        broad_results = self.ensemble_retriever.invoke(query)[:broad_k]
        
        # Ensure valid indices and matching array sizes
        valid_results = [doc for doc in broad_results if 'chunk_id' in doc.metadata]
        doc_indices = [doc.metadata['chunk_id'] for doc in valid_results]
        
        if not doc_indices:
            return []
            
        # Get scores only for valid documents
        richness_scores = self.content_richness_scores[doc_indices]
        query_embedding = self.embedding_model.embed_query(query)
        doc_embeddings = self.vector_store.index.reconstruct_batch(doc_indices)
        semantic_scores = np.dot(doc_embeddings, query_embedding)
        
        # Ensure TF-IDF vectors match original feature space
        doc_texts = [doc.page_content for doc in valid_results]
        doc_vectors = self.tfidf_vectorizer.transform(doc_texts)
        keyword_scores = cosine_similarity(
            self.tfidf_vectorizer.transform([query]), 
            doc_vectors
        ).flatten()
        
        # Verify array dimensions match
        if len(semantic_scores) != len(keyword_scores) or len(semantic_scores) != len(richness_scores):
            min_length = min(len(semantic_scores), len(keyword_scores), len(richness_scores))
            semantic_scores = semantic_scores[:min_length]
            keyword_scores = keyword_scores[:min_length]
            richness_scores = richness_scores[:min_length]
        
        combined_scores = 0.6*semantic_scores + 0.4*keyword_scores + 0.2*richness_scores
        
        # Sort and refine
        sorted_indices = np.argsort(combined_scores)[::-1]
        return [valid_results[i] for i in sorted_indices[:refine_k]]

    def save_index(self, db_path: str):
        """Save all index components to disk"""
        path = Path(db_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        self.vector_store.save_local(str(path / "faiss_index"))
        
        # Save BM25 retriever documents and metadata
        with open(path / "bm25_retriever.pkl", "wb") as f:
            pickle.dump({
                'docs': [doc.page_content for doc in self.bm25_retriever.docs],
                'metadata': [doc.metadata for doc in self.bm25_retriever.docs]
            }, f)
        
        # Save TF-IDF components
        with open(path / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save content richness scores
        np.save(path / "content_scores.npy", self.content_richness_scores)

    def load_index(self, db_path: str):
        """Load all index components from disk"""
        path = Path(db_path)
        
        # Load FAISS index
        self.vector_store = FAISS.load_local(
            str(path / "faiss_index"),
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        
        # Load BM25 retriever
        with open(path / "bm25_retriever.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25_retriever = BM25Retriever.from_documents(
            documents=[
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(bm25_data['docs'], bm25_data['metadata'])
            ]
        )
        
        # Recreate ensemble retriever after loading components
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_store.as_retriever()],
            weights=[self.hybrid_weight, 1-self.hybrid_weight]
        )
        
        # Load TF-IDF components
        with open(path / "tfidf_vectorizer.pkl", "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        # Load content richness scores
        self.content_richness_scores = np.load(path / "content_scores.npy")