import argparse
import sys
from pathlib import Path
from document_processor import process_documents
from vector_indexer import VectorIndexer
from query_rewriter import QueryRewriter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from sklearn.metrics.pairwise import cosine_similarity

class RAGSystem:
    def __init__(self, data_dir: str, chunk_dir: str, db_dir: str):
        self.data_dir = data_dir
        self.chunk_dir = chunk_dir
        self.db_dir = db_dir
        self.indexer = VectorIndexer()
        self.query_rewriter = QueryRewriter()
        self.debug = False  # Removed erroneous build_indices call

    def process(self, debug: bool = False):
        self.debug = debug
        try:
            print("\n🚀 Starting document processing...")
            if not Path(self.data_dir).exists():
                raise FileNotFoundError(f"Input directory {self.data_dir} not found")
                
            process_documents(
                input_dir=self.data_dir,
                output_dir=self.chunk_dir,
                debug=self.debug
            )
            
            print("\n🔨 Building vector database...")
            documents, metadata = self.indexer.load_chunks(self.chunk_dir)
            if not documents:
                raise ValueError("No valid chunks found for indexing")
                
            self.indexer.build_indices(documents, metadata)  # Pass metadata
            self.indexer.save_index(self.db_dir)
            print(f"\n✅ Successfully built vector database in {self.db_dir}")
            print(f"Total chunks indexed: {len(documents)}")
            
        except Exception as e:
            print(f"\n❌ Processing failed: {str(e)}")
            sys.exit(1)

    def query(self, question: str, top_k: int = 3):
        try:
            print("\n🔍 Loading vector database...")
            self.indexer.load_index(self.db_dir)
            
            original_query = question
            rewritten_query = self.query_rewriter.rewrite(question)
            if self.debug:
                print(f"Original query: {original_query}")
                print(f"Rewritten query: {rewritten_query}")
            
            results = self.indexer.hierarchical_retrieve(
                rewritten_query, 
                broad_k=top_k*4, 
                refine_k=top_k
            )
            
            if not results:
                print("\nNo relevant results found")
                return
                
            print(f"\nTop {len(results)} results:")
            for i, doc in enumerate(results, 1):
                metadata = getattr(doc, 'metadata', {}) or {}
                source = metadata.get('source', 'unknown')
                content = getattr(doc, 'page_content', '')[:350]
                
                # Now uses the properly imported cosine_similarity
                query_vec = self.indexer.tfidf_vectorizer.transform([rewritten_query])
                doc_vec = self.indexer.tfidf_vectorizer.transform([doc.page_content])
                tfidf_score = cosine_similarity(query_vec, doc_vec)[0][0]
                
                print(f"\n📄 Result {i} (Score: {tfidf_score:.2f})")
                print(f"📂 Source: {source}")
                print(f"📝 Content: {content}...")
                print("-" * 80)
                
        except Exception as e:
            print(f"\n❌ Query failed: {str(e)}")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="RAG System CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    process_parser = subparsers.add_parser('process', help='Process documents')
    process_parser.add_argument('--data', default="./docs", help='Input documents directory')
    process_parser.add_argument('--chunks', default="./chunks", help='Output chunks directory')
    process_parser.add_argument('--db', default="./vector_db", help='Vector database directory')
    process_parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    query_parser = subparsers.add_parser('query', help='Search documents')
    query_parser.add_argument('question', help='Search query')
    query_parser.add_argument('--top_k', type=int, default=3, help='Number of results')
    query_parser.add_argument('--db', default="./vector_db", help='Vector database directory')

    args = parser.parse_args()
    
    try:
        if args.command == 'process':
            rag = RAGSystem(args.data, args.chunks, args.db)
            rag.process(debug=args.debug)
        elif args.command == 'query':
            rag = RAGSystem("", "", args.db)  # chunk_dir not needed for query
            rag.query(args.question, args.top_k)
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()