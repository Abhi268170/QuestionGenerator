import argparse
import sys
from pathlib import Path
from document_processor import process_documents
from vector_indexer import VectorIndexer

class RAGSystem:
    def __init__(self, data_dir: str, chunk_dir: str, db_dir: str):
        self.data_dir = data_dir
        self.chunk_dir = chunk_dir
        self.db_dir = db_dir
        self.indexer = VectorIndexer()
        self.debug = False

    def process(self, debug: bool = False):
        self.debug = debug
        try:
            print("\n🚀 Starting document processing...")
            if not Path(self.data_dir).exists():
                raise FileNotFoundError(f"Input directory {self.data_dir} not found")
                
            chunks = process_documents(
                input_dir=self.data_dir,
                output_dir=self.chunk_dir,
                debug=self.debug
            )
            
            print("\n🔨 Building vector database...")
            documents, metadata = self.indexer.load_chunks(self.chunk_dir)
            if not documents:
                raise ValueError("No valid chunks found for indexing")
                
            self.indexer.build_indices(documents)
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
            
            print(f"\nSearching for: '{question}'")
            results = self.indexer.ensemble_retriever.invoke(question)[:top_k]  # Updated method
            
            if not results:
                print("\nNo relevant results found")
                return
                
            print(f"\nTop {len(results)} results:")
            for i, doc in enumerate(results, 1):
                # Safeguard against invalid metadata
                metadata = getattr(doc, 'metadata', {}) or {}
                source = metadata.get('source', 'unknown')
                content = getattr(doc, 'page_content', '')[:250]
                
                print(f"\n📄 Result {i}")
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

    # FIXED: Removed erroneous .add_parser chain
    process_parser = subparsers.add_parser('process', help='Process documents')
    process_parser.add_argument('--data', default="./docs", help='Input documents directory')
    process_parser.add_argument('--chunks', default="./chunks", help='Output chunks directory')  # Corrected line
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
            rag = RAGSystem("", "", args.db)
            rag.query(args.question, args.top_k)
    except Exception as e:
        print(f"\n❌ Critical error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()