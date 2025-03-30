import re
import os
import fitz
import docx
import nltk
import numpy as np
import pytesseract
import pdfplumber
from typing import List, Tuple
from pathlib import Path
from PIL import Image
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class SemanticChunker:
    def __init__(self, 
                 window_size: int = 4,
                 smoothing_window: int = 2,
                 similarity_threshold: float = 0.15,
                 min_chunk_length: int = 4,
                 max_chunk_length: int = 8,
                 debug: bool = False):
        self.window_size = window_size
        self.smoothing_window = smoothing_window
        self.similarity_threshold = similarity_threshold
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.debug = debug
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update({'example', 'exercise', 'solution', 'note', 'summary'})
        self.heading_pattern = re.compile(r'(?i)^(\d+\.\d+(?:\.\d+)*|CHAPTER\s+\d+|Section\s+\d+|^[A-Z][A-Z0-9\s]+)\s+.*')
        self.figure_pattern = re.compile(r'^Figure\s+\d+\.\d+:', re.IGNORECASE)
        self.question_pattern = re.compile(r'^(?:Q\d*|Question|Practice Problem)\s+\d+[.:]')

    def _preprocess_terms(self, text: str) -> List[str]:
        tokens = re.findall(r'\b[\w+]+\b', text.lower())
        return [self.stemmer.stem(t) for t in tokens if t not in self.stop_words and len(t) > 2]

    def _compute_similarity(self, window1: List[str], window2: List[str]) -> float:
        vocab = list(set(window1 + window2))
        idf = {t: np.log((2 + 1e-10) / (1 + (t in window1) + (t in window2))) for t in vocab}
        tfidf1 = [window1.count(t) * idf[t] for t in vocab]
        tfidf2 = [window2.count(t) * idf[t] for t in vocab]
        dot = np.dot(tfidf1, tfidf2)
        norm = np.linalg.norm(tfidf1) * np.linalg.norm(tfidf2)
        return dot / (norm + 1e-10)

    def _calculate_depth_scores(self, sentences: List[str]) -> List[float]:
        if len(sentences) < self.window_size * 2: 
            return []
        sentence_terms = [self._preprocess_terms(s) for s in sentences]
        scores = [1 - self._compute_similarity(
            [t for terms in sentence_terms[i:i+self.window_size] for t in terms],
            [t for terms in sentence_terms[i+self.window_size:i+self.window_size*2] for t in terms]
        ) for i in range(len(sentence_terms) - self.window_size * 2 + 1)]
        return np.convolve(scores, np.ones(self.smoothing_window)/self.smoothing_window, mode='valid').tolist()

    def _detect_structural_breaks(self, sentences: List[str]) -> List[int]:
        breaks = []
        for idx, sentence in enumerate(sentences):
            if self.heading_pattern.match(sentence.strip()):
                breaks.extend([max(0, idx-1), idx])
            if self.figure_pattern.match(sentence):
                breaks.append(idx)
            if self.question_pattern.match(sentence):
                breaks.append(idx)
        return sorted(set(filter(lambda x: 0 <= x < len(sentences), breaks)))

    def chunk_text(self, text: str) -> List[str]:
        text = re.sub(r'(\d+)\.(\d+)', r'\1. \2', text)
        sentences = sent_tokenize(text)
        structural_breaks = self._detect_structural_breaks(sentences)
        scores = self._calculate_depth_scores(sentences)
        semantic_breaks = [(i + self.window_size) * 2 for i, score in enumerate(scores) 
                          if score > (np.mean(scores) + np.std(scores) * self.similarity_threshold)] if scores else []
        all_breaks = sorted(set(structural_breaks + semantic_breaks))
        chunks = []
        prev = 0
        for br in all_breaks:
            if prev < br <= prev + self.max_chunk_length and br - prev >= self.min_chunk_length:
                chunks.append(' '.join(sentences[prev:br]).strip())
                prev = br
        if prev < len(sentences):
            chunks.append(' '.join(sentences[prev:]).strip())
        return [c for c in chunks if c]

def load_documents(file_path: str) -> List[str]:
    try:
        if file_path.endswith('.pdf'):
            with fitz.open(file_path) as doc:
                text = []
                for page in doc:
                    page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES).replace('\n', ' ')
                    if page_text.strip():
                        text.append(page_text)
                return [' '.join(text)] if text else []
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            return ['\n'.join(para.text for para in doc.paragraphs if para.text.strip())]
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return [content] if content.strip() else []
        return []
    except Exception as e:
        print(f"Error processing {Path(file_path).name}: {str(e)}")
        return []

def process_scanned_pdf(pdf_path: str) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return ' '.join([pytesseract.image_to_string(page.to_image(resolution=400).original, 
                       config='--psm 1 --oem 3') for page in pdf.pages])
    except Exception as e:
        print(f"OCR failed for {Path(pdf_path).name}: {str(e)}")
        return ''

def preprocess_text(text: str) -> str:
    text = re.sub(r'\b\d{4}-\d{2,4}\b|\[\d+\]|Figure\s+\d+\.\d+:.*?(\n\n|$)', '', text, flags=re.DOTALL)
    return re.sub(r'\s+', ' ', text).strip()

def process_documents(input_dir: str = "./docs", output_dir: str = "./chunks", debug: bool = False) -> List[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    chunker = SemanticChunker(debug=debug)
    all_chunks = []
    
    valid_extensions = ('.pdf', '.docx', '.txt')
    processed_files = 0
    
    for file_path in Path(input_dir).iterdir():
        if file_path.suffix.lower() not in valid_extensions:
            continue
            
        try:
            if debug:
                print(f"Processing {file_path.name}")
                
            raw_docs = load_documents(str(file_path))
            if not raw_docs or not any(raw_docs):
                print(f"Skipped empty/unprocessable file: {file_path.name}")
                continue
                
            for doc in raw_docs:
                cleaned = preprocess_text(doc)
                if not cleaned.strip():
                    print(f"Empty content after preprocessing: {file_path.name}")
                    continue
                    
                chunks = chunker.chunk_text(cleaned)
                if chunks:
                    all_chunks.extend(chunks)
                    processed_files += 1
                    
        except Exception as e:
            print(f"Failed to process {file_path.name}: {str(e)}")
            continue

    if not all_chunks:
        raise ValueError("No valid content found in input documents")
        
    for i, chunk in enumerate(all_chunks):
        chunk_path = Path(output_dir)/f"chunk_{i:04d}.txt"
        with open(chunk_path, "w", encoding='utf-8') as f:
            f.write(chunk)
            
    print(f"Processed {processed_files} files, generated {len(all_chunks)} chunks")
    return all_chunks