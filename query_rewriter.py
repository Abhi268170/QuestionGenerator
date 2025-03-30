import nltk
from nltk.corpus import wordnet
from typing import List
import re

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

class QueryRewriter:
    def __init__(self):
        self.lemmatizer = nltk.WordNetLemmatizer()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
    def _expand_synonyms(self, word: str) -> List[str]:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)
    
    def _paraphrase_query(self, query: str) -> str:
        tagged = nltk.pos_tag(nltk.word_tokenize(query))
        expanded = []
        
        for word, tag in tagged:
            if word.lower() in self.stop_words:
                continue
                
            pos = None
            if tag.startswith('N'): pos = wordnet.NOUN
            elif tag.startswith('V'): pos = wordnet.VERB
            elif tag.startswith('J'): pos = wordnet.ADJ
            elif tag.startswith('R'): pos = wordnet.ADV
            
            if pos:
                lemma = self.lemmatizer.lemmatize(word, pos)
                expanded.append(lemma)
                synonyms = self._expand_synonyms(lemma)[:2]
                expanded.extend(synonyms)
            else:
                expanded.append(word)
                
        return ' '.join(expanded)
    
    def rewrite(self, query: str) -> str:
        query = re.sub(r'[^\w\s]', '', query.lower())
        paraphrased = self._paraphrase_query(query)
        seen = set()
        return ' '.join([w for w in paraphrased.split() if not (w in seen or seen.add(w))])