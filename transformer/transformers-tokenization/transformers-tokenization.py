import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3
        
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        items = []
        for sentence in texts :
            sentence=sentence.lower().split()
            items.extend(sentence)
        unique_sorted_items = sorted(list(dict.fromkeys(items)))
        counter=4    
        for item in unique_sorted_items:
            self.word_to_id[item] = counter
            self.id_to_word[counter] = item
            counter += 1
        self.vocab_size = len(self.word_to_id)    
        pass
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words=text.lower().split()
        number_list=[]    
        for word in words:
            if word in self.word_to_id :
                id=self.word_to_id[word]
                number_list.append(id)
            else :
                number_list.append(self.word_to_id[self.unk_token])
        return number_list
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words_list=[]
        for id in ids:
            if id in self.id_to_word :
                word=self.id_to_word[id]
                words_list.append(word)
            else :
                words_list.append(self.unk_token)    
        return " ".join(words_list)