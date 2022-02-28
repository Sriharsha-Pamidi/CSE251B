##Taken from https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/build_vocab.py
import nltk
import numpy as np
import torch
from collections import Counter
from pycocotools.coco import COCO
import os, pickle, json, csv, copy
from gensim.models import Word2Vec


# A simple wrapper class for Vocabulary. No changes are required in this file
class Vocabulary(object):
    """Simple vocabulary wrapper."""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
    
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
    
    def __call__(self, word):
        if not word.lower() in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word.lower()]
    
    def __len__(self):
        return len(self.word2idx)


def load_vocab(json, threshold):
    if os.path.isfile('savedVocab'):
        with open('savedVocab', 'rb') as savedVocab:
            vocab = pickle.load(savedVocab)
            print("Using the saved vocab.")
    
    else:
        vocab = build_vocab(json, threshold)
        with open('savedVocab', 'wb') as savedVocab:
            pickle.dump(vocab, savedVocab)
            print("Saved the vocab.")
    
    return vocab


def build_vocab(json, threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        
        if (i + 1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i + 1, len(ids)))
    
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    
    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def train_word2vec_model(json):
    file_exists = os.path.exists('word_vectors.pkl')
    
    if not file_exists:
        coco = COCO(json)
        ids = coco.anns.keys()
        corpus = []
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]['caption'])
            corpus.append(caption.lower())
        
        model = Word2Vec(corpus, vector_size=300, window=5, min_count=5, workers=1)
        # vocab, embeddings = [], []
        # with open('glove.6B.300d.txt', 'rt') as fi:
        #     full_content = fi.read().strip().split('\n')
        # for i in range(len(full_content)):
        #     i_word = full_content[i].split(' ')[0]
        #     i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        #     vocab.append(i_word)
        #     embeddings.append(i_embeddings)
        #
        # vocab_npa = np.array(vocab)
        # embs_npa = np.array(embeddings)
        #
        # # insert '<pad>' and '<unk>' tokens at start of vocab_npa.
        # vocab_npa = np.insert(vocab_npa, 0, '<pad>')
        # vocab_npa = np.insert(vocab_npa, 1, '<unk>')
        #
        # pad_emb_npa = np.zeros((1, embs_npa.shape[1]))  # embedding for '<pad>' token.
        # unk_emb_npa = np.mean(embs_npa, axis=0, keepdims=True)  # embedding for '<unk>' token.
        #
        # # insert embeddings for pad and unk tokens at top of embs_npa.
        # embs_npa = np.vstack((pad_emb_npa, unk_emb_npa, embs_npa))
        pickle.dump(torch.FloatTensor(model.wv.vectors), open('word_vectors.pkl', 'wb'))
    else:
        pass


def load_word2vec_model():
#     wv_vectors = pickle.load('word_vectors.pkl')
    file = open("word_vectors.pkl",'rb')
    wv_vectors = pickle.load(file)
    file.close()
    return wv_vectors
