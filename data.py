# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to read the train/eval/test data from file and process it, and read the vocab data from file and process it"""

import glob
import random
import struct
import csv
from tensorflow.core.example import example_pb2
from os.path import join
import nlc_data

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
# SENTENCE_START = '<sos>'
# SENTENCE_END = '<eos>'

_PAD = '<pad>' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
_UNK = '<unk>' # This has a vocab id, which is used to represent out-of-vocabulary words
_SOS = '<sos>' # This has a vocab id, which is used at the start of every decoder input sequence
_EOS = '<eos>' # This has a vocab id, which is used at the end of untruncated target sequences

_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3
# Note: none of <s>, </s>, [PAD], [UNK], [START], [STOP] should appear in the vocab file.


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                w = line.strip('\n')
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
        print "Finished constructing vocabulary of %i total words. Last word added: %s" % (self._count, self._id_to_word[self._count-1])

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[_UNK]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def write_metadata(self, fpath):
        """Writes metadata file for Tensorboard word embedding visualizer as described here:
          https://www.tensorflow.org/get_started/embedding_viz

        Args:
          fpath: place to write the metadata file
        """
        print "Writing word embedding metadata file to %s..." % (fpath)
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in xrange(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass):
    """Generates tf.Examples from data files.

      Binary data format: <length><blob>. <length> represents the byte size
      of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
      the tokenized article text and summary.

    Args:
      data_path:
        Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
      single_pass:
        Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.

    Yields:
      Deserialized tf.Example.
    """
    while True:
        list_x = []
        for line in file(data_path + '.x.txt'):
            list_x.append(line.strip())
        list_y = []
        for line in file(data_path + '.y.txt'):
            list_y.append(line.strip())
        filelist = zip(list_x, list_y)
        if not single_pass:
            random.shuffle(filelist)

        for i in range(len(filelist)):
            yield (filelist[i][0], filelist[i][1])
        if single_pass:
            print "example_generator completed reading all datafiles. No more data."
            break


def article2ids(article_words, vocab):
    """Map the article words to their ids. Also return a list of OOVs in the article.

    Args:
      article_words: list of words (strings)
      vocab: Vocabulary object

    Returns:
      ids:
        A list of word ids (integers); OOVs are represented by their temporary article OOV number. If the vocabulary size is 50k and the article has 3 OOVs, then these temporary OOV numbers will be 50000, 50001, 50002.
      oovs:
        A list of the OOV words in the article (strings), in the order corresponding to their temporary article OOV numbers."""
    ids = []
    oovs = []
    unk_id = vocab.word2id(_UNK)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is OOV
            if w not in oovs: # Add to list of OOVs
                oovs.append(w)
            oov_num = oovs.index(w) # This is 0 for the first article OOV, 1 for the second article OOV...
            ids.append(vocab.size() + oov_num) # This is e.g. 50000 for the first article OOV, 50001 for the second...
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    """Map the abstract words to their ids. In-article OOVs are mapped to their temporary OOV numbers.

    Args:
      abstract_words: list of words (strings)
      vocab: Vocabulary object
      article_oovs: list of in-article OOV words (strings), in the order corresponding to their temporary article OOV numbers

    Returns:
      ids: List of ids (integers). In-article OOV words are mapped to their temporary OOV numbers. Out-of-article OOV words are mapped to the UNK token id."""
    ids = []
    unk_id = vocab.word2id(_UNK)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id: # If w is an OOV word
            if w in article_oovs: # If w is an in-article OOV
                vocab_idx = vocab.size() + article_oovs.index(w) # Map to its temporary article OOV number
                ids.append(vocab_idx)
            else: # If w is an out-of-article OOV
                ids.append(unk_id) # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    """Maps output ids to words, including mapping in-article OOVs from their temporary ids to the original OOV string (applicable in pointer-generator mode).

    Args:
      id_list: list of ids (integers)
      vocab: Vocabulary object
      article_oovs: list of OOV words (strings) in the order corresponding to their temporary article OOV ids (that have been assigned in pointer-generator mode), or None (in baseline mode)

    Returns:
      words: list of words (strings)
    """
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i) # might be [UNK]
        except ValueError as e: # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e: # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def show_art_oovs(article, vocab):
    """Returns the article string, highlighting the OOVs by placing __underscores__ around them"""
    unk_token = vocab.word2id(_UNK)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(w)==unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    """Returns the abstract string, highlighting the article OOVs with __underscores__.

    If a list of article_oovs is provided, non-article OOVs are differentiated like !!__this__!!.

    Args:
      abstract: string
      vocab: Vocabulary object
      article_oovs: list of words (strings), or None (in baseline mode)
    """
    unk_token = vocab.word2id(_UNK)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token: # w is oov
            if article_oovs is None: # baseline mode
                new_words.append("__%s__" % w)
            else: # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else: # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str
