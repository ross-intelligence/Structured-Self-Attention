from __future__ import print_function
import torch
seed = 233
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import torchtext
from keras.preprocessing.sequence import pad_sequences
import os

USELESS_LABEL = "-"

class NLIDataloader():

    def __init__(self, multinli_path, snli_path, embedding_name):
        self.multinli_path = multinli_path
        self.snli_path = snli_path
        self.embedding_name = embedding_name


    def load_nlidata(self, batch_size, gpu_option, tokenizer, dataset="snli", max_len=50):
        if tokenizer == "spacy":
            TEXT_FIELD = torchtext.data.Field(sequential=True, tokenize=torchtext.data.get_tokenizer('spacy'),
                                              batch_first=True, include_lengths=True, lower=True, fix_length=max_len)
        else:
            TEXT_FIELD = torchtext.data.Field(sequential=True, tokenize=(lambda s: s.split(' ')),
                                               batch_first=True, include_lengths=True, lower=True, fix_length=max_len)
        LABEL_FIELD = torchtext.data.Field(sequential=False, batch_first=True, unk_token=None)

        if dataset == "snli":
            print("Loading snlidata json")
            snli_train, snli_dev, snli_test = self.load_snlidata_json(TEXT_FIELD, LABEL_FIELD)

            print("Building text vocab")
            TEXT_FIELD.build_vocab(snli_train, snli_dev, snli_test, vectors=self.embedding_name)
            print("Building label vocab")
            LABEL_FIELD.build_vocab(snli_train, snli_dev, snli_test)

            print("Creating iterators")
            snli_train_iter, snli_val_iter, snli_test_iter \
                = torchtext.data.BucketIterator.splits(datasets=(snli_train, snli_dev, snli_test),
                                                    batch_size=batch_size,
                                                    repeat = False,
                                                    sort_key=lambda x: len(x.premise),
                                                    device=gpu_option)

            print("Done.")
            return (snli_train_iter, snli_val_iter, snli_test_iter), TEXT_FIELD, LABEL_FIELD, None

        elif dataset == "multinli":
            GENRE_FIELD = torchtext.data.Field(sequential=False, batch_first=True, unk_token=None)
            print("Loading multidata json")
            multinli_train, multinli_match, multinli_mis_match = self.load_mutidata_json(TEXT_FIELD, LABEL_FIELD, GENRE_FIELD)

            print("Building text vocab")
            TEXT_FIELD.build_vocab(multinli_train, multinli_match, multinli_mis_match, vectors=self.embedding_name)
            print("Building label vocab")
            LABEL_FIELD.build_vocab(multinli_train, multinli_match, multinli_mis_match)

            print("Building genre vocab")
            GENRE_FIELD.build_vocab(multinli_train, multinli_match, multinli_mis_match)

            print("Creating iterators")
            multinli_train_iter, multinli_match_iter, multinli_mis_match_iter \
                = torchtext.data.BucketIterator.splits(datasets=(multinli_train, multinli_match, multinli_mis_match),
                                                    batch_size=batch_size,
                                                    repeat=False,
                                                    sort_key=lambda x: len(x.premise),
                                                    device=gpu_option)

            print("Done.")
            return (multinli_train_iter, multinli_match_iter, multinli_mis_match_iter), TEXT_FIELD, LABEL_FIELD, GENRE_FIELD


    def load_mutidata_json(self, text_field, label_field, genre_field):
        """
        load the data in json format and form torchtext dataset
        :param path:
        :return:
        """

        train, match, mis_match = torchtext.data.TabularDataset.splits(path=self.multinli_path, format='json',
                                                                       train='multinli_1.0_train.jsonl',
                                                                       validation='multinli_1.0_dev_matched.jsonl',
                                                                       test='multinli_1.0_dev_mismatched.jsonl',
                                                                    #    train='train.jsonl',
                                                                    #    validation='train.jsonl',
                                                                    #    test='train.jsonl',
                                                                       fields={'sentence1': ('premise', text_field),
                                                                               'sentence2': ('hypothesis', text_field),
                                                                               # 'sentence1_binary_parse': ('premise_parse', text_field),
                                                                               # 'sentence2_binary_parse': ('hypothesis_parse', text_field),
                                                                               'gold_label': ('label', label_field),
                                                                               'genre': ('genre', genre_field)},
                                                                       filter_pred=lambda ex: ex.label != USELESS_LABEL)


        return train, match, mis_match

    def load_snlidata_json(self, text_field, label_field):
        train, dev, test = torchtext.data.TabularDataset.splits(path=self.snli_path, format='json',
                                                                       train='snli_1.0_dev.jsonl',
                                                                       validation='snli_1.0_dev.jsonl',
                                                                       test='snli_1.0_test.jsonl',
                                                                    #    train='train.jsonl',
                                                                    #    validation='train.jsonl',
                                                                    #    test='train.jsonl',
                                                                       fields={'sentence1': ('premise', text_field),
                                                                               'sentence2': ('hypothesis', text_field),
                                                                               # 'sentence1_binary_parse': ('premise_parse', text_field),
                                                                               # 'sentence2_binary_parse': ('hypothesis_parse', text_field),
                                                                               'gold_label': ('label', label_field)},
                                                                       filter_pred=lambda ex: ex.label != USELESS_LABEL)

        return train, dev, test

# if __name__ == '__main__':
#     NLIDataloader('../../data/multinli_1.0/', '../../data/snli_1.0/', 'glove.840B.300d')

