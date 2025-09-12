"""
Data module.
Adapted from original code at https://github.com/BenSaunders27/ProgressiveTransformersSLP
"""

import sys
import os
import io
import torch
import os.path
from typing import Optional
from torchtext import data
from torchtext.data import Dataset, Iterator, Field

from .constants import UNK_TOKEN, PAD_TOKEN, TARGET_PAD
from .vocabulary import build_vocab, Vocabulary


# -------- Important information --------
# Load the Regression Data
# Data format should be parallel .txt files for src, trg and files
# Each line of the .txt file represents a new sequence, in the same order in each file
# src file should contain a new source input on each line
# trg file should contain skeleton data, with each line a new sequence, each frame following on from the previous
# Joint values were divided by 3 to move to the scale of -1 to 1
# Each joint value should be separated by a space " "
# Each frame is partitioned using the known trg_size length, which includes all joints (In 2D or 3D) and the counter
# Files file should contain the name of each sequence on a new line
# ---------------------------------------

def load_data(cfg: dict) -> (Dataset, Dataset, Optional[Dataset], Vocabulary, Vocabulary):
    """
    Load train, dev and optionally test data as specified in configuration.
    Vocabularies are created from the training set with a limit of `voc_limit`
    tokens and a minimum token frequency of `voc_min_freq`
    (specified in the configuration dictionary).

    The training data is filtered to include sentences up to `max_sent_length`
    on source and target side.

    :param data_cfg: configuration dictionary for data
        ("data" part of configuation file)
    :return:
        - train_data: training dataset
        - dev_data: development dataset
        - test_data: testdata set if given, otherwise None
        - src_vocab: source vocabulary extracted from training data
        - trg_vocab: target vocabulary extracted from training data
    """
    data_cfg = cfg["data"]
    # Source, Target and Files postfixes
    src_lang = data_cfg["src"]
    trg_lang = data_cfg["trg"]
    files_lang = data_cfg.get("files", "files")
    sentences_emb_lang = data_cfg.get("sentences_embeddings", None)
    # Train, Dev and Test Path
    train_path = data_cfg["train"]
    dev_path = data_cfg["dev"]
    test_path = data_cfg["test"]

    level = "word"
    lowercase = False
    max_sent_length = data_cfg["max_sent_length"]
    # Target size is plus one due to the counter required for the model
    trg_size = cfg["model"]["trg_size"] + 1
    # Skip frames is used to skip a set proportion of target frames, to simplify the model requirements
    skip_frames = data_cfg.get("skip_frames", 1)

    EOS_TOKEN = '</s>'
    tok_fun = lambda s: list(s) if level == "char" else s.split()

    # Source field is a tokenised version of the source words
    src_field = data.Field(
        init_token=None,
        eos_token=EOS_TOKEN,
        pad_token=PAD_TOKEN,
        tokenize=tok_fun,
        batch_first=True,
        lower=lowercase,
        unk_token=UNK_TOKEN,
        include_lengths=True
    )

    # Files field is just a raw text field
    files_field = data.RawField()

    def tokenize_features(features):
        features = torch.as_tensor(features)
        ft_list = torch.split(features, 1, dim=0)
        return [ft.squeeze() for ft in ft_list]

    def stack_features(features, something):
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)

    # Creating a regression target field
    # Pad token is a vector of output size, containing the constant TARGET_PAD
    reg_trg_field = data.Field(
        sequential=True,
        use_vocab=False,
        dtype=torch.float32,
        batch_first=True,
        include_lengths=False,
        pad_token=torch.ones((trg_size,))*TARGET_PAD,
        preprocessing=tokenize_features,
        postprocessing=stack_features,
    )

    # Field for sentence embeddings - added by me (GF)
    def parse_sentence_embedding_line(s):
        return list(map(float, s.strip().split()))

    sent_emb_field = data.Field(
        sequential=False,
        use_vocab=False,
        dtype=torch.float32,
        preprocessing=parse_sentence_embedding_line,
        batch_first=True,
    )

    # Defining extensions / fields - added by me (GF) to allow for loading .sbert (or any other sentences embeddings file) if wanted
    if sentences_emb_lang is None:
            extensions = ("." + src_lang, "." + trg_lang, "." + files_lang)
            fields = (src_field, reg_trg_field, files_field)
    else:
        extensions = ("." + src_lang, "." + trg_lang, "." + files_lang, "." + sentences_emb_lang)
        fields = (src_field, reg_trg_field, files_field, sent_emb_field)

    # Create the Training Data, using the SignProdDataset
    train_data = SignProdDataset(
        path=train_path,
        exts=extensions,
        fields=fields,
        trg_size=trg_size,
        skip_frames=skip_frames,
        filter_pred=
        lambda x: len(vars(x)['src'])
        <= max_sent_length
        and len(vars(x)['trg'])
        <= max_sent_length
    )

    src_max_size = data_cfg.get("src_voc_limit", sys.maxsize)
    src_min_freq = data_cfg.get("src_voc_min_freq", 1)
    src_vocab_file = data_cfg.get("src_vocab", None)
    src_vocab = build_vocab(
        field="src",
        dataset=train_data,
        vocab_file=src_vocab_file,
        min_freq=src_min_freq, max_size=src_max_size,
    )

    # Create a target vocab just as big as the required target vector size -
    # So that len(trg_vocab) is # of joints + 1 (for the counter)
    trg_vocab = [None]*trg_size

    # Create the Validation Data
    dev_data = SignProdDataset(
        path=dev_path,
        exts=extensions,
        trg_size=trg_size,
        fields=fields,
        skip_frames=skip_frames
    )

    # Create the Testing Data
    test_data = SignProdDataset(
        path=test_path,
        exts=extensions,
        trg_size=trg_size,
        fields=fields,
        skip_frames=skip_frames
    )

    src_field.vocab = src_vocab

    return train_data, dev_data, test_data, src_vocab, trg_vocab


global max_src_in_batch, max_tgt_in_batch

def token_batch_size_fn(new, count, sofar):
    """Compute batch size based on number of tokens (+padding)."""
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    if hasattr(new, 'trg'):  # for monolingual data sets ("translate" mode)
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        tgt_elements = count * max_tgt_in_batch
    else:
        tgt_elements = 0
    return max(src_elements, tgt_elements)


def make_data_iter(dataset: Dataset,
                   batch_size: int,
                   batch_type: str = "sentence",
                   train: bool = False,
                   shuffle: bool = False) -> Iterator:
    """
    Returns a torchtext iterator for a torchtext dataset.

    :param dataset: torchtext dataset containing src and optionally trg
    :param batch_size: size of the batches the iterator prepares
    :param batch_type: measure batch size by sentence count or by token count
    :param train: whether it's training time, when turned off,
                  bucketing, sorting within batches and shuffling is disabled
    :param shuffle: whether to shuffle the data before each epoch
                    (no effect if set to True for testing)
    :return: torchtext iterator
    """

    batch_size_fn = token_batch_size_fn if batch_type == "token" else None

    if train:
        # optionally shuffle and sort during training
        data_iter = data.BucketIterator(
            repeat=False, sort=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=True, sort_within_batch=True,
            sort_key=lambda x: len(x.src), shuffle=shuffle)
    else:
        # don't sort/shuffle for validation/inference
        data_iter = data.BucketIterator(
            repeat=False, dataset=dataset,
            batch_size=batch_size, batch_size_fn=batch_size_fn,
            train=False, sort=False)

    return data_iter


# Main Dataset Class

class SignProdDataset(data.Dataset):
    def __init__(self, path, exts, fields, trg_size, skip_frames=1, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            new_fields = [
                ('src', fields[0]),
                ('trg', fields[1]),
                ('file_paths', fields[2])
            ]
            if len(exts) == 4:
                new_fields.append(('sent_emb', fields[3]))  # sentences embeddings field
            fields = new_fields

        paths = [os.path.expanduser(path + x) for x in exts]
        src_path, trg_path, file_path = paths[:3]
        emb_path = paths[3] if len(paths) == 4 else None

        if emb_path:
            with io.open(emb_path, mode='r', encoding='utf-8') as emb_file:
                emb_lines = [line.strip() for line in emb_file]
        else:
            emb_lines = None

        examples = []
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
             io.open(trg_path, mode='r', encoding='utf-8') as trg_file, \
             io.open(file_path, mode='r', encoding='utf-8') as files_file:

            if emb_lines:
                loop = zip(src_file, trg_file, files_file, emb_lines)
            else:
                loop = zip(src_file, trg_file, files_file)

            for i, lines in enumerate(loop):
                if emb_lines:
                    src_line, trg_line, files_line, emb_line = lines
                else:
                    src_line, trg_line, files_line = lines

                src_line = src_line.strip()
                trg_line = trg_line.strip()
                files_line = files_line.strip()
                if not src_line or not trg_line:
                    continue

                trg_vals = [(float(j) + 1e-8) for j in trg_line.split(" ")]
                trg_frames = [trg_vals[i:i + trg_size] for i in range(0, len(trg_vals), trg_size * skip_frames)]

                data_list = [src_line, trg_frames, files_line]
                if emb_lines:
                    data_list.append(emb_line)

                examples.append(data.Example.fromlist(data_list, fields))

        super(SignProdDataset, self).__init__(examples, fields, **kwargs)
