# Preparing Sentence-BERT embeddings from original text data

Once the `.text` files have been computed for a given dataset, `.sbert` files can be computed following the
steps of this Python code:

```python
from . import write_sbert_file

# -- Global variables
SUBSET="name_of_the_subset"  # 'train', 'test', 'dev', etc.
DATASET_FOLDER="path/to/your/dataset/folder"  # folder containing 'train.text', 'test.text' and 'dev.text' files
PRETRAINED_SBERT_MODEL = "all-MiniLM-L6-v2"

print("Computing sentences embeddings from prepared dataset sentences using...")
print(f"Using pre-trained model: '{PRETRAINED_SBERT_MODEL}'")

# -- Computing and writing sentence-Transformer embeddings of DATASET/SUBSET.text sentences to DATASET/SUBSET.sbert
write_sbert_file(
    DATASET_FOLDER,
    SUBSET,
    sentence_transformer_pretrained_model=PRETRAINED_SBERT_MODEL,
    batch_size=128,
    show_progress_bar=True,
)

print(f"Saved sentences embeddings at: {DATASET_FOLDER}/{SUBSET}.sbert")
```

Here the default pretrained sentence-Transformer model used is
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2.
Note that any other pretrained model can be used by simply providing the corresponding valid name to
instantiate a `SentenceTransformer` object from `sentence_trasnformers` library. 
The list of valid pretrained models names can be found
at https://www.sbert.net/docs/sentence_transformer/pretrained_models.html.