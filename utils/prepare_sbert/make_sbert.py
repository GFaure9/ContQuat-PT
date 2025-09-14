import io
from typing import List
from tqdm import tqdm

from . import compute_sentences_embeddings


def load_sentences(dataset_folder: str, subset: str) -> List[str]:
    sentences_path = f"{dataset_folder}/{subset}.text"

    with io.open(sentences_path, mode="r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    return sentences


def write_sbert_file(
        dataset_folder: str,
        subset: str,
        sentence_transformer_pretrained_model,
        batch_size: int = 64,
        show_progress_bar: bool = True,
):
    """
    Write dataset's `subset`.sbert file.
    Each line is a sentence embedding obtained with given sBERT pre-trained model.
    Line {k} in the file matches line {k} in `subset`.text file (and the other parallel files).
    """
    sentences = load_sentences(dataset_folder, subset)

    # --- compute sentences embeddings
    sentences_embeddings = compute_sentences_embeddings(
        sentences,
        pretrained_model=sentence_transformer_pretrained_model,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    # --- write embeddings to .sbert file
    with open(f"{dataset_folder}/{subset}.sbert", "w", encoding="utf-8") as f:
        for emb in tqdm(sentences_embeddings):
            f.write(" ".join(map(str, emb.tolist())) + "\n")
