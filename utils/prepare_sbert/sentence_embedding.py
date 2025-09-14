import torch
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


def load_pretrained_sentence_embeddings_model(name: str = None):
    if name is not None:
        return SentenceTransformer(name)
    else:
        return None


def compute_sentences_embeddings(
        sentences: List[str],
        pretrained_model: Union[str, SentenceTransformer] = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress_bar: bool = False,
) -> Union[torch.Tensor, np.ndarray]:

    # --- loading pre-train model
    # (see https://www.sbert.net/docs/sentence_transformer/pretrained_models.html for list of available models)
    if isinstance(pretrained_model, str):
        model = SentenceTransformer(pretrained_model)
    else:
        model = pretrained_model

    # -- calculate embeddings by calling model.encode()
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )  # shape=(N_sentences, dim_emb)
    # print(embeddings.shape)

    return embeddings


if __name__ == "__main__":
    print("Start test...\n\n")

    import time

    my_sentences = [
        "Cat and dogs are animals.",
        "I like playing with my cat but my dog is tired...",
        "What is your favorite animal?",
        "The sky is red!",
    ]

    start = time.time()
    my_embeddings = compute_sentences_embeddings(my_sentences)

    print(f"Computation time: {time.time() - start:.3f} seconds")
    print(50 * "-")
    print(f"Embeddings [shape={my_embeddings.shape}]:\n{my_embeddings}")

    # -> little check by looking at the cosine similarity matrix from computed embeddings
    import torch.nn.functional as F

    norm_emb = F.normalize(torch.from_numpy(my_embeddings), p=2, dim=1)  # normalize embeddings
    cosine_sim_matrix = norm_emb @ norm_emb.T
    print(50 * "-")
    print(f"Cosine similarity matrix:\n{cosine_sim_matrix}")
