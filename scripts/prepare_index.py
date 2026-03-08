import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
from fastembed import TextEmbedding
from tqdm import tqdm


ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.datasets import load_files
from sklearn.datasets._twenty_newsgroups import (
    strip_newsgroup_header,
    strip_newsgroup_quoting,
    strip_newsgroup_footer,
)

def load_corpus():
    """
    Load 20 Newsgroups from local tar.gz extraction and strip headers/footers/quotes,
    which are mostly noise for semantic content.
    """
    data_home = Path(__file__).resolve().parent.parent / "dataset" / "twenty+newsgroups (1)" / "20_newsgroups"
    dataset = load_files(str(data_home), encoding="latin1")

    texts = []
    for text in dataset.data:
        text = strip_newsgroup_header(text)
        text = strip_newsgroup_quoting(text)
        text = strip_newsgroup_footer(text)
        texts.append(text)

    targets = list(dataset.target)
    target_names = dataset.target_names

    return texts, targets, target_names


def basic_cleanup(text: str) -> str:
    text = " ".join(text.strip().split())
    return text


def embed_corpus(texts):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = TextEmbedding(model_name)
    
    embeddings = []
    print("Generating embeddings via fastembed...")
    # fastembed yields embeddings iteratively
    for emb in model.embed(texts, batch_size=64):
        embeddings.append(emb)
    
    embeddings = np.array(embeddings).astype("float32")
    return embeddings, model_name


def choose_gmm_components(embeddings, candidate_ks=(10, 20, 30)):
    rng = np.random.default_rng(42)
    if embeddings.shape[0] > 5000:
        idx = rng.choice(embeddings.shape[0], size=5000, replace=False)
        subset = embeddings[idx]
    else:
        subset = embeddings

    best_k = None
    best_bic = np.inf
    for k in candidate_ks:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            random_state=42,
            max_iter=200,
        )
        gmm.fit(subset)
        bic = gmm.bic(subset)
        print(f"GMM K={k}, BIC={bic:.2f}")
        if bic < best_bic:
            best_bic = bic
            best_k = k

    print(f"Chosen number of clusters (by BIC): K={best_k}")
    return best_k


def fit_gmm(embeddings, n_components: int) -> GaussianMixture:
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="diag",
        random_state=42,
        max_iter=300,
    )
    gmm.fit(embeddings)
    return gmm


def main():
    print("Loading 20 Newsgroups corpus...")
    texts, targets, target_names = load_corpus()

    print(f"Loaded {len(texts)} documents.")
    cleaned_texts = [basic_cleanup(t) for t in texts]

    doc_records = []
    filtered_texts = []
    filtered_targets = []

    for i, (txt, y) in enumerate(zip(cleaned_texts, targets)):
        if not txt:
            continue
        doc_records.append(
            {
                "id": len(doc_records),
                "original_index": i,
                "label": int(y),
                "label_name": target_names[y],
            }
        )
        filtered_texts.append(txt)
        filtered_targets.append(y)

    print(f"Kept {len(filtered_texts)} non-empty documents after cleanup.")

    print("Embedding documents...")
    embeddings, model_name = embed_corpus(filtered_texts)

    print("Selecting number of clusters by BIC...")
    best_k = choose_gmm_components(embeddings)

    print(f"Fitting GMM with K={best_k} on full embeddings...")
    gmm = fit_gmm(embeddings, n_components=best_k)

    print("Computing per-document cluster distributions...")
    cluster_probs = gmm.predict_proba(embeddings).astype("float32")

    print("Saving artifacts to disk...")

    texts_path = ARTIFACT_DIR / "doc_texts.jsonl"
    with texts_path.open("w", encoding="utf-8") as f:
        for rec, txt in zip(doc_records, filtered_texts):
            rec_with_text = {**rec, "text": txt}
            f.write(json.dumps(rec_with_text) + "\n")

    np.save(ARTIFACT_DIR / "doc_embeddings.npy", embeddings)

    with open(ARTIFACT_DIR / "gmm.pkl", "wb") as f:
        pickle.dump(gmm, f)

    np.save(ARTIFACT_DIR / "doc_cluster_probs.npy", cluster_probs)

    config = {
        "embedding_model": model_name,
        "embedding_dim": int(embeddings.shape[1]),
        "num_documents": int(embeddings.shape[0]),
        "num_clusters": int(gmm.n_components),
        "target_names": target_names,
    }
    with open(ARTIFACT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Done. Artifacts written to:", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
