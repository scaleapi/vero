"""Embedding computation for diff analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI

from .config import EMBEDDINGS_CACHE_DIR

# Token limit for text-embedding-3-large (with buffer)
MAX_TOKENS = 8192
EMBEDDING_MODEL = "text-embedding-3-large"

# Tokenizer for the embedding model (cl100k_base is used by text-embedding-3-*)
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(_tokenizer.encode(text))


def _chunk_text(text: str, max_tokens: int = MAX_TOKENS) -> list[str]:
    """Split text into chunks that fit within token limit."""
    tokens = _tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(_tokenizer.decode(chunk_tokens))

    return chunks


def _get_embeddings(
    client: OpenAI, texts: list[str], model: str = EMBEDDING_MODEL
) -> list[list[float]]:
    """Get embeddings for multiple texts in one API call."""
    response = client.embeddings.create(input=texts, model=model)
    # Response order matches input order
    return [item.embedding for item in response.data]


def _get_embeddings_chunked(
    client: OpenAI, texts: list[str], model: str = EMBEDDING_MODEL
) -> list[list[float]]:
    """Get embeddings, chunking and averaging large texts that exceed token limit."""
    results: list[list[float] | None] = [None] * len(texts)
    small_texts: list[str] = []
    small_indices: list[int] = []
    large_items: list[tuple[int, str]] = []

    # Separate by size
    for i, text in enumerate(texts):
        if _count_tokens(text) <= MAX_TOKENS:
            small_texts.append(text)
            small_indices.append(i)
        else:
            large_items.append((i, text))

    # Batch embed small texts (one API call)
    if small_texts:
        embeddings = _get_embeddings(client, small_texts, model)
        for i, emb in zip(small_indices, embeddings):
            results[i] = emb

    # Chunk, embed, average for large texts
    for i, text in large_items:
        chunks = _chunk_text(text)
        chunk_embeddings = _get_embeddings(client, chunks, model)
        avg_embedding = np.mean(chunk_embeddings, axis=0).tolist()
        results[i] = avg_embedding

    return results  # type: ignore


async def compute_cumulative_diff_embeddings(
    session_id: str,
    project_path: Path | str,
    file_patterns: list[str] | None = None,
    chunk_large_diffs: bool = True,
) -> dict[int, list[float]]:
    """Compute embeddings for cumulative diffs (base → each commit).

    For each phase, computes the diff from the base commit to the final
    commit of that phase, then embeds the diff.

    Args:
        session_id: Session UUID
        project_path: Path to the git repository
        file_patterns: Optional glob patterns to filter files (e.g., ["*.py", "*.jinja"])
            If None, includes all files.
        chunk_large_diffs: If True, chunk large diffs and average embeddings.
            If False, raise error for diffs exceeding token limit.

    Returns:
        Dict mapping phase_index -> embedding vector
        Phase 0 uses a placeholder "no changes" embedding.
    """
    import subprocess as _sp

    from vero.traces.analysis.collator import TraceAnalysisPayload

    project_path = Path(project_path)
    payload = await TraceAnalysisPayload.from_session_id(session_id, project_path=project_path)

    base_commit = payload.config.base_commit
    client = OpenAI()

    # Collect all diffs first
    phase_indices = []
    diff_texts = []

    for phase_idx, phase in enumerate(payload.phases):
        final_commit = phase.final_commit.commit

        # Build diff command with optional file filtering
        diff_args = [base_commit, final_commit]
        if file_patterns:
            diff_args.append("--")
            diff_args.extend(file_patterns)

        diff_text = _sp.run(
            ["git", "diff", *diff_args], cwd=project_path,
            capture_output=True, text=True, check=True,
        ).stdout

        # Empty diff (e.g., phase 0 where base==final) -> use single space
        # OpenAI rejects empty strings but accepts " "
        if not diff_text.strip():
            diff_text = " "

        phase_indices.append(phase_idx)
        diff_texts.append(diff_text)

    # Embed diffs
    if chunk_large_diffs:
        embedding_vectors = _get_embeddings_chunked(client, diff_texts)
    else:
        embedding_vectors = _get_embeddings(client, diff_texts)

    return dict(zip(phase_indices, embedding_vectors))


def _get_cache_path(session_id: str, suffix: str = "") -> Path:
    """Get cache file path for session embeddings."""
    return EMBEDDINGS_CACHE_DIR / f"{session_id}_cumulative{suffix}.json"


def load_or_compute_embeddings(
    session_ids: list[str],
    project_path: Path | str,
    file_patterns: list[str] | None = None,
    chunk_large_diffs: bool = True,
    use_cache: bool = True,
    save_cache: bool = True,
) -> dict[str, dict[int, list[float]]]:
    """Load cached embeddings or compute if missing.

    Args:
        session_ids: List of session UUIDs
        project_path: Path to the git repository
        file_patterns: Optional glob patterns to filter files
        chunk_large_diffs: If True, chunk large diffs and average embeddings
        use_cache: Whether to load from cache
        save_cache: Whether to save computed embeddings to cache

    Returns:
        Dict mapping session_id -> (phase_index -> embedding)
    """
    project_path = Path(project_path)
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Cache suffix based on file patterns
    suffix = "_filtered" if file_patterns else ""

    all_embeddings = {}

    for i, session_id in enumerate(session_ids):
        cache_path = _get_cache_path(session_id, suffix)

        # Try loading from cache
        if use_cache and cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                # Convert string keys back to int
                all_embeddings[session_id] = {int(k): v for k, v in cached.items()}
                continue
            except (json.JSONDecodeError, KeyError):
                pass

        # Compute embeddings
        print(f"Computing embeddings for session {i+1}/{len(session_ids)}: {session_id[:8]}...")
        try:
            embeddings = compute_cumulative_diff_embeddings(
                session_id,
                project_path,
                file_patterns=file_patterns,
                chunk_large_diffs=chunk_large_diffs,
            )
            all_embeddings[session_id] = embeddings

            # Save to cache
            if save_cache:
                with open(cache_path, "w") as f:
                    json.dump(embeddings, f)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return all_embeddings


def reduce_embeddings(
    embeddings: dict[str, dict[int, list[float]]],
    method: Literal["umap", "tsne"] = "umap",
    random_state: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """Reduce embeddings to 2D coordinates using UMAP or t-SNE.

    Args:
        embeddings: Dict mapping session_id -> (phase_index -> embedding)
        method: "umap" or "tsne"
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for the reducer
            UMAP: n_neighbors (default 15), min_dist (default 0.1)
            t-SNE: perplexity (default 30), learning_rate (default 200)

    Returns:
        DataFrame with columns: [session_id, phase, x, y]
    """
    # Flatten embeddings to array
    records = []
    vectors = []

    for session_id, phase_embeddings in embeddings.items():
        for phase, embedding in phase_embeddings.items():
            records.append({"session_id": session_id, "phase": phase})
            vectors.append(embedding)

    if not vectors:
        return pd.DataFrame(columns=["session_id", "phase", "x", "y"])

    vectors_array = np.array(vectors)

    if method == "umap":
        from umap import UMAP

        n_neighbors = kwargs.get("n_neighbors", 15)
        min_dist = kwargs.get("min_dist", 0.1)
        reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
        coords = reducer.fit_transform(vectors_array)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = kwargs.get("perplexity", 30)
        learning_rate = kwargs.get("learning_rate", 200)
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=random_state,
            init="pca",
        )
        coords = reducer.fit_transform(vectors_array)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build result DataFrame
    df = pd.DataFrame(records)
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    return df


def cluster_final_embeddings(
    embeddings: dict[str, dict[int, list[float]]],
    method: Literal["kmeans", "dbscan"] = "kmeans",
    n_clusters: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """Cluster the final phase embeddings.

    Args:
        embeddings: Dict mapping session_id -> (phase_index -> embedding)
        method: "kmeans" or "dbscan"
        n_clusters: Number of clusters (for kmeans)
        **kwargs: Additional parameters for the clustering algorithm
            DBSCAN: eps (default 0.5), min_samples (default 5)

    Returns:
        DataFrame with columns: [session_id, final_phase, cluster, x, y]
        x, y are 2D UMAP coordinates for visualization
    """
    from sklearn.cluster import DBSCAN, KMeans
    from umap import UMAP

    # Extract final phase embedding for each session
    records = []
    vectors = []

    for session_id, phase_embeddings in embeddings.items():
        if not phase_embeddings:
            continue
        final_phase = max(phase_embeddings.keys())
        records.append({"session_id": session_id, "final_phase": final_phase})
        vectors.append(phase_embeddings[final_phase])

    if not vectors:
        return pd.DataFrame(columns=["session_id", "final_phase", "cluster", "x", "y"])

    vectors_array = np.array(vectors)

    # Cluster
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(vectors_array)
    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(vectors_array)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Reduce to 2D for visualization
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords = reducer.fit_transform(vectors_array)

    # Build result DataFrame
    df = pd.DataFrame(records)
    df["cluster"] = labels
    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    return df


# Alias for backwards compatibility
def reduce_to_umap(
    embeddings: dict[str, dict[int, list[float]]],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Reduce embeddings to 2D UMAP coordinates. Alias for reduce_embeddings(method='umap')."""
    return reduce_embeddings(
        embeddings,
        method="umap",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
