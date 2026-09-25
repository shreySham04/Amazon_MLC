import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC))
import blocking


class FakeSentenceTransformer:
    def __init__(self, name, device=None):
        self.device = device

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            value = float(sum(ord(char) for char in text) % 101)
            vector = np.array([value, len(text), 1.0], dtype=np.float32)
            vector /= max(np.linalg.norm(vector), 1.0)
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


def sample_frame(rows):
    return pd.DataFrame(rows, columns=[
        "entity_id", "country", "name_core", "name_norm", "addr_norm",
        "postal_code",
    ])


def test_memory_safe_blockers_match_union_and_write_chunks():
    original_model = blocking.SentenceTransformer
    blocking.SentenceTransformer = FakeSentenceTransformer
    split_name = "sample_validation"
    chunk_dir = Path(blocking._CACHE_DIR) / "candidate_chunks" / split_name
    shutil.rmtree(chunk_dir, ignore_errors=True)

    s1 = sample_frame([
        ("s1-a", "us", "acme labs", "acme labs", "10 main springfield", "10001"),
        ("s1-b", "us", "rare comet", "rare comet", "20 oak springfield", ""),
    ])
    s2 = sample_frame([
        ("s2-a", "us", "acme labs", "acme labs", "99 main springfield", "10001"),
        ("s2-b", "us", "rare comet", "rare comet", "20 oak springfield", ""),
        ("s2-c", "us", "unrelated", "unrelated", "4 pine springfield", ""),
    ])
    s3 = sample_frame([
        ("s3-a", "us", "acme labs", "acme labs", "7 main springfield", "10001"),
    ])
    s1 = pd.concat([s1, sample_frame([
        (f"s1-fill-{i}", "us", f"filler s1 {i}", f"filler s1 {i}",
         f"{i + 100} oak springfield", "")
        for i in range(100)
    ])], ignore_index=True)
    s2 = pd.concat([s2, sample_frame([
        (f"s2-fill-{i}", "us", f"filler s2 {i}", f"filler s2 {i}",
         f"{i + 300} oak springfield", "")
        for i in range(100)
    ])], ignore_index=True)
    candidates = pd.concat([s2, s3], ignore_index=True)

    expected = set()
    blocker_counts = {}
    generators = [
        ("postal", blocking._iter_postal_pairs),
        ("rare", blocking._iter_rare_token_pairs),
        ("tfidf", blocking._iter_tfidf_pairs),
        ("city", blocking._iter_city_pairs),
    ]
    for name, generator in generators:
        local_pairs = list(generator(s1, candidates))
        blocker_counts[name] = len(local_pairs)
        expected.update((s1.iloc[i].entity_id, candidates.iloc[j].entity_id)
                        for i, j in local_pairs)

    model = FakeSentenceTransformer("sample", device="cuda:0")
    embedding_pairs = list(blocking._iter_embedding_pairs(s1, candidates, model,
                                                            s1_cache=None, cand_cache=None))
    blocker_counts["embedding"] = len(embedding_pairs)
    expected.update((s1.iloc[i].entity_id, candidates.iloc[j].entity_id)
                    for i, j in embedding_pairs)

    actual_df = blocking.generate_candidates(s1, s2, s3, use_embeddings=True,
                                              split_name=split_name)
    actual = set(zip(actual_df.source1_entity_id, actual_df.candidate_entity_id))

    try:
        assert all(count > 0 for count in blocker_counts.values())
        assert actual == expected
        assert len(actual) == len(actual_df)
        assert list(chunk_dir.glob("*.parquet"))
    finally:
        blocking.SentenceTransformer = original_model


if __name__ == "__main__":
    test_memory_safe_blockers_match_union_and_write_chunks()
    print("sample blocking validation passed")