"""SpeakerDB nearest vs centroid identity matching."""
from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if "chromadb" not in sys.modules:
    sys.modules["chromadb"] = MagicMock()
    sys.modules["chromadb.config"] = MagicMock()

from config import parse_speaker_match_mode
from database.speakers import (
    SpeakerDB,
    embedding_cosine_similarity,
    speaker_centroid,
)


def _unit(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / float(np.linalg.norm(a))


class FakeCollection:
    """Minimal Chroma-like store for cosine query / get / add / delete."""

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._metadatas: List[Dict[str, Any]] = []

    def _matches(self, where: Optional[dict], i: int) -> bool:
        if not where:
            return True
        meta = self._metadatas[i]
        return all(meta.get(k) == v for k, v in where.items())

    def add(self, ids, embeddings, metadatas) -> None:
        for i, chroma_id in enumerate(ids):
            self._ids.append(str(chroma_id))
            self._embeddings.append(np.asarray(embeddings[i], dtype=np.float32))
            self._metadatas.append(dict(metadatas[i]))

    def count(self) -> int:
        return len(self._ids)

    def get(self, where=None, ids=None, include=None):
        out_ids: List[str] = []
        out_emb: List[np.ndarray] = []
        out_meta: List[Dict[str, Any]] = []
        want = set(ids) if ids is not None else None
        for i, chroma_id in enumerate(self._ids):
            if want is not None and chroma_id not in want:
                continue
            if not self._matches(where, i):
                continue
            out_ids.append(chroma_id)
            out_emb.append(self._embeddings[i])
            out_meta.append(self._metadatas[i])
        result: Dict[str, Any] = {"ids": out_ids, "metadatas": out_meta}
        if include is None or "embeddings" in include:
            result["embeddings"] = out_emb
        return result

    def query(self, query_embeddings, n_results, include=None):
        query = np.asarray(query_embeddings[0], dtype=np.float32)
        scored = []
        for i, emb in enumerate(self._embeddings):
            sim = embedding_cosine_similarity(query, emb)
            scored.append((1.0 - sim, i))
        scored.sort(key=lambda t: t[0])
        scored = scored[: max(0, int(n_results))]
        ids = [self._ids[i] for _, i in scored]
        metas = [self._metadatas[i] for _, i in scored]
        dists = [float(d) for d, _ in scored]
        return {"ids": [ids], "metadatas": [metas], "distances": [dists]}

    def delete(self, where=None, ids=None) -> None:
        keep = []
        want = set(ids) if ids is not None else None
        for i, chroma_id in enumerate(self._ids):
            drop = False
            if want is not None and chroma_id in want:
                drop = True
            if where and self._matches(where, i):
                drop = True
            if not drop:
                keep.append(i)
        self._ids = [self._ids[i] for i in keep]
        self._embeddings = [self._embeddings[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]


def _memory_db(match_mode: str = "nearest") -> SpeakerDB:
    db = SpeakerDB.__new__(SpeakerDB)
    db.db_path = ":memory:"
    db.similarity_threshold = 0.7
    db.match_mode = match_mode
    db._centroid_cache = None
    db._enrollment_cache = None
    db.collection = FakeCollection()
    db.client = None
    return db


class TestParseSpeakerMatchMode(unittest.TestCase):
    def test_default_and_aliases(self):
        self.assertEqual(parse_speaker_match_mode(None), "nearest")
        self.assertEqual(parse_speaker_match_mode(""), "nearest")
        self.assertEqual(parse_speaker_match_mode(" nearest "), "nearest")
        self.assertEqual(parse_speaker_match_mode("CENTROID"), "centroid")

    def test_rejects_unknown(self):
        with self.assertRaises(ValueError):
            parse_speaker_match_mode("average")


class TestSpeakerCentroidMath(unittest.TestCase):
    def test_single_vector_is_unit(self):
        c = speaker_centroid([np.array([3.0, 0.0], dtype=np.float32)])
        np.testing.assert_allclose(c, [1.0, 0.0], atol=1e-5)

    def test_mean_of_orthogonal_units(self):
        c = speaker_centroid(
            [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]
        )
        expected = _unit([0.5, 0.5])
        np.testing.assert_allclose(c, expected, atol=1e-5)
        self.assertAlmostEqual(embedding_cosine_similarity([1.0, 0.0], c), expected[0], places=4)


class TestSpeakerDBMatchModes(unittest.TestCase):
    def test_invalid_match_mode(self):
        with self.assertRaises(ValueError):
            SpeakerDB(db_path="/tmp/unused-speakers", match_mode="maxpool")

    def test_single_embedding_same_in_both_modes(self):
        query = _unit([1.0, 0.0, 0.0])
        nearest_db = _memory_db("nearest")
        nearest_db.create_speaker("Solo", query)

        near_spk, near_sim = nearest_db.query_best_match(query)
        self.assertIsNotNone(near_spk)
        self.assertEqual(near_spk.display_name, "Solo")
        self.assertGreater(near_sim, 0.99)

        centroid_db = _memory_db("centroid")
        centroid_db.collection = nearest_db.collection
        cent_spk, cent_sim = centroid_db.query_best_match(query)
        self.assertIsNotNone(cent_spk)
        self.assertEqual(cent_spk.id, near_spk.id)
        self.assertAlmostEqual(cent_sim, near_sim, places=3)

    def test_nearest_and_centroid_disagree_on_outlier_enrollment(self):
        query = _unit([1.0, 0.0])
        alice_close = _unit([1.0, 0.0])
        alice_far = _unit([0.0, 1.0])
        bob = _unit([0.9, 0.2])

        nearest_db = _memory_db("nearest")
        alice_id = nearest_db.create_speaker("Alice", alice_close)
        nearest_db.add_embedding(alice_id, alice_far)
        bob_id = nearest_db.create_speaker("Bob", bob)

        near_spk, near_sim = nearest_db.query_best_match(query)
        self.assertEqual(near_spk.id, alice_id)
        self.assertGreater(near_sim, 0.99)

        centroid_db = _memory_db("centroid")
        centroid_db.collection = nearest_db.collection
        cent_spk, cent_sim = centroid_db.query_best_match(query)
        self.assertEqual(cent_spk.id, bob_id)
        self.assertNotEqual(cent_spk.id, alice_id)

        alice_centroid = speaker_centroid([alice_close, alice_far])
        self.assertGreater(
            embedding_cosine_similarity(query, bob),
            embedding_cosine_similarity(query, alice_centroid),
        )
        self.assertAlmostEqual(
            cent_sim, embedding_cosine_similarity(query, bob), places=3
        )

        hits = nearest_db.query_nearest_embeddings(query, limit=3)
        self.assertGreaterEqual(len(hits), 1)
        alice_hits = [h for h in hits if h["speaker_id"] == alice_id]
        self.assertTrue(alice_hits)
        self.assertAlmostEqual(
            alice_hits[0]["centroid_similarity"],
            embedding_cosine_similarity(query, alice_centroid),
            places=3,
        )

    def test_centroid_cache_invalidates_after_add_embedding(self):
        query = _unit([1.0, 0.0])
        db = _memory_db("centroid")
        sid = db.create_speaker("Alice", _unit([1.0, 0.0]))
        _, sim_one = db.query_best_match(query)
        self.assertGreater(sim_one, 0.99)
        db.add_embedding(sid, _unit([0.0, 1.0]))
        _, sim_two = db.query_best_match(query)
        self.assertLess(sim_two, sim_one)
        self.assertAlmostEqual(
            sim_two,
            embedding_cosine_similarity(query, speaker_centroid([query, _unit([0.0, 1.0])])),
            places=3,
        )

    def test_nearest_survives_hnsw_query_failure(self):
        query = _unit([1.0, 0.0])
        db = _memory_db("nearest")
        sid = db.create_speaker("Alice", query)

        def boom(*_args, **_kwargs):
            raise RuntimeError(
                "Error executing plan: Internal error: Error creating hnsw "
                "segment reader: Nothing found on disk"
            )

        db.collection.query = boom  # type: ignore[method-assign]
        spk, sim = db.query_best_match(query)
        self.assertIsNotNone(spk)
        self.assertEqual(spk.id, sid)
        self.assertGreater(sim, 0.99)

    def test_refresh_reloads_enrollments_added_elsewhere(self):
        query = _unit([1.0, 0.0])
        db = _memory_db("nearest")
        spk, sim = db.query_best_match(query)
        self.assertIsNone(spk)
        self.assertEqual(sim, 0.0)

        other = _memory_db("nearest")
        other.collection = db.collection
        sid = other.create_speaker("Alice", query)
        # Writer invalidates its own cache; the reader still has an empty cache.
        spk_stale, _ = db.query_best_match(query)
        self.assertIsNone(spk_stale)

        db.refresh()
        spk_fresh, sim_fresh = db.query_best_match(query)
        self.assertIsNotNone(spk_fresh)
        self.assertEqual(spk_fresh.id, sid)
        self.assertGreater(sim_fresh, 0.99)


if __name__ == "__main__":
    unittest.main()
