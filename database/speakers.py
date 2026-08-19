# Speaker database operations using ChromaDB
import logging
import uuid
from typing import Any, Optional, Dict, List, Tuple
from datetime import datetime
import numpy as np
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

VALID_MATCH_MODES = frozenset({"nearest", "centroid"})


def embedding_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; returns -1.0 if vectors are invalid or length-mismatched."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return -1.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def speaker_centroid(embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
    """L2-normalize each vector, mean, then L2-normalize the mean. None if empty/degenerate."""
    unit: List[np.ndarray] = []
    for raw in embeddings:
        v = np.asarray(raw, dtype=np.float32).ravel()
        n = float(np.linalg.norm(v))
        if n == 0.0:
            continue
        unit.append(v / n)
    if not unit:
        return None
    mean = np.mean(np.stack(unit, axis=0), axis=0)
    mn = float(np.linalg.norm(mean))
    if mn == 0.0:
        return mean.astype(np.float32)
    return (mean / mn).astype(np.float32)


class Speaker:
    """Represents a speaker with metadata"""
    def __init__(self, id: str, display_name: str, created_at: str, embedding_count: int):
        self.id = id
        self.display_name = display_name
        self.created_at = created_at
        self.embedding_count = embedding_count

class SpeakerDB:
    """Database manager for speaker embeddings using ChromaDB"""
    
    def __init__(
        self,
        db_path: str,
        similarity_threshold: float = 0.7,
        match_mode: str = "nearest",
    ):
        """
        Initialize the speaker database
        
        Args:
            db_path: Path to the ChromaDB database
            similarity_threshold: Cosine similarity threshold for matching (default: 0.7)
            match_mode: ``nearest`` (best enrollment) or ``centroid`` (mean of enrollments)
        """
        mode = (match_mode or "nearest").strip().lower() or "nearest"
        if mode not in VALID_MATCH_MODES:
            raise ValueError(
                f"match_mode={match_mode!r} is invalid; use 'nearest' or 'centroid'"
            )
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.match_mode = mode
        self._centroid_cache: Optional[Dict[str, Dict[str, Any]]] = None

        # Initialize ChromaDB client
        try:
            self.client = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create the speakers collection
            self.collection = self.client.get_or_create_collection(
                name="speakers",
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(
                "Initialized speaker database at %s (match_mode=%s)",
                db_path,
                self.match_mode,
            )
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise

    def _invalidate_centroid_cache(self) -> None:
        self._centroid_cache = None

    def _speaker_centroids(self) -> Dict[str, Dict[str, Any]]:
        """Map speaker_id -> display_name, created_at, embedding_count, centroid."""
        if self._centroid_cache is not None:
            return self._centroid_cache

        results = self.collection.get(include=["embeddings", "metadatas"])
        ids = results.get("ids") or []
        embeddings = results.get("embeddings")
        metadatas = results.get("metadatas") or []
        grouped: Dict[str, Dict[str, Any]] = {}
        if embeddings is None or len(ids) == 0:
            self._centroid_cache = {}
            return self._centroid_cache

        for i, _chroma_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else None
            if not meta:
                continue
            speaker_id = str(meta.get("speaker_id", "") or "")
            if not speaker_id:
                continue
            vec = embeddings[i]
            if vec is None:
                continue
            bucket = grouped.setdefault(
                speaker_id,
                {
                    "display_name": str(meta.get("display_name", "") or ""),
                    "created_at": str(meta.get("created_at", "") or ""),
                    "vectors": [],
                },
            )
            bucket["vectors"].append(np.asarray(vec, dtype=np.float32))

        cache: Dict[str, Dict[str, Any]] = {}
        for speaker_id, info in grouped.items():
            centroid = speaker_centroid(info["vectors"])
            if centroid is None:
                continue
            cache[speaker_id] = {
                "display_name": info["display_name"],
                "created_at": info["created_at"],
                "embedding_count": len(info["vectors"]),
                "centroid": centroid,
            }
        self._centroid_cache = cache
        return cache
    
    def create_speaker(self, display_name: str, embedding: np.ndarray) -> str:
        """
        Create a new speaker with an initial embedding
        
        Args:
            display_name: Display name for the speaker
            embedding: NumPy array representing the speaker embedding
            
        Returns:
            Speaker ID (UUID string)
        """
        speaker_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()
        
        # Convert embedding to list for ChromaDB
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        # Store in ChromaDB with metadata
        self.collection.add(
            ids=[f"{speaker_id}_0"],  # First embedding gets index 0
            embeddings=[embedding_list],
            metadatas=[{
                "speaker_id": speaker_id,
                "display_name": display_name,
                "created_at": created_at,
                "embedding_index": 0
            }]
        )
        
        self._invalidate_centroid_cache()
        logger.info(f"Created speaker {speaker_id} with display name '{display_name}'")
        return speaker_id
    
    def add_embedding(self, speaker_id: str, embedding: np.ndarray) -> bool:
        """
        Add an additional embedding to an existing speaker
        
        Args:
            speaker_id: ID of the speaker
            embedding: NumPy array representing the speaker embedding
            
        Returns:
            True if successful, False otherwise
        """
        # Get existing embeddings for this speaker to determine next index
        existing = self.collection.get(
            where={"speaker_id": speaker_id}
        )
        
        if not existing['ids']:
            logger.warning(f"Speaker {speaker_id} not found")
            return False
        
        # Get metadata from first embedding to retrieve display_name
        first_metadata = existing['metadatas'][0]
        display_name = first_metadata['display_name']
        created_at = first_metadata['created_at']
        
        # Determine next embedding index
        max_index = max([int(m.get('embedding_index', 0)) for m in existing['metadatas']])
        next_index = max_index + 1
        
        # Convert embedding to list
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        # Add new embedding
        self.collection.add(
            ids=[f"{speaker_id}_{next_index}"],
            embeddings=[embedding_list],
            metadatas=[{
                "speaker_id": speaker_id,
                "display_name": display_name,
                "created_at": created_at,
                "embedding_index": next_index
            }]
        )
        
        self._invalidate_centroid_cache()
        logger.info(f"Added embedding {next_index} to speaker {speaker_id}")
        return True
    
    def query_nearest_embeddings(
        self, embedding: np.ndarray, limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Query ChromaDB for the nearest stored embeddings regardless of threshold.

        Returns:
            List of dicts with speaker_id, display_name, embedding_index, similarity,
            centroid_similarity (best enrollment first). Empty when the collection
            has no embeddings.
        """
        limit = max(1, min(int(limit), 10))
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

        try:
            count = self.collection.count()
        except Exception:
            count = 0
        if count <= 0:
            return []

        n_results = min(limit, count)
        results = self.collection.query(
            query_embeddings=[embedding_list],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        out: List[Dict[str, Any]] = []
        for i, chroma_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = round(float(1 - distance), 4)
            emb_idx = metadata.get("embedding_index")
            if emb_idx is None and chroma_id and "_" in str(chroma_id):
                try:
                    emb_idx = int(str(chroma_id).rsplit("_", 1)[-1])
                except ValueError:
                    emb_idx = 0
            out.append(
                {
                    "speaker_id": str(metadata.get("speaker_id", "")),
                    "display_name": str(metadata.get("display_name", "")),
                    "embedding_index": int(emb_idx if emb_idx is not None else 0),
                    "similarity": similarity,
                }
            )
        centroids = self._speaker_centroids()
        query = np.asarray(embedding, dtype=np.float32)
        for item in out:
            info = centroids.get(item["speaker_id"])
            if info and info.get("centroid") is not None:
                item["centroid_similarity"] = round(
                    embedding_cosine_similarity(query, info["centroid"]), 4
                )
            else:
                item["centroid_similarity"] = None
        return out

    def _speaker_from_id(
        self, speaker_id: str, display_name_fallback: str = ""
    ) -> Optional[Speaker]:
        all_embeddings = self.collection.get(where={"speaker_id": speaker_id})
        if not all_embeddings.get("ids"):
            return None
        first_meta = all_embeddings["metadatas"][0] if all_embeddings.get("metadatas") else {}
        return Speaker(
            id=speaker_id,
            display_name=display_name_fallback or first_meta.get("display_name", ""),
            created_at=str(first_meta.get("created_at", "")),
            embedding_count=len(all_embeddings["ids"]),
        )

    def _query_best_centroid_match(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[Speaker], float]:
        centroids = self._speaker_centroids()
        if not centroids:
            return None, 0.0
        query = np.asarray(embedding, dtype=np.float32)
        best_id: Optional[str] = None
        best_sim = -1.0
        best_info: Optional[Dict[str, Any]] = None
        for speaker_id, info in centroids.items():
            centroid = info.get("centroid")
            if centroid is None:
                continue
            sim = embedding_cosine_similarity(query, centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = speaker_id
                best_info = info
        if best_id is None or best_info is None:
            return None, 0.0
        return Speaker(
            id=best_id,
            display_name=str(best_info.get("display_name", "")),
            created_at=str(best_info.get("created_at", "")),
            embedding_count=int(best_info.get("embedding_count", 0)),
        ), round(float(best_sim), 4)

    def query_best_match(self, embedding: np.ndarray) -> Tuple[Optional[Speaker], float]:
        """
        Best identity match regardless of threshold.

        ``nearest``: highest cosine against any stored enrollment (Chroma).
        ``centroid``: highest cosine against each speaker's enrollment centroid.

        Returns:
            (Speaker, similarity) when a stored embedding exists; (None, 0.0) otherwise.
        """
        if self.match_mode == "centroid":
            return self._query_best_centroid_match(embedding)

        hits = self.query_nearest_embeddings(embedding, limit=1)
        if not hits:
            return None, 0.0

        hit = hits[0]
        speaker = self._speaker_from_id(hit["speaker_id"], hit["display_name"])
        if speaker is None:
            return None, 0.0
        return speaker, float(hit["similarity"])

    def find_similar_speaker(self, embedding: np.ndarray, threshold: Optional[float] = None) -> Optional[Speaker]:
        """
        Find a speaker with similar embedding using the configured match mode
        
        Args:
            embedding: NumPy array representing the speaker embedding to match
            threshold: Optional similarity threshold (uses instance default if not provided)
            
        Returns:
            Speaker object if match found, None otherwise
        """
        if threshold is None:
            threshold = self.similarity_threshold

        speaker, similarity = self.query_best_match(embedding)
        if speaker and similarity >= threshold:
            return speaker
        return None
    
    def get_speaker_by_id(self, speaker_id: str) -> Optional[Speaker]:
        """
        Get speaker information by ID
        
        Args:
            speaker_id: ID of the speaker
            
        Returns:
            Speaker object if found, None otherwise
        """
        results = self.collection.get(
            where={"speaker_id": speaker_id}
        )
        
        if not results['ids']:
            return None
        
        # Get metadata from first embedding
        metadata = results['metadatas'][0]
        
        return Speaker(
            id=speaker_id,
            display_name=metadata['display_name'],
            created_at=metadata['created_at'],
            embedding_count=len(results['ids'])
        )
    
    def match_speakers(self, embeddings: Dict[str, np.ndarray], threshold: Optional[float] = None) -> Dict[str, str]:
        """
        Match multiple embeddings to speakers
        
        Args:
            embeddings: Dictionary mapping speaker labels (e.g., "SPEAKER_00") to embeddings
            threshold: Optional similarity threshold
            
        Returns:
            Dictionary mapping speaker labels to display names (or original label if no match)
        """
        matches = {}
        
        for speaker_label, embedding in embeddings.items():
            speaker = self.find_similar_speaker(embedding, threshold)
            if speaker:
                matches[speaker_label] = speaker.display_name
            else:
                matches[speaker_label] = speaker_label  # Keep original label if no match
        
        return matches
    
    def get_all_speakers(self) -> List[Speaker]:
        """
        Get all speakers in the database
        
        Returns:
            List of Speaker objects
        """
        # Get all items from the collection
        results = self.collection.get()
        
        if not results['ids']:
            return []
        
        # Group by speaker_id to get unique speakers
        speakers_dict = {}
        for i, metadata in enumerate(results['metadatas']):
            speaker_id = metadata['speaker_id']
            if speaker_id not in speakers_dict:
                speakers_dict[speaker_id] = {
                    'display_name': metadata['display_name'],
                    'created_at': metadata['created_at'],
                    'embedding_count': 0
                }
            speakers_dict[speaker_id]['embedding_count'] += 1
        
        # Convert to list of Speaker objects
        speakers = []
        for speaker_id, info in speakers_dict.items():
            speakers.append(Speaker(
                id=speaker_id,
                display_name=info['display_name'],
                created_at=info['created_at'],
                embedding_count=info['embedding_count']
            ))
        
        return speakers
    
    def delete_speaker(self, speaker_id: str) -> bool:
        """
        Delete a speaker and all its embeddings from the database
        
        Args:
            speaker_id: ID of the speaker to delete
            
        Returns:
            True if speaker was found and deleted, False otherwise
        """
        # Get all embeddings for this speaker
        existing = self.collection.get(
            where={"speaker_id": speaker_id}
        )
        
        if not existing['ids']:
            logger.warning(f"Speaker {speaker_id} not found for deletion")
            return False
        
        # Delete all embeddings for this speaker
        self.collection.delete(
            where={"speaker_id": speaker_id}
        )
        
        self._invalidate_centroid_cache()
        logger.info(f"Deleted speaker {speaker_id} with {len(existing['ids'])} embeddings")
        return True

    def get_speaker_embeddings(self, speaker_id: str) -> List[Dict[str, Any]]:
        """
        Return metadata for each stored embedding for a speaker, sorted by embedding_index.

        Each dict has: embedding_index (int), created_at (str).
        """
        results = self.collection.get(
            where={"speaker_id": speaker_id},
            include=["metadatas"],
        )
        if not results["ids"]:
            return []
        out: List[Dict[str, Any]] = []
        for meta in results["metadatas"]:
            if not meta:
                continue
            idx = int(meta.get("embedding_index", 0))
            out.append(
                {
                    "embedding_index": idx,
                    "created_at": str(meta.get("created_at", "")),
                }
            )
        out.sort(key=lambda x: x["embedding_index"])
        return out

    def get_embedding_vector(
        self, speaker_id: str, embedding_index: int
    ) -> Optional[np.ndarray]:
        """Load a single embedding vector from Chroma, or None if missing."""
        chroma_id = f"{speaker_id}_{embedding_index}"
        got = self.collection.get(ids=[chroma_id], include=["embeddings"])
        ids = got.get("ids")
        embeddings = got.get("embeddings")
        # Avoid boolean evaluation on numpy arrays (ChromaDB may return ndarray).
        if ids is None or len(ids) == 0:
            return None
        if embeddings is None or len(embeddings) == 0:
            return None
        first = embeddings[0]
        if first is None:
            return None
        return np.asarray(first, dtype=np.float32)

    def delete_embedding(self, speaker_id: str, embedding_index: int) -> Tuple[bool, Optional[str]]:
        """
        Delete one embedding by index. Refuses if it would remove the speaker's last embedding.

        Returns:
            (True, None) on success
            (False, "not_found") if speaker or index missing
            (False, "last_embedding") if only one embedding remains
        """
        existing = self.collection.get(where={"speaker_id": speaker_id})
        if not existing["ids"]:
            return False, "not_found"
        if len(existing["ids"]) <= 1:
            return False, "last_embedding"
        chroma_id = f"{speaker_id}_{embedding_index}"
        if chroma_id not in existing["ids"]:
            return False, "not_found"
        self.collection.delete(ids=[chroma_id])
        self._invalidate_centroid_cache()
        logger.info("Deleted embedding %s for speaker %s", embedding_index, speaker_id)
        return True, None
