# Speaker database operations using ChromaDB
import logging
import uuid
from typing import Any, Optional, Dict, List, Tuple
from datetime import datetime
import numpy as np
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class Speaker:
    """Represents a speaker with metadata"""
    def __init__(self, id: str, display_name: str, created_at: str, embedding_count: int):
        self.id = id
        self.display_name = display_name
        self.created_at = created_at
        self.embedding_count = embedding_count

class SpeakerDB:
    """Database manager for speaker embeddings using ChromaDB"""
    
    def __init__(self, db_path: str, similarity_threshold: float = 0.7):
        """
        Initialize the speaker database
        
        Args:
            db_path: Path to the ChromaDB database
            similarity_threshold: Cosine similarity threshold for matching (default: 0.7)
        """
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        
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
            
            logger.info(f"Initialized speaker database at {db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            raise
    
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
        
        logger.info(f"Added embedding {next_index} to speaker {speaker_id}")
        return True
    
    def find_similar_speaker(self, embedding: np.ndarray, threshold: Optional[float] = None) -> Optional[Speaker]:
        """
        Find a speaker with similar embedding using cosine similarity
        
        Args:
            embedding: NumPy array representing the speaker embedding to match
            threshold: Optional similarity threshold (uses instance default if not provided)
            
        Returns:
            Speaker object if match found, None otherwise
        """
        if threshold is None:
            threshold = self.similarity_threshold
        
        # Convert embedding to list
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        
        # Query ChromaDB for similar embeddings
        results = self.collection.query(
            query_embeddings=[embedding_list],
            n_results=1,
            include=['metadatas', 'distances']
        )
        
        if not results['ids'] or not results['ids'][0]:
            return None
        
        # Get the best match
        distance = results['distances'][0][0]
        metadata = results['metadatas'][0][0]
        
        # ChromaDB returns distance (1 - cosine_similarity), so we convert to similarity
        similarity = 1 - distance
        
        if similarity >= threshold:
            speaker_id = metadata['speaker_id']
            
            # Get all embeddings for this speaker to count them
            all_embeddings = self.collection.get(
                where={"speaker_id": speaker_id}
            )
            
            return Speaker(
                id=speaker_id,
                display_name=metadata['display_name'],
                created_at=metadata['created_at'],
                embedding_count=len(all_embeddings['ids'])
            )
        
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
        logger.info("Deleted embedding %s for speaker %s", embedding_index, speaker_id)
        return True, None
