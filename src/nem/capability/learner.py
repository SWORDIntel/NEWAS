"""Capability Learning Engine for NEMWAS"""

import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
import faiss
from dataclasses import dataclass, asdict
from collections import defaultdict

from sentence_transformers import SentenceTransformer
import torch

from ..core.npu_manager import NPUManager
from ..utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Capability:
    """Represents a learned capability"""
    id: str
    name: str
    description: str
    embedding: np.ndarray
    examples: List[Dict[str, Any]]
    performance_stats: Dict[str, float]
    created_at: float
    last_used: float
    usage_count: int = 0
    success_rate: float = 0.0


class CapabilityLearner:
    """Learns and manages agent capabilities using neural embeddings"""

    def __init__(self,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 storage_path: str = "./data/capabilities",
                 use_npu: bool = True):

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize NPU manager if requested
        self.use_npu = use_npu
        if use_npu:
            self.npu_manager = NPUManager()

        # Initialize embedding model
        self._initialize_embedder(embedding_model)

        # Initialize capability storage
        self.capabilities: Dict[str, Capability] = {}
        self.capability_index = None
        self._initialize_index()

        # Pattern matching for capability discovery
        self.pattern_cache = defaultdict(list)

        # Load existing capabilities
        self._load_capabilities()

        logger.info(f"Capability Learner initialized with {len(self.capabilities)} capabilities")

    def _initialize_embedder(self, model_name: str):
        """Initialize embedding model with optional NPU optimization"""

        logger.info(f"Initializing embedder: {model_name}")

        # Load sentence transformer
        self.embedder = SentenceTransformer(model_name)

        if self.use_npu and hasattr(self, 'npu_manager'):
            # Try to optimize for NPU
            try:
                # Export to ONNX first
                dummy_input = ["sample text"]
                dummy_encoding = self.embedder.encode(dummy_input, convert_to_tensor=True)

                # Get the model
                model = self.embedder[0].auto_model

                # Export to ONNX
                onnx_path = self.storage_path / f"{model_name.replace('/', '_')}.onnx"
                if not onnx_path.exists():
                    torch.onnx.export(
                        model,
                        (self.embedder.tokenize(dummy_input)['input_ids'],),
                        onnx_path,
                        export_params=True,
                        opset_version=14,
                        do_constant_folding=True,
                        input_names=['input_ids'],
                        output_names=['embeddings'],
                        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}}
                    )

                # Optimize for NPU
                optimized_path = self.npu_manager.optimize_model_for_npu(
                    str(onnx_path),
                    model_type="embedder",
                    quantization_preset="performance"
                )

                logger.info(f"Embedder optimized for NPU: {optimized_path}")
                # In production, we would load the optimized model here

            except Exception as e:
                logger.warning(f"Could not optimize embedder for NPU: {e}")

    def _initialize_index(self):
        """Initialize FAISS index for similarity search"""

        # Get embedding dimension
        embedding_dim = self.embedder.get_sentence_embedding_dimension()

        # Create index - using IndexFlatIP for inner product (cosine similarity)
        self.capability_index = faiss.IndexFlatIP(embedding_dim)

        # If we have many capabilities, we could use more sophisticated indices
        # like IndexIVFFlat or IndexHNSWFlat for better performance

    async def learn(self, execution_data: Dict[str, Any]) -> str:
        """Learn from successful execution"""

        # Extract pattern from execution
        pattern = self._extract_pattern(execution_data)

        # Generate embedding
        embedding = self._generate_embedding(pattern['description'])

        # Check if similar capability exists
        similar_capabilities = self._find_similar_capabilities(embedding, threshold=0.85)

        if similar_capabilities:
            # Update existing capability
            cap_id = similar_capabilities[0]['id']
            self._update_capability(cap_id, execution_data)
            logger.info(f"Updated existing capability: {cap_id}")
            return cap_id
        else:
            # Create new capability
            cap_id = self._create_capability(pattern, embedding, execution_data)
            logger.info(f"Created new capability: {cap_id}")
            return cap_id

    def _extract_pattern(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract reusable pattern from execution data"""

        query = execution_data.get('query_pattern', '')
        tools = execution_data.get('successful_tools', [])

        # Build pattern description
        tool_sequence = " -> ".join([t['tool'] for t in tools])
        description = f"Query: {query}\nTools: {tool_sequence}"

        # Extract key features
        pattern = {
            'description': description,
            'query_keywords': self._extract_keywords(query),
            'tool_sequence': [t['tool'] for t in tools],
            'tool_contexts': [t.get('context', '') for t in tools],
            'execution_time': execution_data.get('execution_time', 0)
        }

        return pattern

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""

        # Use NPU-optimized embedder if available
        embedding = self.embedder.encode(text, convert_to_numpy=True)

        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def _find_similar_capabilities(self,
                                  embedding: np.ndarray,
                                  k: int = 5,
                                  threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find similar capabilities using embedding similarity"""

        if self.capability_index.ntotal == 0:
            return []

        # Search index
        scores, indices = self.capability_index.search(
            embedding.reshape(1, -1),
            min(k, self.capability_index.ntotal)
        )

        # Filter by threshold and prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold:
                # Find capability by index
                cap_id = list(self.capabilities.keys())[idx]
                results.append({
                    'id': cap_id,
                    'capability': self.capabilities[cap_id],
                    'similarity': float(score)
                })

        return results

    def _create_capability(self,
                          pattern: Dict[str, Any],
                          embedding: np.ndarray,
                          execution_data: Dict[str, Any]) -> str:
        """Create new capability"""

        import uuid

        cap_id = str(uuid.uuid4())[:8]

        # Generate name from pattern
        name = self._generate_capability_name(pattern)

        capability = Capability(
            id=cap_id,
            name=name,
            description=pattern['description'],
            embedding=embedding,
            examples=[execution_data],
            performance_stats={
                'avg_execution_time': pattern['execution_time'],
                'success_count': 1,
                'failure_count': 0
            },
            created_at=time.time(),
            last_used=time.time(),
            usage_count=1,
            success_rate=1.0
        )

        # Store capability
        self.capabilities[cap_id] = capability

        # Add to index
        self.capability_index.add(embedding.reshape(1, -1))

        # Persist
        self._save_capability(capability)

        return cap_id

    def _update_capability(self, cap_id: str, execution_data: Dict[str, Any]):
        """Update existing capability with new execution data"""

        if cap_id not in self.capabilities:
            return

        cap = self.capabilities[cap_id]

        # Add example
        cap.examples.append(execution_data)

        # Update stats
        cap.usage_count += 1
        cap.last_used = time.time()

        # Update performance stats
        stats = cap.performance_stats
        old_avg = stats.get('avg_execution_time', 0)
        new_time = execution_data.get('execution_time', 0)

        # Running average
        stats['avg_execution_time'] = (old_avg * (cap.usage_count - 1) + new_time) / cap.usage_count

        if execution_data.get('success', True):
            stats['success_count'] = stats.get('success_count', 0) + 1
        else:
            stats['failure_count'] = stats.get('failure_count', 0) + 1

        cap.success_rate = stats['success_count'] / (stats['success_count'] + stats['failure_count'])

        # Persist updates
        self._save_capability(cap)

    def find_capabilities(self, query: str, k: int = 5) -> List[Tuple[Capability, float]]:
        """Find relevant capabilities for a query"""

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Find similar capabilities
        similar = self._find_similar_capabilities(query_embedding, k=k, threshold=0.6)

        # Return capability objects with scores
        results = []
        for item in similar:
            results.append((item['capability'], item['similarity']))

        return results

    def get_capability_suggestions(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get capability suggestions based on context"""

        suggestions = []

        # Extract context features
        query = context.get('query', '')
        history = context.get('history', [])

        # Find relevant capabilities
        if query:
            relevant_caps = self.find_capabilities(query, k=3)

            for cap, score in relevant_caps:
                if cap.success_rate > 0.7:  # Only suggest successful capabilities
                    suggestions.append({
                        'capability_id': cap.id,
                        'name': cap.name,
                        'confidence': score * cap.success_rate,
                        'avg_execution_time': cap.performance_stats['avg_execution_time'],
                        'tool_sequence': self._extract_tool_sequence(cap)
                    })

        # Sort by confidence
        suggestions.sort(key=lambda x: x['confidence'], reverse=True)

        return suggestions

    def _extract_tool_sequence(self, capability: Capability) -> List[str]:
        """Extract common tool sequence from capability examples"""

        if not capability.examples:
            return []

        # Get most recent example
        latest_example = capability.examples[-1]
        tools = latest_example.get('successful_tools', [])

        return [t['tool'] for t in tools]

    def _generate_capability_name(self, pattern: Dict[str, Any]) -> str:
        """Generate human-readable name for capability"""

        keywords = pattern.get('query_keywords', [])
        tools = pattern.get('tool_sequence', [])

        if keywords and tools:
            return f"{keywords[0]}_{tools[0]}" if keywords else f"capability_{tools[0]}"

        return "unnamed_capability"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""

        # Simple keyword extraction - in production, use better NLP
        import re

        # Remove common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

        # Tokenize and filter
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        # Return top 5 keywords
        return keywords[:5]

    def _save_capability(self, capability: Capability):
        """Save capability to disk"""

        cap_file = self.storage_path / f"{capability.id}.json"

        # Convert to serializable format
        cap_data = {
            'id': capability.id,
            'name': capability.name,
            'description': capability.description,
            'embedding': capability.embedding.tolist(),
            'examples': capability.examples,
            'performance_stats': capability.performance_stats,
            'created_at': capability.created_at,
            'last_used': capability.last_used,
            'usage_count': capability.usage_count,
            'success_rate': capability.success_rate
        }

        with open(cap_file, 'w') as f:
            json.dump(cap_data, f, indent=2)

    def _load_capabilities(self):
        """Load capabilities from disk"""

        for cap_file in self.storage_path.glob("*.json"):
            try:
                with open(cap_file, 'r') as f:
                    cap_data = json.load(f)

                # Reconstruct capability
                capability = Capability(
                    id=cap_data['id'],
                    name=cap_data['name'],
                    description=cap_data['description'],
                    embedding=np.array(cap_data['embedding']),
                    examples=cap_data['examples'],
                    performance_stats=cap_data['performance_stats'],
                    created_at=cap_data['created_at'],
                    last_used=cap_data['last_used'],
                    usage_count=cap_data.get('usage_count', 0),
                    success_rate=cap_data.get('success_rate', 0.0)
                )

                self.capabilities[capability.id] = capability

                # Add to index
                if self.capability_index is not None:
                    self.capability_index.add(capability.embedding.reshape(1, -1))

            except Exception as e:
                logger.error(f"Failed to load capability from {cap_file}: {e}")

    def export_capabilities(self) -> Dict[str, Any]:
        """Export all capabilities for backup or transfer"""

        return {
            'version': '1.0',
            'capabilities': [
                {
                    'id': cap.id,
                    'name': cap.name,
                    'description': cap.description,
                    'performance_stats': cap.performance_stats,
                    'usage_count': cap.usage_count,
                    'success_rate': cap.success_rate
                }
                for cap in self.capabilities.values()
            ],
            'total_capabilities': len(self.capabilities),
            'export_time': time.time()
        }
