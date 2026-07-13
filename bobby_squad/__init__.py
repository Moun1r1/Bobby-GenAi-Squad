"""bobby_squad — a domain-free persistent-self component for long-running LLM agents.

The proven core (see FINDINGS.md): put an agent's identity + goals + accumulated progress in a PINNED tier that
context-compaction can never touch, re-ground it periodically, and it maintains long-term goal persistence,
stable identity, and context-independent reasoning — with a ~5-6x gain in sustained progress across real
context wipes vs. a naive agent. No virtual world required.

Quickstart:
    from bobby_squad import Agent, SelfCore, LLM
    a = Agent(SelfCore(identity="a precise enumerator", goal="list primes in increasing order"), llm=LLM())
    while not a.converged():
        step = a.next_step("Output the next prime, digits only.")
        if step is None: break
"""
from .core import SelfCore, PersistentContext, Agent, stream_observer
from .squad import squad_solve
from .proving import confirm_gain, prove
from .ledger import IdeaLedger
from .llm import LLM
from .society import Society
from .dedup import near_dup, words
from .retrieval import LexicalRetriever, EmbeddingRetriever, embedding_available
from .room import KnowledgeRoom
from .search_agent import HypothesisSearcher
from .correction_memory import SemanticMemory, CorrectionMemory, FindingsMemory
from .agent_tools import ReadOnlyTools, SandboxTools, DgxTools, investigate, TOOL_SCHEMAS
from .metacognition import BehaviorTrace, MetaTools, area_of
from .board_tools import BoardTools, BOARD_SCHEMAS
from .worldsense import (WorldSense, perceive, signal, WorldStreamSource, FileChangeSource, LedgerSource,
                         PeerSource, ClockSource, EmotionState, SelfModelSource)
from .vault import KnowledgeVault, VaultHub, Note, slug, link_id
from .learned_retriever import LearnedRetriever, load_retriever
from .observability import RunStats

# torch is an OPTIONAL dependency — only the encoder bank / world-transformer layer need it (for training the
# learned heads on a GPU worker). `import bobby_squad` must work without torch installed, so these are guarded.
try:
    from .world_layer import WorldEncoder, WorldPrefixLM
    from .encoders import (ValueHead, RetrievalEncoder, TrajectoryMonitor, SelfMonitor, trajectory_dpo)
    _TORCH_LAYERS = ["WorldEncoder", "WorldPrefixLM", "ValueHead", "RetrievalEncoder", "TrajectoryMonitor",
                     "SelfMonitor", "trajectory_dpo"]
except ImportError:                                        # torch not installed → the training layers are unavailable
    _TORCH_LAYERS = []

__all__ = ["SelfCore", "PersistentContext", "Agent", "LLM", "Society", "near_dup", "words",
           "LexicalRetriever", "EmbeddingRetriever", "embedding_available", "KnowledgeRoom", "HypothesisSearcher",
           "SemanticMemory", "CorrectionMemory", "FindingsMemory", "ReadOnlyTools", "SandboxTools", "DgxTools", "investigate",
           "TOOL_SCHEMAS", "stream_observer", "squad_solve", "confirm_gain", "prove", "IdeaLedger",
           "BehaviorTrace", "MetaTools", "area_of", "BoardTools", "BOARD_SCHEMAS",
           "WorldSense", "perceive", "signal", "WorldStreamSource", "FileChangeSource", "LedgerSource",
           "PeerSource", "ClockSource", "EmotionState", "SelfModelSource",
           "KnowledgeVault", "VaultHub", "Note", "slug", "link_id",
           "LearnedRetriever", "load_retriever", "RunStats"] + _TORCH_LAYERS
__version__ = "0.1.0"
