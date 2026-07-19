from .core import SelfCore, PersistentContext, Agent, stream_observer
from .squad import squad_solve
from .soma_flywheel import PluginStore, DistillationCorpus
from .proving import confirm_gain, prove
from .ledger import IdeaLedger
from .jobs import JobRegistry
from .dedup_ast import AstDedup, fingerprint
from .engine import Engine, EventLog, PluginRegistry, Plugin, Event, plugin_router
from .router import OODGate, ood_plugin_router, competence_router
from . import burn_in
from . import primitive_intel
from . import primitive_lib
from .ops_world import OpsWorld, WORKFLOWS, operate
from . import swe_bench
from .fsm import FSM, cluster_match
from .telemetry import Telemetry
from .surrogate import Surrogate, code_features
from .blackboard import Blackboard
from .harness import Scenario, DataCollector, Report, verdict as harness_verdict, ci95
from . import synthbench
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

# numpy is OPTIONAL — only the Sheaf-ADMM consensus harvest and the learned retriever need it. Keep the core
# stdlib-only (`import bobby_squad` must work without numpy), so these are guarded.
try:
    from .sheaf_consensus import sheaf_consensus, make_consensus_harvest, ConsensusResult
    from .learned_retriever import LearnedRetriever, load_retriever
    _NUMPY = ["sheaf_consensus", "make_consensus_harvest", "ConsensusResult", "LearnedRetriever", "load_retriever"]
except ImportError:                                        # numpy not installed → these features are unavailable
    _NUMPY = []

__all__ = ["SelfCore", "PersistentContext", "Agent", "LLM", "Society", "near_dup", "words",
           "LexicalRetriever", "EmbeddingRetriever", "embedding_available", "KnowledgeRoom", "HypothesisSearcher",
           "SemanticMemory", "CorrectionMemory", "FindingsMemory", "ReadOnlyTools", "SandboxTools", "DgxTools", "investigate",
           "TOOL_SCHEMAS", "stream_observer", "squad_solve",
           "PluginStore", "DistillationCorpus", "confirm_gain", "prove", "IdeaLedger",
           "JobRegistry", "AstDedup", "fingerprint",
           "Engine", "EventLog", "PluginRegistry", "Plugin", "Event", "plugin_router",
           "OODGate", "ood_plugin_router", "competence_router", "burn_in",
           "FSM", "cluster_match", "Telemetry", "Surrogate", "code_features", "Blackboard",
           "Scenario", "DataCollector", "Report", "harness_verdict", "ci95", "synthbench",
           "BehaviorTrace", "MetaTools", "area_of", "BoardTools", "BOARD_SCHEMAS",
           "WorldSense", "perceive", "signal", "WorldStreamSource", "FileChangeSource", "LedgerSource",
           "PeerSource", "ClockSource", "EmotionState", "SelfModelSource",
           "KnowledgeVault", "VaultHub", "Note", "slug", "link_id",
           "RunStats", "OpsWorld", "WORKFLOWS", "operate"] + _TORCH_LAYERS + _NUMPY
__version__ = "0.1.0"
