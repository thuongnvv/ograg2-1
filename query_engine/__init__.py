# Optional imports - only load if dependencies available
__all__ = []

try:
    from query_engine.knowledge_graph_query_engine import (
        KnowledgeGraphListQueryEngine, KnowledgeGraphListQueryEngineOG, 
        KnowledgeGraphListQueryEngineReverse, KnowledgeGraphListQueryEngineDefault, 
        KnowledgeTriplesGraphQueryEngine
    )
    __all__.extend([
        "KnowledgeGraphListQueryEngine",
        "KnowledgeGraphListQueryEngineOG",
        "KnowledgeGraphListQueryEngineReverse",
        "KnowledgeGraphListQueryEngineDefault",
        "KnowledgeTriplesGraphQueryEngine"
    ])
except ImportError:
    pass

try:
    from query_engine.llm_query_engine import LLMQueryEngine
    __all__.append("LLMQueryEngine")
except ImportError:
    pass

try:
    from query_engine.rag_query_engine import RAGQueryEngine
    __all__.append("RAGQueryEngine")
except ImportError:
    pass

try:
    from query_engine.snippet_rag_query_engine import SnippetRAGQueryEngine
    __all__.append("SnippetRAGQueryEngine")
except ImportError:
    pass

try:
    from query_engine.ontograph_query_engine import OntoHyperGraphQueryEngine
    __all__.append("OntoHyperGraphQueryEngine")
except ImportError:
    pass

try:
    from query_engine.ontograph_query_engine_copy import OntoGraphQueryEngine
    __all__.append("OntoGraphQueryEngine")
except ImportError:
    pass

try:
    from query_engine.full_onto_query_engine import FullOntoQueryEngine
    __all__.append("FullOntoQueryEngine")
except ImportError:
    pass

try:
    from query_engine.raptor_query_engine import RaptorQueryEngine
    __all__.append("RaptorQueryEngine")
except ImportError:
    pass

# Skip graphrag_query_engine due to syntax errors
# try:
#     from query_engine.graphrag_query_engine import GraphRAGQueryEngine
#     __all__.append("GraphRAGQueryEngine")
# except ImportError:
#     pass

# GO Graph Retriever (always available - minimal dependencies)
try:
    from query_engine.go_graph_retriever import GOGraphRetriever
    __all__.append("GOGraphRetriever")
except ImportError:
    pass