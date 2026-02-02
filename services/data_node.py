import argparse
import asyncio
import hashlib
import json
import re
import base64
import struct
from pathlib import Path
from typing import Any, Dict, List

import sys
import yaml
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from kademlia.network import Server as DHTServer

# Allow running from services/ without installing as package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_engine.hypergraph_query_engine import HyperGraphQueryEngine


def short_hash(text: str) -> str:
    """Stable short hash for evidence chunks."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_config(path: Path = Path("api_keys.yaml")) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# Handcrafted ontology descriptions to improve domain discrimination (English, concise).
DOMAIN_DESCRIPTIONS = {
    "univ_bench": (
        "University benchmark ontology. It models academic roles and relations in a university: "
        "professor, lecturer, faculty member, student (undergraduate and graduate), teaching assistant, "
        "employee relationships, courses and classes, departments, research groups, teaching and advising "
        "relationships such as teaches, takesCourse, worksFor. It is about higher-education people and roles, "
        "not about food or products."
    ),
    "pizza": (
        "Pizza ontology. It describes pizzas, named pizzas on a menu, pizza bases, and toppings such as "
        "cheese, tomato, pepperoni, vegetables, meat, and spiciness levels. It includes value partitions "
        "for toppings (mild/medium/hot), ingredient relations, and examples of different pizza combinations. "
        "It is about food items, not academic roles or people."
    ),
}


def extract_domain_from_path(path: Path) -> str:
    """Best-effort domain name from path name."""
    return path.name.replace("_parsed", "")


def format_fact(fact: Dict[str, Any], fact_id: str, text: str, ontology_prefix: str) -> Dict[str, Any]:
    """Normalize fact payload for API."""
    term_id = fact.get("_term_id") or fact.get("id", "")
    chunk_type = fact.get("_chunk_type", "unknown")
    return {
        "fact_id": fact_id,
        "term_id": term_id,
        "ontology_id": ontology_prefix,
        "chunk_type": chunk_type,
        "text": text,
        "hash": short_hash(text),
    }


def build_capability(name: str, domain: str, languages: List[str], ontologies: List[str], port: int, domain_embedding_b64: str) -> Dict[str, Any]:
    return {
        "name": name,
        "type": "data",
        "domains": [domain],
        "languages": languages,
        "ontologies": ontologies,
        "health": "ready",
        "endpoint": f"http://localhost:{port}",
        "domain_embedding_b64": domain_embedding_b64,
    }


class DataNode:
    def __init__(self, ontology_dir: Path, domain: str, port: int, languages: List[str], dht_port: int, dht_bootstrap: List[str]):
        self.ontology_dir = ontology_dir
        self.domain = domain
        self.port = port
        self.languages = languages
        self.dht_port = dht_port
        self.dht_bootstrap = dht_bootstrap
        self.dht = None  # type: ignore
        cfg = load_config()
        # LLM model name only affects optional generation; retrieval uses embeddings from hypergraph + query embedding via Ollama embeddings endpoint.
        ollama_model = cfg.get("OLLAMA_MODEL", "tinyllama")
        ollama_base = cfg.get("OLLAMA_BASE_URL", "http://localhost:11434")
        embed_model = cfg.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.embed_endpoint = ollama_base.rstrip("/")

        self.engine = HyperGraphQueryEngine(
            str(ontology_dir),
            use_ollama=True,
            ollama_model=ollama_model,
            ollama_base_url=f"{ollama_base.rstrip('/')}/v1",
        )

        # Compute domain embedding from domain + ontology name + sample labels/definitions
        domain_text = self._build_domain_text(domain, self.engine.metadata.get("ontology_name", ""))
        emb = self._get_embedding(domain_text, embed_model)
        emb_b64 = ""
        if emb:
            float32 = struct.pack(f"{len(emb)}f", *emb)  # no truncation; still ~3KB for 768 dims
            emb_b64 = base64.b64encode(float32).decode("ascii")
        self.domain_embedding_b64 = emb_b64

        # Derive ontology names from metadata
        ontology_name = self.engine.metadata.get("ontology_name", domain)
        self.capability = build_capability(
            name=f"data-node-{domain}",
            domain=domain,
            languages=languages,
            ontologies=[ontology_name],
            port=port,
            domain_embedding_b64=self.domain_embedding_b64,
        )

    async def start_dht(self):
        self.dht = DHTServer()
        await self.dht.listen(self.dht_port)
        peers = self.dht_bootstrap or [f"127.0.0.1:{self.dht_port}"]
        peer_addrs = [(h, int(p)) for h, p in (peer.split(":") for peer in peers)]
        await self.dht.bootstrap(peer_addrs)
        # Publish capability keyed by domain/lang (append-safe)
        key = f"domain:{self.domain}"
        current = await self.dht.get(key)
        caps = []
        if current:
            try:
                caps = json.loads(current)
                if not isinstance(caps, list):
                    caps = [caps]
            except Exception:
                caps = []
        # Deduplicate by endpoint
        endpoints = {c.get("endpoint") for c in caps}
        if self.capability["endpoint"] not in endpoints:
            caps.append(self.capability)
        await self.dht.set(key, json.dumps(caps))
        print(f"[data-node] DHT set key={key} count={len(caps)} value={caps}")

    def _get_embedding(self, text: str, model: str) -> List[float]:
        try:
            resp = requests.post(
                f"{self.embed_endpoint}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            emb = resp.json().get("embedding") or resp.json().get("embeddings", [])[0]
            return emb
        except Exception as e:
            print(f"[data-node] embedding error: {e}")
            return []

    def _build_domain_text(self, domain: str, ontology_name: str) -> str:
        """Create a richer domain description to improve embedding discrimination."""
        # Prefer handcrafted description if available
        if domain in DOMAIN_DESCRIPTIONS:
            base = DOMAIN_DESCRIPTIONS[domain]
        else:
            base = f"Ontology about {ontology_name or domain}"

        labels = []
        term_files = sorted(self.ontology_dir.glob("term_*.json"))[:50]
        for tf in term_files:
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    if obj.get("label"):
                        labels.append(obj["label"])
                    if obj.get("definition"):
                        labels.append(obj["definition"])
                    elif obj.get("id"):
                        labels.append(obj["id"])
            except Exception:
                continue
        label_text = " ".join(labels)
        return f"{base}. Key terms and definitions: {label_text}"

    async def stop_dht(self):
        if self.dht:
            self.dht.stop()


def create_app(node: DataNode) -> FastAPI:
    app = FastAPI(title="Data Node", version="0.1")

    @app.get("/capability")
    def capability():
        return node.capability

    @app.post("/query")
    def query(payload: Dict[str, Any]):
        question = payload.get("question", "")
        top_k = int(payload.get("top_k", 5))
        expand_hierarchy = bool(payload.get("expand_hierarchy", True))
        use_hyde = bool(payload.get("use_hyde", True))
        if not question:
            raise HTTPException(status_code=400, detail="question required")

        retrieved = node.engine.retrieve(
            question,
            top_k=top_k,
            expand_hierarchy=expand_hierarchy,
            use_hyde=use_hyde,
        )
        facts = []
        for idx, r in enumerate(retrieved):
            fact_obj = r["fact"]
            text = node.engine._format_fact(fact_obj)
            ontology_prefix = fact_obj.get("_ontology_prefix") or node.engine.metadata.get("id_prefix", "")
            facts.append(
                format_fact(fact_obj, fact_id=f"{node.domain}-f{idx:03d}", text=text, ontology_prefix=ontology_prefix)
                | {
                    "score": float(r["score"]),
                    "_raw_term": fact_obj.get("_raw_term", {}),
                    "_chunk_type": fact_obj.get("_chunk_type", ""),
                    "_parents": fact_obj.get("_parents", []),
                    "_searchable_text": fact_obj.get("_searchable_text", ""),
                }
            )
        return JSONResponse({"facts": facts, "count": len(facts), "query": question, "domain": node.domain})

    return app


def main():
    parser = argparse.ArgumentParser(description="Data Node (hypergraph-backed, DHT advertised).")
    parser.add_argument("--ontology-dir", type=Path, required=True, help="Path to parsed ontology directory.")
    parser.add_argument("--domain", type=str, help="Domain identifier (default from path name).")
    parser.add_argument("--port", type=int, default=9001, help="HTTP port for the node.")
    parser.add_argument("--languages", type=str, nargs="+", default=["en", "vi"], help="Supported languages.")
    parser.add_argument("--dht-port", type=int, default=7001, help="UDP port for DHT server.")
    parser.add_argument("--dht-bootstrap", type=str, nargs="*", default=[], help="Bootstrap peers host:port for DHT.")
    args = parser.parse_args()

    domain = args.domain or extract_domain_from_path(args.ontology_dir)
    node = DataNode(args.ontology_dir, domain, args.port, args.languages, args.dht_port, args.dht_bootstrap)

    async def runner():
        await node.start_dht()
        app = create_app(node)
        config = uvicorn.Config(app, host="0.0.0.0", port=args.port, log_level="warning")
        server = uvicorn.Server(config)
        print(f"[data-node] Serving HTTP on port {args.port} for domain={domain}")
        await server.serve()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
