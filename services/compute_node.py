
import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
import base64
import struct
from kademlia.network import Server as DHTServer
from openai import OpenAI
import requests
import numpy as np


def load_api_config(path: Path = Path("api_keys.yaml")) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def guess_domains(question: str) -> List[str]:
    return ["univ_bench", "pizza"]  # no keyword routing; let embeddings decide


async def dht_get(server: DHTServer, key: str) -> List[Dict[str, Any]]:
    raw = await server.get(key)
    if not raw:
        return []
    try:
        # Could be single capability or list
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return [data]
    except Exception:
        return []


async def fetch_data_node(client: httpx.AsyncClient, node: Dict[str, Any], question: str, top_k: int) -> Dict[str, Any]:
    url = node["endpoint"].rstrip("/") + "/query"
    t0 = time.time()
    resp = await client.post(
        url,
        json={
            "question": question,
            "top_k": top_k,
            "expand_hierarchy": True,
            "use_hyde": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    data["latency_ms"] = int((time.time() - t0) * 1000)
    data["node"] = node
    return data


def merge_evidence(responses: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Merge by hash; keep highest score and accumulate sources
    evidence = {}
    for r in responses:
        node_name = r["node"]["name"]
        for fact in r.get("facts", []):
            h = fact["hash"]
            if h not in evidence:
                evidence[h] = {
                    "fact": fact,
                    "sources": [],
                    "score": fact.get("score", 0.0),
                }
            evidence[h]["sources"].append(node_name)
            evidence[h]["score"] = max(evidence[h]["score"], fact.get("score", 0.0))
    merged = list(evidence.values())
    merged.sort(key=lambda x: x["score"], reverse=True)
    return {"evidence": merged}


def render_answer_llm(question: str, merged: Dict[str, Any], model: str, base_url: str, api_key: str) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    bullets = []
    for item in merged["evidence"][:5]:
        f = item["fact"]
        bullets.append(f"- {f['text']} (sources: {', '.join(item['sources'])})")
    context = "\n".join(bullets) if bullets else "No evidence."
    prompt = f"Question: {question}\nEvidence:\n{context}\n\nAnswer concisely using only the evidence. If insufficient, say so."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise QA assistant. Use only provided evidence."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    return resp.choices[0].message.content or ""


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def get_embedding(text: str, model: str, base_url: str) -> Optional[np.ndarray]:
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        emb = data.get("embedding") or (data.get("embeddings") or [None])[0]
        if emb is None:
            return None
        return np.array(emb, dtype=float)
    except Exception as e:
        print(f"[compute-node] embedding error: {e}")
        return None


def decode_domain_embedding(cap: Dict[str, Any]) -> Optional[np.ndarray]:
    """Decode domain embedding from capability (base64 float32 or plain list)."""
    if cap.get("domain_embedding_b64"):
        try:
            raw = base64.b64decode(cap["domain_embedding_b64"])
            count = len(raw) // 4
            vals = struct.unpack(f"{count}f", raw)
            return np.array(vals, dtype=float)
        except Exception:
            return None
    if cap.get("domain_embedding"):
        try:
            return np.array(cap["domain_embedding"], dtype=float)
        except Exception:
            return None
    return None


async def main_async(args):
    cfg = load_api_config()
    embed_model = cfg.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    embed_base = cfg.get("OLLAMA_BASE_URL", "http://localhost:11434")
    evidence_threshold = float(cfg.get("EVIDENCE_SCORE_THRESHOLD", 0.3))
    domain_fail_msg = cfg.get("DOMAIN_FAIL_MESSAGE", "No suitable ontology found for this question.")

    # Start DHT client/server
    dht = DHTServer()
    await dht.listen(args.dht_port)
    if args.dht_bootstrap:
        peers = [(h, int(p)) for h, p in (peer.split(":") for peer in args.dht_bootstrap)]
        await dht.bootstrap(peers)

    # Embed query to score domains
    query_emb = get_embedding(args.question, embed_model, embed_base)
    domains = args.domains or guess_domains(args.question)
    candidates = []
    for domain in domains:
        key = f"domain:{domain}"
        nodes = await dht_get(dht, key)
        for n in nodes:
            n["domain"] = domain
            # score by cosine if both embeddings present
            score = 0.0
            if query_emb is not None:
                try:
                    dom_emb = decode_domain_embedding(n)
                    if dom_emb is not None:
                        score = cosine(query_emb, dom_emb)
                except Exception:
                    score = 0.0
            n["domain_score"] = score
        candidates.extend(nodes)

    # If we have embeddings, keep top by domain_score and threshold
    if query_emb is not None and candidates:
        candidates.sort(key=lambda x: x.get("domain_score", 0.0), reverse=True)
        if args.domain_threshold > 0:
            filtered = [c for c in candidates if c.get("domain_score", 0.0) >= args.domain_threshold]
            if filtered:
                candidates = filtered
        if args.domain_top and args.domain_top > 0:
            candidates = candidates[: args.domain_top]

    # Guard: if best domain score still below threshold, abort early
    if query_emb is not None and candidates:
        best_score = candidates[0].get("domain_score", 0.0)
        if best_score < args.domain_threshold:
            print(json.dumps({"answer": domain_fail_msg, "domain_scores": {c['name']: c.get('domain_score', 0.0) for c in candidates}}, ensure_ascii=False, indent=2))
            dht.stop()
            return

    if not candidates:
        print(f"No nodes found for domains {domains}")
        return

    async with httpx.AsyncClient() as client:
        tasks = [fetch_data_node(client, n, args.question, args.top_k) for n in candidates]
        results = await asyncio.gather(*tasks)

    merged = merge_evidence(results)

    # Guardrail: insufficient evidence
    if not merged["evidence"]:
        output = {
            "question": args.question,
            "domains": domains,
            "nodes_contacted": [r["node"]["name"] for r in results],
            "domain_scores": {r["node"]["name"]: r["node"].get("domain_score", 0.0) for r in results},
            "answer": "Insufficient evidence.",
            "evidence": [],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        dht.stop()
        return
    # If top evidence score too low, return cautious answer
    top_score = merged["evidence"][0]["score"]
    if top_score < evidence_threshold:
        output = {
            "question": args.question,
            "domains": domains,
            "nodes_contacted": [r["node"]["name"] for r in results],
            "domain_scores": {r["node"]["name"]: r["node"].get("domain_score", 0.0) for r in results},
            "answer": "Insufficient evidence.",
            "evidence": merged["evidence"],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        dht.stop()
        return

    answer = "LLM disabled"
    if args.use_llm:
        llm_base = args.llm_base_url or cfg.get("openai_base_url")
        llm_model = args.llm_model or cfg.get("openai_model", "gpt-4")
        llm_key = args.llm_api_key or cfg.get("openai_api_key")
        if not llm_key:
            raise RuntimeError("LLM requested but no API key provided (set in api_keys.yaml or via --llm-api-key)")
        answer = render_answer_llm(
            args.question,
            merged,
            model=llm_model,
            base_url=llm_base,
            api_key=llm_key,
        )

    output = {
        "question": args.question,
        "domains": domains,
        "nodes_contacted": [r["node"]["name"] for r in results],
        "latencies_ms": {r["node"]["name"]: r["latency_ms"] for r in results},
        "domain_scores": {r["node"]["name"]: r["node"].get("domain_score", 0.0) for r in results},
        "evidence": merged["evidence"],
        "answer": answer,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))

    dht.stop()


def main():
    parser = argparse.ArgumentParser(description="Compute Node (single) with DHT discovery.")
    parser.add_argument("--question", type=str, required=True, help="User question.")
    parser.add_argument("--domains", type=str, nargs="*", default=[], help="Force domains (override guess).")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument("--domain-top", type=int, default=1, dest="domain_top", help="Limit number of data nodes by domain score (0 = no limit).")
    parser.add_argument("--domain-threshold", type=float, default=0.15, dest="domain_threshold", help="Minimum cosine score to keep a data node (fallback to best if none).")
    parser.add_argument("--dht-port", type=int, default=7101, help="UDP port to listen for DHT.")
    parser.add_argument("--dht-bootstrap", type=str, nargs="*", default=[], help="Bootstrap peers host:port.")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM to render answer.")
    parser.add_argument("--llm-model", type=str, default=None, help="LLM model name (default from api_keys.yaml or gpt-4).")
    parser.add_argument("--llm-base-url", type=str, default=None, help="LLM base URL (default from api_keys.yaml).")
    parser.add_argument("--llm-api-key", type=str, default=None, help="LLM API key (default from api_keys.yaml).")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
