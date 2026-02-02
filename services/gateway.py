
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from services.compute_node import main_async, argparse, load_api_config

app = FastAPI(title="OG-RAG Gateway", version="0.1")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_llm: bool = True
    domain_top: int = 1
    domain_threshold: float = 0.15
    dht_bootstrap: list[str] = []


@app.post("/ask")
async def ask(req: QueryRequest):
    if not req.question:
        raise HTTPException(status_code=400, detail="question required")
    # Build args namespace compatible with compute_node.main_async
    parser = argparse.ArgumentParser()
    parser.add_argument("--dummy")
    args = parser.parse_args([])  # empty
    # attach fields manually
    args.question = req.question
    args.domains = []
    args.top_k = req.top_k
    args.domain_top = req.domain_top
    args.domain_threshold = req.domain_threshold
    args.dht_port = 7101
    args.dht_bootstrap = req.dht_bootstrap
    args.use_llm = req.use_llm
    args.llm_model = None
    args.llm_base_url = None
    args.llm_api_key = None

    # Run compute logic and capture stdout via return structure
    try:
        await main_async(args)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}  # Response is printed by compute_node; gateway keeps simple


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
