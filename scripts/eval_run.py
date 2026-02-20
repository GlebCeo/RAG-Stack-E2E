import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.eval.harness import run_eval

async def main():
    results = await run_eval(
        qa_path="data/eval_qa.jsonl",
        modes=["vector", "hybrid", "hybrid+rerank"],
    )
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n💾 Results saved to eval_results.json")

if __name__ == "__main__":
    asyncio.run(main())
