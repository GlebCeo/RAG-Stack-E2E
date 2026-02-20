import json
import time
import httpx
import asyncio
from app.eval.metrics import answer_similarity, faithfulness_score

async def run_eval(qa_path: str, modes: list[str], api_url: str = "http://localhost:8001"):
    with open(qa_path) as f:
        qa_pairs = [json.loads(l) for l in f if l.strip()]

    results = {}

    for mode in modes:
        print(f"\n{'='*50}")
        print(f"🔍 Mode: {mode}  ({len(qa_pairs)} questions)")
        print(f"{'='*50}")

        mode_results = []
        async with httpx.AsyncClient(timeout=60) as client:
            for i, qa in enumerate(qa_pairs):
                t0 = time.time()
                r = await client.post(f"{api_url}/ask", json={"question": qa["question"], "mode": mode})
                data = r.json()
                latency = (time.time() - t0) * 1000

                chunks_text = [s["content"] for s in data.get("sources", [])]
                sim = answer_similarity(data["answer"], qa["answer"])
                faith = faithfulness_score(data["answer"], chunks_text)

                mode_results.append({
                    "question": qa["question"],
                    "expected": qa["answer"],
                    "generated": data["answer"],
                    "similarity": sim,
                    "faithfulness": faith,
                    "latency_ms": round(latency, 1),
                })

                status = "✅" if sim > 0.6 else "⚠️"
                print(f"  {status} Q{i+1}: sim={sim:.3f}  faith={faith:.3f}  {latency:.0f}ms")
                print(f"     Q: {qa['question'][:60]}...")

        avg_sim = sum(r["similarity"] for r in mode_results) / len(mode_results)
        avg_faith = sum(r["faithfulness"] for r in mode_results) / len(mode_results)
        avg_lat = sum(r["latency_ms"] for r in mode_results) / len(mode_results)

        results[mode] = {
            "avg_similarity": round(avg_sim, 4),
            "avg_faithfulness": round(avg_faith, 4),
            "avg_latency_ms": round(avg_lat, 1),
            "details": mode_results,
        }

        print(f"\n  📊 {mode}: similarity={avg_sim:.3f}  faithfulness={avg_faith:.3f}  latency={avg_lat:.0f}ms")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Mode':<20} {'Similarity':>10} {'Faithfulness':>13} {'Latency':>10}")
    print(f"{'-'*60}")
    for mode, r in results.items():
        print(f"{mode:<20} {r['avg_similarity']:>10.3f} {r['avg_faithfulness']:>13.3f} {r['avg_latency_ms']:>9.0f}ms")
    print(f"{'='*60}")

    return results
