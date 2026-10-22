"""End-to-end check of near_dup.embedding_dedupe_gpu's per-language overrides:
mt uses LaBSE, de keeps the default MiniLM.

usage: python _override_check.py <kept.jsonl> <minilm_dir> <labse_dir>
"""
import json
import sys

from synthgen.pipeline import near_dup


def main() -> None:
    path, minilm, labse = sys.argv[1:4]
    rows, n = [], {"mt": 0, "de": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["lang"] in n and n[r["lang"]] < 3000:
                n[r["lang"]] += 1
                rows.append({"lang": r["lang"], "prompt": r["prompt"]})
            if all(v >= 3000 for v in n.values()):
                break
    kept, rep = near_dup.embedding_dedupe_gpu(
        rows, model_name=minilm, threshold=0.97,
        lang_overrides={"mt": {"model": labse, "threshold": 0.95}})
    print(rep)
    assert rep["lang_overrides"]["mt"] == {"threshold": 0.95, "model": labse}, "override not applied"
    assert "de" not in rep["lang_overrides"]
    print("OVERRIDE CHECK: PASS")


if __name__ == "__main__":
    main()
