import json, sys, math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "phanerozoic/BERT-Sentiment-Classifier"

if len(sys.argv) < 3:
    sys.exit("Usage: python make_bert_signal.py <transcript.json> <out.json> [grid_length_s]")
in_path, out_path = sys.argv[1], sys.argv[2]
grid_len = int(sys.argv[3]) if len(sys.argv) > 3 else None

data = json.load(open(in_path))
segs = data["segments"]
print(f"{data.get('film','?')}: {len(segs)} segments, span {segs[0]['start']:.1f}-{segs[-1]['end']:.1f}s")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL); model.eval()

texts = [s["text"] for s in segs]
with torch.no_grad():
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    probs = torch.softmax(model(**enc).logits, dim=-1)[:, 1].tolist()

seg_val = [2.0 * p - 1.0 for p in probs]

end_time = max(s["end"] for s in segs)
N = grid_len if grid_len else int(math.ceil(end_time))
valence = [float("nan")] * N
for s, v in zip(segs, seg_val):
    a = int(math.floor(s["start"])); b = int(math.ceil(s["end"]))
    for t in range(max(0, a), min(N, b)):
        valence[t] = v
coverage = sum(1 for v in valence if not math.isnan(v))

out = {"film": data.get("film","?"), "fps": 1, "duration_s": N,
       "valence": valence, "n_segments": len(segs), "coverage_s": coverage,
       "subtitle_offset_s": 0}
json.dump(out, open(out_path, "w"))
print(f"grid length: {N}s, dialogue coverage: {coverage}s ({100*coverage/N:.1f}% of grid)")
print("per-segment P(pos):", [round(p,2) for p in probs])
print("saved:", out_path)
