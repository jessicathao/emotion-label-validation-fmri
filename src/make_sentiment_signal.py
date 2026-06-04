import json, sys, math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if len(sys.argv) < 4:
    sys.exit("Usage: python make_sentiment_signal.py <transcript.json> <out.json> <model_name> [grid_length_s]")
in_path, out_path, MODEL = sys.argv[1], sys.argv[2], sys.argv[3]
grid_len = int(sys.argv[4]) if len(sys.argv) > 4 else None

data = json.load(open(in_path))
segs = data["segments"]
print(f"{data.get('film','?')}: {len(segs)} segments | model={MODEL}")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL); model.eval()

# find which output index means POSITIVE, from the model's own label map
id2label = model.config.id2label
pos_idx = None
for idx, lab in id2label.items():
    if str(lab).lower() in ("positive", "label_1", "pos", "5 stars", "4 stars"):
        pos_idx = int(idx)
# fallback: if binary and no clear name, assume higher index = positive
if pos_idx is None:
    pos_idx = max(int(i) for i in id2label) if len(id2label) <= 3 else None
if pos_idx is None:
    sys.exit(f"Cannot determine positive label from {id2label}")
print(f"label map: {id2label} -> using index {pos_idx} as POSITIVE")

texts = [s["text"] for s in segs]
with torch.no_grad():
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    probs = torch.softmax(model(**enc).logits, dim=-1)[:, pos_idx].tolist()

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
       "model": MODEL, "subtitle_offset_s": 0}
json.dump(out, open(out_path, "w"))
print(f"coverage: {coverage}s ({100*coverage/N:.1f}%); saved {out_path}")
