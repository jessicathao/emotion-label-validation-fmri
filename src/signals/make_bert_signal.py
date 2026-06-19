import json, sys, math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "phanerozoic/BERT-Sentiment-Classifier"

# ---- args: <transcript.json> <out.json> [grid_length_s] [--window N] ----
# --window 0 (default): each segment scored in ISOLATION (original behavior).
# --window N: the target plus its +/-N neighbours are concatenated and scored as
#   one input. NOTE: BERT has no instruction channel, so unlike the Gemini context
#   arm (which is GIVEN context and scores only the target), BERT-context simply
#   sees more text and blends it. Overlapping windows also induce mild smoothing.
#   This asymmetry is the point: context helps a model that can USE it as context.
args = sys.argv[1:]
window = 0
if "--window" in args:
    wi = args.index("--window"); window = int(args[wi + 1]); del args[wi:wi + 2]
if len(args) < 2:
    sys.exit("Usage: python make_bert_signal.py <transcript.json> <out.json> [grid_length_s] [--window N]")
in_path, out_path = args[0], args[1]
grid_len = int(args[2]) if len(args) > 2 else None

data = json.load(open(in_path))
segs = data["segments"]
print(f"{data.get('film','?')}: {len(segs)} segments, span {segs[0]['start']:.1f}-{segs[-1]['end']:.1f}s, window={window}")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL); model.eval()

def windowed_text(i):
    lo = max(0, i - window); hi = min(len(segs), i + window + 1)
    return " ".join(s["text"] for s in segs[lo:hi])

if window > 0:
    texts = [windowed_text(i) for i in range(len(segs))]
    max_len = 512
else:
    texts = [s["text"] for s in segs]
    max_len = 256

with torch.no_grad():
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
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
       "subtitle_offset_s": 0,
       "condition": "context" if window > 0 else "isolation", "window": window}
json.dump(out, open(out_path, "w"))
print(f"grid length: {N}s, dialogue coverage: {coverage}s ({100*coverage/N:.1f}% of grid)")
print("per-segment P(pos):", [round(p,2) for p in probs])
print("saved:", out_path)
