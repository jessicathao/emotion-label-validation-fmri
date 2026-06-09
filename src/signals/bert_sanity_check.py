import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "phanerozoic/BERT-Sentiment-Classifier"

print(f"Loading {MODEL} (first run downloads ~400 MB) ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()
print("  label map (id2label):", model.config.id2label, flush=True)

@torch.no_grad()
def predict(texts):
    enc = tok(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
    probs = torch.softmax(model(**enc).logits, dim=-1)
    pos_idx = 1
    for i, lab in model.config.id2label.items():
        if "pos" in lab.lower() or lab.upper() == "LABEL_1":
            pos_idx = i
    pred_ids = probs.argmax(dim=-1).tolist()
    labels = [model.config.id2label[i] for i in pred_ids]
    return labels, probs[:, pos_idx].tolist()

print("\n=== PART 1: SMOKE TEST ===")
examples = [
    "This is the best film I have ever seen, absolutely wonderful.",
    "A heartwarming and beautiful story that made me cry happy tears.",
    "What a fantastic, uplifting experience.",
    "This was a complete waste of time, utterly terrible.",
    "I hated every minute of it; boring and depressing.",
    "Awful acting and a miserable, painful plot.",
    "The weather is cloudy today.",
]
labels, pos = predict(examples)
for txt, lab, p in zip(examples, labels, pos):
    print(f"  [{lab:>9}  P(pos)={p:0.2f}]  {txt}")
print("  -> first 3 should read positive, next 3 negative.")

print("\n=== PART 2: BENCHMARK on SST-2 (first 300 validation examples) ===")
try:
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="validation[:300]")
    texts = [r["sentence"].strip() for r in ds]
    gold = [r["label"] for r in ds]
    _, pos = predict(texts)
    pred = [1 if p >= 0.5 else 0 for p in pos]
    correct = sum(int(a == b) for a, b in zip(pred, gold))
    print(f"  accuracy: {correct/len(gold)*100:0.1f}%  ({correct}/{len(gold)})")
    print("  model card reports ~89% on its own eval.")
except Exception as e:
    print(f"  benchmark skipped: {e}")
    print("  (smoke test above still tells you a lot)")

print("\nDone.")
