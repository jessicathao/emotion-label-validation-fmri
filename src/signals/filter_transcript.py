import json, sys, re

def normalize(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

def filter_segments(segs, max_repeat=3):
    kept, dropped = [], []
    i = 0; n = len(segs)
    while i < n:
        j = i
        base = normalize(segs[i]["text"])
        while j + 1 < n and normalize(segs[j+1]["text"]) == base and base != "":
            j += 1
        run_len = j - i + 1
        if run_len > max_repeat and base != "":
            kept.append(segs[i])
            for k in range(i+1, j+1):
                dropped.append(segs[k])
        else:
            kept.extend(segs[i:j+1])
        i = j + 1
    return kept, dropped

in_path, out_path = sys.argv[1], sys.argv[2]
max_repeat = int(sys.argv[3]) if len(sys.argv) > 3 else 3
data = json.load(open(in_path))
segs = data["segments"]
kept, dropped = filter_segments(segs, max_repeat)
data["segments"] = kept
data["n_segments"] = len(kept)
data["filtered_out"] = len(dropped)
json.dump(data, open(out_path, "w"), indent=2, ensure_ascii=False)
print(f"input segments : {len(segs)}")
print(f"kept           : {len(kept)}")
print(f"dropped (loops): {len(dropped)}")
if dropped:
    print("examples of dropped repeats:")
    seen=set()
    for d in dropped:
        t=d["text"]
        if t not in seen:
            print(f"   [{d['start']:.1f}s] {t}"); seen.add(t)
        if len(seen)>=5: break
print("saved:", out_path)
