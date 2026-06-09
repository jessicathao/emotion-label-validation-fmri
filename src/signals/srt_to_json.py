import json, re, sys
from pathlib import Path

def srt_time_to_seconds(t):
    t = t.strip().replace(",", ".")
    hh, mm, ss = t.split(":")
    return int(hh)*3600 + int(mm)*60 + float(ss)

def parse_srt(text):
    blocks = re.split(r"\n\s*\n", text.strip())
    segments = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if len(lines) < 2:
            continue
        ti = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if ti is None:
            continue
        a, b = lines[ti].split("-->")
        start, end = srt_time_to_seconds(a), srt_time_to_seconds(b)
        seg_text = " ".join(tl.strip() for tl in lines[ti+1:]).strip()
        if seg_text:
            segments.append({"start": round(start,3), "end": round(end,3), "text": seg_text})
    return segments

if len(sys.argv) != 4:
    sys.exit("Usage: python srt_to_json.py <srt_file> <film_name> <output_json>")
srt_path, film_name, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
raw = Path(srt_path).read_text(encoding="utf-8-sig")
segments = parse_srt(raw)
if not segments:
    sys.exit("No segments parsed - is this a valid .srt file?")
payload = {
    "film": film_name, "source": "srt",
    "duration_s": segments[-1]["end"], "n_segments": len(segments),
    "segments": segments,
    "full_text": " ".join(s["text"] for s in segments),
}
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
print(f"Parsed {len(segments)} segments from {srt_path}")
print(f"First line  : [{segments[0]['start']:.2f}s] {segments[0]['text']}")
print(f"Last line   : [{segments[-1]['start']:.2f}s] {segments[-1]['text']}")
print(f"Span        : {segments[0]['start']:.1f}s -> {segments[-1]['end']:.1f}s")
print(f"Saved JSON  : {out_path}")
