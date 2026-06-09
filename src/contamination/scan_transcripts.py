#!/usr/bin/env python3
"""Flag likely transcript contamination in Whisper .srt files so you only read
the suspicious parts. Checks for: credits hallucinations, repeated/looping
lines, and rhyme/fragment patterns suggesting song lyrics."""
import os, re, glob
from collections import Counter

SRT_DIR = os.path.expanduser("~/emofilm/media/whisper_out")
CREDIT_PAT = re.compile(r"thank you for watching|subscribe|copyright|©|like and|"
                        r"see you next|thanks for watching|amara\.org|transcri|"
                        r"caption|www\.|http", re.I)

# Whisper-sourced films to audit (ToS + Sintel use official subs; skip them)
FILES = [
    "After_The_Rain_exp.srt", "Between_Viewings_exp.srt", "Chatter_exp.srt",
    "Lesson_Learned_exp.srt", "Payload_exp.srt", "Spaceman_exp.srt",
    "Superhero_exp.srt", "The_secret_number_exp.srt",
    "To_Claire_From_Sonny_exp.srt", "You_Again_exp.srt",
]

def parse_srt(path):
    """Return list of (index, start_s, end_s, text)."""
    blocks = open(path, encoding="utf-8", errors="replace").read().strip().split("\n\n")
    out = []
    for b in blocks:
        lines = b.strip().split("\n")
        if len(lines) < 2: continue
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)", lines[1])
        if not m: continue
        h1,m1,s1 = int(m.group(1)),int(m.group(2)),int(m.group(3))
        h2,m2,s2 = int(m.group(5)),int(m.group(6)),int(m.group(7))
        start = h1*3600+m1*60+s1; end = h2*3600+m2*60+s2
        text = " ".join(lines[2:]).strip()
        out.append((lines[0], start, end, text))
    return out

for fn in FILES:
    path = os.path.join(SRT_DIR, fn)
    print(f"\n=== {fn} ===")
    if not os.path.exists(path):
        print("  FILE NOT FOUND"); continue
    segs = parse_srt(path)
    if not segs:
        print("  no segments parsed"); continue
    texts = [t for _,_,_,t in segs]

    # 1) credits hallucination in last 4 segments
    for _,st,en,t in segs[-4:]:
        if CREDIT_PAT.search(t):
            print(f"  [CREDITS?] {st}-{en}s: {t[:75]}")

    # 2) repeated lines (3+ identical)
    counts = Counter(t.lower() for t in texts if len(t) > 3)
    for txt, c in counts.items():
        if c >= 3:
            print(f"  [REPEAT x{c}] {txt[:70]}")

    # 3) long single segments (>=12s), possible song/montage held text
    for _,st,en,t in segs:
        if en-st >= 12:
            print(f"  [LONG {en-st}s] {st}s: {t[:75]}")

    # 4) gap then isolated block (>=20s silence before a segment), montage/song marker
    for i in range(1,len(segs)):
        gap = segs[i][1] - segs[i-1][2]
        if gap >= 20:
            print(f"  [GAP {gap}s before] {segs[i][1]}s: {segs[i][3][:60]}")

    print(f"  ({len(segs)} segments, {segs[-1][2]}s total)")
