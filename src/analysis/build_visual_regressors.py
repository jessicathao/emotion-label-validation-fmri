#!/usr/bin/env python3
"""build_visual_regressors.py - extract 1 Hz low-level visual features from a film.

Produces the stimulus-side regressors used to test whether occipital "valence"
is just the low-level visual confound (luminance, contrast, motion, edges, cuts,
saturation). Output is RAW 1 Hz, index 0 = film second 0, so it aligns to the
PleasantOther annotation and flows through decode_brain.place_on_tr exactly like
the human target (HRF convolution + TR interpolation happen there, not here).

USAGE
  python build_visual_regressors.py <film> [--video PATH] [--hz 6] [--cut-thresh 0.6]

Looks for the film at --video, else $EMOFILM_FILMS/<film>.* , else ~/emofilm_films/<film>.* .
Writes  data/visual/<film>_features_1hz.npz  and a sibling .csv for inspection.

Deps:  pip install opencv-python numpy   (opencv brings ffmpeg decoders)
"""
import os, sys, glob, argparse
import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv not found.  pip install opencv-python")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "data", "visual")
FILMS_DIR = os.environ.get("EMOFILM_FILMS", os.path.expanduser("~/emofilm_films"))

FEATURES = ["luminance", "rms_contrast", "motion", "edges", "cuts", "saturation"]
PROC_W = 160  # downscale width for speed; features are global statistics, robust to this


def find_video(film, explicit):
    if explicit:
        if not os.path.exists(explicit):
            sys.exit(f"video not found: {explicit}")
        return explicit
    for ext in ("mp4", "mkv", "mov", "avi", "webm", "m4v"):
        hits = glob.glob(os.path.join(FILMS_DIR, f"{film}.{ext}")) + \
               glob.glob(os.path.join(FILMS_DIR, f"*{film}*.{ext}"))
        if hits:
            return hits[0]
    sys.exit(f"no video for '{film}' in {FILMS_DIR}; pass --video PATH")


def frame_features(bgr):
    """Per-frame scalars from a downscaled BGR frame."""
    h = int(PROC_W * bgr.shape[0] / bgr.shape[1])
    small = cv2.resize(bgr, (PROC_W, h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = float(np.mean(np.sqrt(gx * gx + gy * gy)))
    return {
        "luminance": float(gray.mean()),
        "rms_contrast": float(gray.std()),
        "edges": edges,
        "saturation": float(sat.mean()),
        "_gray": gray,                 # carried out for motion / cut comparisons
        "_hsv": hsv,
    }


def hsv_hist(hsv):
    hh = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hh, hh).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("film")
    ap.add_argument("--video", default=None)
    ap.add_argument("--hz", type=float, default=6.0, help="frame sampling rate (Hz)")
    ap.add_argument("--cut-thresh", type=float, default=0.6,
                    help="HSV-hist correlation below this between sampled frames = a cut")
    a = ap.parse_args()

    video = find_video(a.film, a.video)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = nfr / fps if nfr else None
    step = max(1, int(round(fps / a.hz)))
    n_sec = int(np.floor(dur)) if dur else None
    print(f"{a.film}: {video}\n  fps {fps:.2f}, frames {nfr}, dur "
          f"{dur:.1f}s, sampling every {step} frames (~{fps/step:.1f} Hz)")

    # accumulate per-second sums + counts
    acc = {f: {} for f in FEATURES}          # feature -> {sec: [values]}
    prev_gray, prev_hist = None, None
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % step == 0:
            sec = int(fi / fps)
            ff = frame_features(frame)
            for f in ("luminance", "rms_contrast", "edges", "saturation"):
                acc[f].setdefault(sec, []).append(ff[f])
            if prev_gray is not None:
                acc["motion"].setdefault(sec, []).append(
                    float(np.mean(np.abs(ff["_gray"] - prev_gray))))
                corr = cv2.compareHist(prev_hist, hsv_hist(ff["_hsv"]), cv2.HISTCMP_CORREL)
                if corr < a.cut_thresh:
                    acc["cuts"].setdefault(sec, []).append(1.0)
            prev_gray, prev_hist = ff["_gray"], hsv_hist(ff["_hsv"])
        fi += 1
    cap.release()

    if n_sec is None:
        n_sec = max(max(d) for d in acc.values() if d) + 1

    out = {}
    for f in FEATURES:
        arr = np.zeros(n_sec, dtype=np.float32)
        for s in range(n_sec):
            vals = acc[f].get(s, [])
            arr[s] = (np.sum(vals) if f == "cuts" else np.mean(vals)) if vals else 0.0
        out[f] = arr

    os.makedirs(OUTDIR, exist_ok=True)
    npz = os.path.join(OUTDIR, f"{a.film}_features_1hz.npz")
    np.savez_compressed(npz, names=np.array(FEATURES), n_sec=n_sec,
                        hz_sampled=fps / step, **out)
    csv = os.path.join(OUTDIR, f"{a.film}_features_1hz.csv")
    with open(csv, "w") as fh:
        fh.write("sec," + ",".join(FEATURES) + "\n")
        for s in range(n_sec):
            fh.write(f"{s}," + ",".join(f"{out[f][s]:.5f}" for f in FEATURES) + "\n")
    print(f"  wrote {n_sec}s x {len(FEATURES)} features -> {npz}")
    print(f"  inspect: {csv}")
    print("  per-feature mean/sd:")
    for f in FEATURES:
        print(f"    {f:13s} mean {out[f].mean():+.4f}  sd {out[f].std():.4f}  "
              f"(cuts total {int(out['cuts'].sum())})" if f == "cuts"
              else f"    {f:13s} mean {out[f].mean():+.4f}  sd {out[f].std():.4f}")


if __name__ == "__main__":
    main()
