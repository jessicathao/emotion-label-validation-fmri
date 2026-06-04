#!/bin/bash
MEDIA="$HOME/Downloads/Movies_cut"
OUT="$HOME/emofilm/media/whisper_out"
mkdir -p "$OUT"

FILMS=(
  "After_The_Rain_exp.mp4|496|AfterTheRain"
  "Between_Viewings_exp.mp4|808|BetweenViewings"
  "Chatter_exp.mp4|405|Chatter"
  "First_Bite_exp.mp4|599|FirstBite"
  "Lesson_Learned_exp.mp4|667|LessonLearned"
  "Payload_exp.mp4|1008|Payload"
  "Spaceman_exp.mp4|805|Spaceman"
  "To_Claire_From_Sonny_exp.mp4|402|ToClaireFromSonny"
  "You_Again_exp.mp4|798|YouAgain"
)

echo "=== Batch transcription: ${#FILMS[@]} films ==="
echo "Start: $(date)"; echo

for entry in "${FILMS[@]}"; do
  IFS='|' read -r fname expsec nicename <<< "$entry"
  fpath="$MEDIA/$fname"
  srtpath="$OUT/${fname%.mp4}.srt"
  echo "----------------------------------------"
  echo "FILM: $nicename  ($fname)"
  if [ -f "$srtpath" ]; then echo "  SKIP: srt already exists"; continue; fi
  if [ ! -f "$fpath" ]; then echo "  ERROR: file not found: $fpath"; continue; fi
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$fpath" 2>/dev/null)
  dur_int=${dur%.*}
  diff=$(( dur_int - expsec )); diff=${diff#-}
  if [ "$diff" -gt 5 ]; then
    echo "  WARNING: duration ${dur_int}s vs expected ${expsec}s (off by ${diff}s) - verify alignment later"
  else
    echo "  duration OK: ${dur_int}s (expected ~${expsec}s)"
  fi
  echo "  transcribing whisper small... ($(date +%H:%M:%S))"
  whisper "$fpath" --model small --language en --output_format srt --output_dir "$OUT" 2>&1 | tail -3
  echo "  done: $srtpath"
done

echo; echo "=== All done: $(date) ==="
ls -1 "$OUT"/*.srt 2>/dev/null | wc -l | xargs echo "Total .srt files now present:"
