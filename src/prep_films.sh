#!/bin/bash
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/signals"
OUT="$HOME/emofilm/media/whisper_out"
WORK="$HOME/emofilm/media"

FILMS=(
  "After_The_Rain_exp|AfterTheRain"
  "Between_Viewings_exp|BetweenViewings"
  "Lesson_Learned_exp|LessonLearned"
  "Spaceman_exp|Spaceman"
  "To_Claire_From_Sonny_exp|ToClaireFromSonny"
  "You_Again_exp|YouAgain"
  "Payload_exp|Payload"
  "Chatter_exp|Chatter"
)

for entry in "${FILMS[@]}"; do
  IFS='|' read -r base nice <<< "$entry"
  srt="$OUT/${base}.srt"
  raw="$WORK/${nice}_whisper.json"
  filt="$WORK/${nice}_filt3.json"
  bert="$WORK/${nice}_bert.json"
  echo "==== $nice ===="
  python "$SRC/srt_to_json.py" "$srt" "$nice" "$raw"        | tail -1
  python "$SRC/filter_transcript.py" "$raw" "$filt" 3       | grep -E "kept|dropped"
  python "$SRC/make_bert_signal.py" "$filt" "$bert" 2>/dev/null | grep -E "coverage"
  echo
done
echo "=== prep done. BERT signals in $WORK/<Film>_bert.json ==="
