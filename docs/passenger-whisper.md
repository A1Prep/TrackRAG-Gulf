# English passenger speech on CPU

The release can transcribe an uploaded WAV or a microphone turn with English Whisper Small. The passenger reviews and edits the words before sending them to the visual model. Each voice turn keeps the original WAV hash, the recognizer and its original text. The edited instruction is recorded separately. Choosing a destination still requires confirmation.

## Run on the project laptop

The release detects the installed Whisper model and packages. To select it explicitly in PowerShell, run the following from the project root.

```powershell
$py = ".venv-grounding-local/Scripts/python.exe"
$small = Join-Path $env:LOCALAPPDATA "TrackRAG-method-cache/faster-whisper-small.en"
& $py -m tools.passenger_release start --whisper-model "$small"
```

Use `--asr-model <vosk-directory>` for the original lightweight recognizer or `--text-only` for typed instructions. Voice recognition runs locally on CPU. The browser accepts 16-bit PCM, mono or stereo WAV, 16–48 kHz, up to 60 seconds.

For a fresh Python 3.11 project environment, install the speech dependencies and fetch the four model files once. In the project laptop's existing environment, only three additive wheels were needed. They sit in a separate cache without replacing the grounding packages. A recorded compatibility check imported the actual Qwen3-VL class, transcribed a real WAV through Chromium, verified the archived WAV, edited the draft and cancelled the turn. [Browser and Python runtime check](../experiments/dropoff/passenger-whisper-browser-health-20261122.json)

```powershell
& $py -m pip install --only-binary=:all: -r requirements-speech-whisper.txt onnxruntime==1.20.1
& $py -m tools.passenger_whisper fetch --out "$small"
& $py -m tools.passenger_whisper transcribe --model-dir "$small" --wav "your-command.wav"
```

The downloaded files total 486,098,798 bytes. Model revision `d1d751a5f8271d482d14ca55d9e2deeebbae577f` is checked by per-file SHA-256. Recognition uses faster-whisper 1.2.1, CTranslate2 4.6.1, PyAV 15.1.0, int8 on four CPU threads, beam size five and bundled voice activity detection. The English speech model and its conversion are MIT licensed. [Conversion](https://huggingface.co/Systran/faster-whisper-small.en) · [Original model](https://huggingface.co/openai/whisper-small.en) · [Engine documentation](https://github.com/SYSTRAN/faster-whisper)

## Results on public spoken transport requests

One candidate and its decoding settings were fixed before running Whisper. The original 16 people were a comparative development group, because their Vosk results had already motivated the change. A second group used 13 different speakers, source sentences and audio files. One taxi, six travel queries and six ticket requests met its predeclared selection rules. The fixed archive prefix contained insufficient additional eligible taxi speakers to reach the planned 16. [Frozen comparison plan](../experiments/dropoff/passenger-asr-comparison-protocol-20261122.json)

| Measure | Original 16, Vosk | Original 16, Whisper | Fresh 13, Vosk | Fresh 13, Whisper |
|---|---:|---:|---:|---:|
| Word edits / reference words | 84/160 | 30/160 | 49/119 | 23/119 |
| Word error rate | 52.5% | 18.75% | 41.18% | 19.33% |
| Fully correct transcripts | 0/16 | 6/16 | 1/13 | 5/13 |
| Exact place-name mentions recovered in spoken order | 7/20 | 13/20 | 9/18 | 13/18 |
| Clips retaining all place names | 5/16 | 9/16 | 5/13 | 9/13 |
| Median CPU decode | 1.39 s | 3.09 s | 1.39 s | 2.91 s |

Whisper produced fewer errors on 9 of the 13 fresh clips and tied on four. All four conditions returned text for every clip. On the Ryzen 7 4800H, Whisper's standalone process used about 1.05 GiB peak working set and loaded in 8.87 seconds, including model validation and imports. Fresh-clip 95th percentile decode was 3.35 seconds. Named places remained fallible, which makes the passenger's transcript review important. The available references are the corpus's published prompts. Speakers may have differed from the script. These English transport destinations do not directly measure spoken references to nearby EMT landmarks. [Paired results](../experiments/dropoff/passenger-asr-comparison-20261122.json) · [SLURP research paper](https://aclanthology.org/2020.emnlp-main.588/) · [Original SLURP audio archive and licence](https://zenodo.org/records/4274930)

The 30-clip, three-speaker team study is separate. After volunteers contribute, the same consented original WAVs can be scored with Whisper without changing what the participant saw or correcting the recognizer outputs. The report distinguishes the original Vosk display, the volunteer's edit and the new Whisper score. [Team collection instructions](passenger-speech-study.md)

```powershell
& $py -m tools.passenger_speech_study score --study runs/passenger-human-speech-20261122 --whisper-model "$small" --out runs/team-whisper-score.json
```

To reproduce the public comparison, follow the original speech study's SLURP download and selection commands in a fresh local cache. Then run the following commands, setting `$asr` to the local Vosk model. The freeze hashes on a new machine incorporate its selection timestamp. The comparer verifies the source texts, speaker IDs and original WAV identifiers against the original published 16-clip cohort. Both conditions use the same newly frozen fresh cohort.

```powershell
$panel = & $py -m tools.passenger_asr_comparison freeze --cache "$cache" --flac "$flacExe" | ConvertFrom-Json
& $py -m tools.passenger_asr_comparison run --backend vosk --model-dir "$asr" --cache "$cache" --fresh-sha $panel.selection_sha256
& $py -m tools.passenger_asr_comparison run --backend whisper --model-dir "$small" --cache "$cache" --fresh-sha $panel.selection_sha256
& $py -m tools.passenger_asr_comparison report --cache "$cache" --fresh-sha $panel.selection_sha256 --out runs/replayed-asr-comparison.json
```

Original voices, volunteer identifiers and private transcripts stay on the machine. Public experiment files contain hashes and scores. The SLURP audio is CC BY-NC 4.0 and its metadata is CC BY 4.0. [Dataset authors](https://github.com/pswietojanski/slurp)

## Recognition inside moving cars

Whisper made 31 word edits across 242 manually transcribed words in 26 short clips drawn from 13 recordings. Vosk made 93. Whisper preserved 21 of 22 direction words in order and Vosk preserved 17. Both returned text on all 26 clips and both added one extra direction word. Whisper had fewer word edits on 23 clips and tied on three. Both recognizers processed the same original audio locally, restricted to one effective CPU core. [Paired results, counts and audio hashes](../experiments/dropoff/passenger-pursuit-speech-results-20261122.json)

The PURSUIT corpus records people describing their route during seven drives in Pensacola, Florida, typically from both the lead and follow cars. We froze two utterances per original recording before recognition, one containing directional language and one containing an annotated location reference. The 13 recording identifiers describe cars and drives. We did not count unique speakers. The authors manually transcribed the words and estimated their timing within utterances. Approximate boundaries can include adjacent speech or cut off words. The separate team recording protocol evaluates spoken project instructions tied to EMT images. [Original corpus and citation](https://www.cs.rochester.edu/research/speech/pursuit/) · [Corpus documentation](https://www.cs.rochester.edu/research/speech/pursuit/pursuit.pdf) · [Frozen selection and scoring rules](../experiments/dropoff/passenger-pursuit-speech-protocol-20261122.json)

| Recording | Vosk word error rate | Whisper word error rate | Direction words retained by Vosk | Direction words retained by Whisper |
|---|---:|---:|---:|---:|
| s1.foll | 23.8% | 19.0% | 1/1 | 1/1 |
| s1.lead | 36.4% | 9.1% | 3/3 | 3/3 |
| s2.foll | 41.2% | 23.5% | 1/2 | 2/2 |
| s2.lead | 58.8% | 17.6% | 1/1 | 1/1 |
| s3.lead | 29.4% | 5.9% | 0/1 | 1/1 |
| s4.foll | 50.0% | 25.0% | 3/3 | 3/3 |
| s4.lead | 52.6% | 15.8% | 2/2 | 2/2 |
| s5.foll | 5.9% | 0.0% | 1/1 | 1/1 |
| s5.lead | 20.0% | 5.0% | 0/1 | 1/1 |
| s6.foll | 25.0% | 6.2% | 1/2 | 2/2 |
| s6.lead | 52.2% | 4.3% | 0/1 | 0/1 |
| s7.foll | 35.3% | 5.9% | 2/2 | 2/2 |
| s7.lead | 68.8% | 31.2% | 2/2 | 2/2 |
| All 26 clips | **38.4%** | **12.8%** | **17/22** | **21/22** |

For reproduction, place both publisher ZIPs in a fresh private folder and set `$cache` to that folder, `$asr` to the installed Vosk model and `$small` to the Whisper model. The measured local environment used `psutil 7.2.2` to restrict CPU affinity. The final command writes only hashes, aggregate measurements, per-recording counts and per-clip scores. [Audio archive](https://www.cs.rochester.edu/research/speech/pursuit/PURSUIT-1.0.zip) · [Annotation archive](https://www.cs.rochester.edu/research/speech/pursuit/PURSUIT-1.0-noaudio.zip)

```powershell
$chosen = & $py -m tools.passenger_pursuit_speech choose --cache "$cache" | ConvertFrom-Json
$bound = & $py -m tools.passenger_pursuit_speech freeze --cache "$cache" --selection-sha $chosen.annotation_selection_sha256 | ConvertFrom-Json
& $py -m tools.passenger_pursuit_speech run --cache "$cache" --bound-sha $bound.clip_binding_sha256 --backend vosk --model-dir "$asr"
& $py -m tools.passenger_pursuit_speech run --cache "$cache" --bound-sha $bound.clip_binding_sha256 --backend whisper --model-dir "$small"
& $py -m tools.passenger_pursuit_speech report --cache "$cache" --bound-sha $bound.clip_binding_sha256 --out runs/replayed-pursuit-scores.json
```
## Spoken instructions that describe a place

Sixteen different speakers in the HCRC Map Task gave spontaneous directions to a partner. The team selected one excerpt per speaker and dialogue before running speech recognition. Eight referred to a place between landmarks. The other eight described a location another way. Scottish map directions supply human speech and named landmarks. The team’s Gulf passenger recordings use a separate collection form. [HCRC audio and transcripts](https://groups.inf.ed.ac.uk/maptask/maptasknxt.html) · [Team recording form](passenger-speech-study.md)

English Whisper Small made 38 word edits in 269 reference words (14.1%). Seven of the 16 transcripts matched the normalized reference exactly. It retained 20 of 21 spatial-relation words, including all eight occurrences of “between”, and 18 of 23 named landmark mentions. All 16 transcripts were editable in the passenger application. The browser displayed the original recognition result again for one clip after its inspection connection lost the response body. Each clip had one genuine recognition result. The study ended at transcript review, before a destination was requested. [Recorded results, model revision and source details](../experiments/dropoff/passenger-external-speech-20261122.json)

To repeat the study, install the Whisper runtime, the passenger release bundle and its road-model cache. Prepare the frame 781 bookmark. Use Node with Playwright and its installed Chromium browser. A custom install can use `TRACKRAG_PLAYWRIGHT` for the Playwright module path and `TRACKRAG_CHROMIUM` for the browser executable. Run from the repository root with a fresh private working directory. [Frame 781 installation](passenger-pair-bookmark.md) · [Passenger release installation](passenger-release.md)

```powershell
$py = ".venv-grounding-local/Scripts/python.exe"
$small = Join-Path $env:LOCALAPPDATA "TrackRAG-method-cache/faster-whisper-small.en"
$study = "runs/hcrc-speech-replication"
$scene = "runs/passenger-pair-scene/requests.jsonl"
$selection = & $py -m tools.passenger_external_speech_evaluation freeze --work $study | ConvertFrom-Json
$audio = & $py -m tools.passenger_external_speech_evaluation audio --work $study --sha $selection.cohort_sha256 | ConvertFrom-Json
& $py -m tools.passenger_external_speech_evaluation app --work $study --sha $audio.bound_sha256 --scene-bank $scene --model $small --vision-cache models/passenger-support/segformer-b0
& $py -m tools.passenger_external_speech_evaluation score --work $study --sha $audio.bound_sha256 --scene-bank $scene --out "$study/results.json"
```
