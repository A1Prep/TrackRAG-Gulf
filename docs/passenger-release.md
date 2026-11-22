# Run the passenger drop-off and footage review application

From the prepared laptop checkout, run the following command and open the local page. Both sections start with ten bookmarked scenes. Select **Open another frame in this recording** to use any original frame covered by the same saved detector predictions. Each of the two example recordings has 150 frames. The first passenger request loads the 8B model.

```powershell
.\.venv-grounding-local\Scripts\python.exe -m tools.passenger_release start
```

[Open the local application](http://127.0.0.1:8765/)

## Try the passenger interaction

1. Select recording **054907**, frame **1**. Choose **Record voice** and say “Drop me here.” Check the recognized words and send them. The assistant should ask for a landmark.
2. Reply “I'd like the patch of road just behind the white SUV on the left.” If the recognizer hears “as you v,” edit it to “SUV” before sending it. The screen shows the proposed point and estimated area. Confirm it, cancel it, or mark a correction.
3. Start a new request on the same frame. Try “I mean the open road between the white SUV on our left and the large open-sided truck on our right, just behind them.” The two resolved vehicle positions constrain the proposed road point.
4. In recording **054604**, frame **150**, try “Drop me beside the yellow bus on the left.” A confirmation applies to the exact proposal on the displayed frame. Changing the proposal invalidates its previous confirmation.

The passenger speaks English or types. The prepared laptop uses Whisper Small in English on its CPU and shows the transcript for editing. Qwen3-VL-8B sees the original frame, the passenger's words and current detector boxes. For an explicitly named pair it resolves each description against the same image. A learned referent ranker helps identify requests that still match multiple vehicles. A separately fitted model estimates a destination relative to the resolved landmarks. Predicted road pixels and the positions of visible road users constrain the point. The passenger makes the final choice. [Measured browser grounding run](../experiments/dropoff/passenger-release-browser-results-20261122.json)

The dot is one proposed road location. The rectangular outline estimates the passenger's requested area and can include vehicle or dashboard pixels. This recorded-scene application interprets passenger intent. Moving a vehicle, stopping legality and physical clearance are outside its scope.

## Search the footage

Choose **Search footage**, select the latest permitted frame and enter a question. **Find evidence locally** returns ranked intervals and clickable original-frame evidence. **Ask Gemini** sends privacy-processed evidence and returns a cited answer, an insufficient-evidence result or a provider error. Untimed object queries use Task 4's lexical track-window baseline. Questions about the opening, the end or a named frame retrieve those original scenes. Boundary questions also retrieve the observed track history and the preceding or following scene when available. Task 3's answerer checks the cited times. The CLI exposes the same profile through `--retrieval-adapter trackrag.retrieval_temporal`. The release defaults to Gemini 3.5 Flash-Lite for footage questions when no model was configured. Supply your existing `GEMINI_API_KEY` for hosted answers. [Original 36-question baseline](../experiments/dropoff/passenger-qa-full36-20261122.json)

At frame 150 in recording 054907, “What colour is the prominent SUV on the right near the end?” retrieved frame 150 and Gemini answered **black**. In 054604, the model identified the **yellow bus on the left** and explained that the supplied views could not establish the boundary crossing. Across 36 previously used development questions, independent review graded 19 answers fully supported, 13 qualified and three missed. One provider error remains in those counts. A separately recorded retry answered that question. Expected evidence appeared in the top three results for 20 of 25 scorable retrieval cases, compared with 10 of 25 for the original search. [Paired before-and-after results](../experiments/dropoff/passenger-qa-final36-20261122.json)

## Start on another Windows machine

Use Python 3.11 and the CUDA environment instructions for the pinned Qwen release. From the repository root in that activated environment, install the speech, recording and optional hosted-answer dependencies. The base model download is around 7.24 GB. [Local 8B installation](dropoff-local-training.md)

```powershell
python -m pip install -r requirements-speech.txt -r requirements-hosted.txt -c requirements-grounding-local.txt "opencv-python-headless==4.11.0.86"
python -m tools.dropoff_prepare_streamed_model
python -m tools.passenger_release fetch-support
python -m pip install -r requirements-speech-whisper.txt -c requirements-grounding-local.txt
python -m tools.passenger_whisper fetch --out models/passenger-support/faster-whisper-small.en
gh release download passenger-feature-20261122 --repo A1Prep/TrackRAG-Gulf --pattern passenger-feature.zip --dir models
python -m zipfile -e models/passenger-feature.zip models/passenger-feature
python -m tools.passenger_release verify
```

The private release carries the saved detector predictions for the two demonstration recordings. The image installer fetches only their first 150 frames each from the official EMT archive, about 283 MiB in total. It checks all 300 JPEGs against Task 1's published original-frame hashes and resumes from any frames already verified. If your team has the EMT dataset, set `$emtImages` to its root and skip the image download. [Official EMT dataset](https://huggingface.co/datasets/KuAvLab/EMT/tree/4418cec01b03c6f797cadb06c9405843a30988e2) · [Prediction verification](../experiments/dropoff/passenger-track-handoff-20261122.json)

```powershell
$emtImages = "data/passenger-demo-images"
python -m tools.passenger_demo_images --out "$emtImages"
gh release download passenger-feature-20261122 --repo A1Prep/TrackRAG-Gulf --pattern passenger-finetuned-emt-tracks-20261122.zip --dir models
$savedPredictions = "models/passenger-demo-predictions"
python -m zipfile -e models/passenger-finetuned-emt-tracks-20261122.zip "$savedPredictions"
python -m tools.dropoff_prepare_scene --data-root "$emtImages" --predictions "$savedPredictions" --vision-config configs/vision/finetuned.json --instructions configs/dropoff/passenger-demo-scenes.jsonl --out runs/passenger-inputs
python -m tools.dropoff_prepare_passenger_ground --requests runs/passenger-inputs/requests.jsonl --vision-cache models/passenger-support/segformer-b0 --report runs/passenger-road-preparation.json
python -m tools.passenger_release start --qa-data-root "$emtImages" --qa-predictions "$savedPredictions"
```

With the default directories shown above, subsequent launches need only `python -m tools.passenger_release start`. Keep the files in `images` alongside the prepared `requests.jsonl` if moving the scene bank. A custom EMT subset must include the recording prefix through the requested frame. A fresh download and the released track files reproduced all ten original demonstration inputs. [Image and scene verification](../experiments/dropoff/passenger-demo-image-verification-20261122.json) · [Private release files](https://github.com/A1Prep/TrackRAG-Gulf/releases/tag/passenger-feature-20261122)

The launcher selects Whisper when its runtime and local model are installed. To explicitly choose a recognizer, supply `--whisper-model <folder>` or `--asr-model <Vosk-folder>`. A Vosk-only installation also supports editable voice input. Use `--text-only` to start with typing.

## Evaluation and review

The released passenger workflow uses the adapter trained on 500 public examples. The separate locally continued adapter used 17 AI-authored Gulf instructions for another two epochs. Its release asset and results support a direct-generation comparison. Each condition has its own receipts and weight hashes. [Frozen passenger comparison](../experiments/dropoff/passenger-release-protocol-20261122.json) · [Adapter training and reload](../experiments/dropoff/passenger-native-child-reload-20261121.json)

| Independent test | Released 8B selector | Pretrained 8B selector |
| --- | ---: | ---: |
| Correct propose, clarify or missing decision on 20 nighttime Gulf requests | 15/20 | 15/20 |
| Point inside the requested area on 12 clear nighttime instructions | 7/12 | 5/12 |
| Proposed an area despite an unresolved nighttime instruction | 1/8 | 1/8 |
| Point inside the original participant mask on 16 unseen public scenes | 1/16 | 3/16 |

The Gulf figures use AI-authored regions on six frames from one recording. The public figures use original human-marked destinations from 16 different scenes without detector proposals. Both model conditions used the same learned region estimator and surface checks. The pretrained public run had one timeout, retained in the denominator. Precise placement varies considerably across scenes. The passenger can clarify, mark a correction or decline. [Independent scoring and per-condition counts](../experiments/dropoff/passenger-external-independent-audit-20261122.json)

Arjun’s corrected handoff contains four training and two development photographs with two recorded reviewer confirmations. Each repeats the same ambiguous request and follow-up question. The team kept the 44 approved final cases and six cases awaiting reconfirmation outside all training and model selection. Another set of blank forms covers 17 daytime training and 19 development instructions, including proposed stopping areas and missing landmarks. [Corrected package, review forms and reproduction commands](passenger-human-intake.md)

For the passenger voice study, three consenting speakers each read ten frozen scene instructions. The recorder keeps the original audio, the words spoken, the raw transcription and any passenger correction. The scorer reports word error rate and errors in colours, directions, counts and negation by speaker. [Voice study instructions](passenger-speech-study.md)

On 13 previously unused public transport speakers, Whisper made 23 word edits in 119 reference words (19.3%). The earlier Vosk recognizer made 49 (41.2%). Whisper retained 13 of 18 place names and Vosk retained nine. The browser shows every transcript for correction. The formal EMT passenger recordings are a separate study. [Speech comparison and recording setup](passenger-whisper.md)
