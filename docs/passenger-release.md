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

The passenger speaks English or types. Vosk transcribes audio on the laptop. Qwen3-VL-8B sees the original frame, the passenger's words and current detector boxes. For an explicitly named pair, it also resolves each quoted description against the same image. A learned referent ranker helps identify requests that still match multiple vehicles. A separately fitted language-conditioned model estimates a destination relative to the resolved landmarks. Predicted road support and exclusions for current road users constrain the point. The passenger makes the final choice. [Measured browser run](../experiments/dropoff/passenger-release-browser-results-20261122.json)

The dot is one proposed road location. The rectangular outline estimates the passenger's requested area and can include vehicle or dashboard pixels. This recorded-scene application interprets passenger intent. Moving a vehicle, stopping legality and physical clearance are outside its scope.

## Search the footage

Choose **Search footage**, select the latest permitted frame and enter a question. **Find evidence locally** returns ranked intervals and clickable original-frame evidence. **Ask Gemini** sends privacy-processed evidence and returns a cited answer, an insufficient-evidence result or a provider error. Untimed object queries use Task 4's lexical track-window baseline. Questions about the opening, the end or a named frame retrieve those original scenes. Boundary questions also retrieve the observed track history and the preceding or following scene when available. Task 3's answerer checks the cited times. The CLI exposes the same profile through `--retrieval-adapter trackrag.retrieval_temporal`. The release defaults to Gemini 3.5 Flash-Lite for footage questions when no model was configured. Supply your existing `GEMINI_API_KEY` for hosted answers. [Original 36-question baseline](../experiments/dropoff/passenger-qa-full36-20261122.json)

## Start on another Windows machine

Use Python 3.11 and the CUDA environment instructions for the pinned Qwen release. From the repository root in that activated environment, install the speech, recording and optional hosted-answer dependencies. The base model download is around 7.24 GB. [Local 8B installation](dropoff-local-training.md)

```powershell
python -m pip install -r requirements-speech.txt -r requirements-hosted.txt -c requirements-grounding-local.txt "opencv-python-headless==4.11.0.86"
python -m tools.dropoff_prepare_streamed_model
python -m tools.passenger_release fetch-support
gh release download passenger-feature-20261122 --repo A1Prep/TrackRAG-Gulf --pattern passenger-feature.zip --dir models
python -m zipfile -e models/passenger-feature.zip models/passenger-feature
python -m tools.passenger_release verify
```

The private release carries the exact saved predictions for the two demonstration recordings. Set `$emtImages` to your team's original EMT root containing `images/train` (the reader also accepts the official split-first layout). The commands below unpack the two tracks files and their provenance, prepare the ten demo frames and precompute the road masks to keep GPU memory available. [Official EMT dataset](https://huggingface.co/datasets/KuAvLab/EMT/tree/4418cec01b03c6f797cadb06c9405843a30988e2) · [Prediction verification](../experiments/dropoff/passenger-track-handoff-20261122.json)

```powershell
gh release download passenger-feature-20261122 --repo A1Prep/TrackRAG-Gulf --pattern passenger-finetuned-emt-tracks-20261122.zip --dir models
$savedPredictions = "models/passenger-demo-predictions"
python -m zipfile -e models/passenger-finetuned-emt-tracks-20261122.zip "$savedPredictions"
python -m tools.dropoff_prepare_scene --data-root "$emtImages" --predictions "$savedPredictions" --vision-config configs/vision/finetuned.json --instructions configs/dropoff/passenger-demo-scenes.jsonl --out runs/passenger-inputs
python -m tools.dropoff_prepare_passenger_ground --requests runs/passenger-inputs/requests.jsonl --vision-cache models/passenger-support/segformer-b0 --report runs/passenger-road-preparation.json
python -m tools.passenger_release start --qa-data-root "$emtImages" --qa-predictions "$savedPredictions"
```

Keep the files in `images` alongside the prepared `requests.jsonl` if moving the scene bank. Use the original recordings for footage search and the saved predictions from the same run that prepared the scenes. A supplied EMT subset must include the recording prefix through the requested frame. [Scene preparation](dropoff-scene-inputs.md) · [Private release files](https://github.com/A1Prep/TrackRAG-Gulf/releases/tag/passenger-feature-20261122)

## Evaluation and review

The released passenger workflow uses the adapter trained on 500 public examples. The separate locally continued adapter used 17 AI-authored Gulf instructions for another two epochs. Its release asset and results support a direct-generation comparison. Each condition has its own receipts and weight hashes. [Frozen passenger comparison](../experiments/dropoff/passenger-release-protocol-20261122.json) · [Adapter training and reload](../experiments/dropoff/passenger-native-child-reload-20261121.json)

| Independent test | Released 8B selector | Pretrained 8B selector |
| --- | ---: | ---: |
| Correct propose, clarify or missing decision on 20 nighttime Gulf requests | 15/20 | 15/20 |
| Point inside the requested area on 12 clear nighttime instructions | 7/12 | 5/12 |
| Proposed an area despite an unresolved nighttime instruction | 1/8 | 1/8 |
| Point inside the original participant mask on 16 unseen public scenes | 1/16 | 3/16 |

The Gulf figures use AI-authored regions on six frames from one recording. The public figures use original human-marked destinations from 16 different scenes without detector proposals. Both model conditions used the same learned region estimator and surface checks. The pretrained public run had one timeout, retained in the denominator. Precise placement varies considerably across scenes. The passenger can clarify, mark a correction or decline. [Independent scoring and per-condition counts](../experiments/dropoff/passenger-external-independent-audit-20261122.json)

Two people can independently label the 17 daytime training and 19 development cases from blank review forms. Each reviewer marks a point and area, asks for clarification or records missing evidence before seeing any predictions. The data owner can import their signed exports, resolve disagreements and validate the resulting files for the next local training run. [Review forms and import commands](passenger-human-intake.md)
