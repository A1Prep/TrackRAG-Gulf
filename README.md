# TrackRAG-Gulf

TrackRAG-Gulf connects two questions about a recorded UAE road scene. A reviewer can ask what happened and open the frames cited in the answer. A passenger can describe where they mean to get out, inspect a proposed point, correct it and confirm it. The demonstration operates on paused recordings and photographs. The pin records passenger intent. The driver decides where stopping is permitted and practical.

## Run the demonstration

On the prepared project laptop, run this from the repository root and leave the terminal open.

```powershell
.\.venv-grounding-local\Scripts\python.exe -m tools.passenger_release start --port 8775 --require-footage 2 --low-memory --parking-backend modal --public-places --vision-cache runs/model-cache/airport-segformer-b0
```

- **Review footage.** Select recording 054604, frame 150. Ask “What colour is the bus visible on the left near the end?” Choose the Gemini answer mode and open a cited frame. Then ask for its speed in kilometres per hour. The camera supplies visible colour, but this clip provides no calibrated speed measurement. [Open footage search](http://127.0.0.1:8775/footage)
- **Try a passenger request.** On the shop-front photo, enter “Please drop me off in the marked space between the dark blue sedan and the silver SUV outside the shops on the right.” Inspect the pin. If the proposed spot misses the intended patch, mark the patch, check it and confirm the updated proposal. English speech uses the same flow after the passenger reviews the transcript. [Open the Dubai example](http://127.0.0.1:8775/?scene=f4d997a6f22c942559a798dd)

On a fresh machine, install Python 3.11, restore the private adapter and saved predictions, and prepare the two original EMT recordings. Install the UAE photo bank for the shop-front example. The general passenger model runs on a compatible NVIDIA GPU. English speech and local footage retrieval run on CPU. Hosted footage answers use a Gemini API key, and the automatic two-vehicle specialist uses a configured Modal account. The laptop flag releases the local 8B worker after each inference. [Fresh-machine installation](docs/passenger-release.md#start-on-another-windows-machine) · [UAE photos, cloud setup and attribution](docs/passenger-parking-feature.md)

## What the team built

| Component | Implementation and evidence |
| --- | --- |
| Vision · MAAI 901 | RF-DETR Large fine-tuned on EMT, followed by BoT-SORT tracking. On the five official test recordings, road-user detection AP improved from 0.303 to 0.355 and IDF1 from 0.400 to 0.443. Identity switches increased. [Training, augmentations and full results](docs/task1-vision.md) |
| Footage QA · MAAI 902 | The retriever ranks track intervals and scene evidence up to the question frame using text and temporal features. It exports original-frame citations. In hosted mode Gemini receives privacy-processed evidence and can answer or report insufficient evidence. [Retrieval and evaluation](docs/temporal-evidence.md) · [Handling footage](docs/privacy.md) |
| Passenger language grounding · MAAI 902 | A locally fine-tuned Qwen3-VL-8B interprets general English descriptions. A learned location estimator and road evidence place the proposal. A separate pretrained Qwen3.5-9B and Mask2Former handle the optional two-vehicle parking route. Whisper Small transcribes English for review. [8B training and independent results](docs/passenger-independent-comparisons.md) · [Measured parking successes and misses](docs/passenger-parking-feature.md) |

## Verify the project

Run the automated suite in the development environment with `python -m pytest tests -q`. To replay the 36 development questions through the running application, use the command below. It opens and hashes cited images and checks the original recording and time boundary for each response. Omitting `--hosted` allows local retrieval without sending images.

```powershell
python -m tools.footage_acceptance --questions experiments/temporal-development-questions.jsonl experiments/temporal-second-questions.jsonl --hosted --out runs/footage-replay
```

The release record includes the 36-question footage replay, the passenger microphone and WAV checks, location accuracy on unseen photos and the remaining participant work. [End-to-end results and remaining work](docs/project-release.md) · [Assessment criteria](docs/assessment-alignment.md) · [Five task owners and handoffs](docs/final-completion-part2.md)
