# TrackRAG-Gulf

TrackRAG-Gulf connects two questions about a recorded UAE road scene. A reviewer can ask what happened and open the frames cited in the answer. A passenger can describe where they mean to get out, inspect a proposed point, correct it and confirm it. The demonstration operates on paused recordings and photographs. The pin records passenger intent. The driver decides where stopping is permitted and practical.

## Run the demonstration

To watch from another computer, sign in to GitHub with an account that has access to this private repository. The four-minute recording shows both footage review and passenger drop-off. Download the MP4 and English captions to watch offline. A lecturer can also inspect the passenger’s words, the two detected vehicles and the proposed point in a browser screenshot. The recording and caption file can accompany a course submission for an examiner without repository access. [Walkthrough and captions](https://github.com/A1Prep/TrackRAG-Gulf/releases/tag/assessment-3-demo-20261123) · [Passenger confirmation screen](docs/passenger-ux-desktop.jpg)

The interactive links below open the application on the computer running it. To run it on another Windows machine, follow the installation guide first. [Fresh-machine installation](docs/passenger-release.md#start-on-another-windows-machine)

On the prepared Windows laptop, double-click **Start TrackRAG.cmd**. It opens Search footage in the browser and checks that both recordings are available. It reuses the running app or starts it in the background. The passenger page is in the top navigation. [One-minute presentation rehearsal](docs/passenger-release.md#one-minute-presentation-rehearsal)

For a foreground server, run this from the repository root and leave the terminal open.

```powershell
.\.venv-grounding-local\Scripts\python.exe -m tools.passenger_release start --port 8775 --require-footage 2 --low-memory --parking-backend modal --public-places --vision-cache runs/model-cache/airport-segformer-b0
```

- **Watch and search.** Open the bus recording and select “When did the bus first appear?” The local answer links to frame 109 at 10.8 seconds. Click the evidence thumbnail to jump to that frame with the bus outlined. To ask about a street feature, choose Gemini under **Search settings**. [Open the bus recording](http://127.0.0.1:8775/footage?recording=video_054604&frame=118) · [Try the other recording](http://127.0.0.1:8775/footage?recording=video_054907&frame=54)
- **Try a passenger instruction.** On the hotel photo, type or say “Could you drop me behind the white car?” English speech opens an editable transcript. Inspect the outlined SUV and the gold ring behind it before saving the location. On the white-sedan photo, the image checks disagree about the lane edge, so the app offers a point for the passenger to review. [Hotel photo](http://127.0.0.1:8775/?scene=b478c884c0832f9ea6d0fc05) · [White sedan](http://127.0.0.1:8775/?scene=10e8284093286234e0bdd0df)
- **Try two plausible spots.** Ask “Could you drop me next to the restaurant?” The photo shows two marked bays beside the frontage. Choose an outlined bay and confirm, or start again and tap the paint shared by the two bays. Tapping the traffic lane in the same photo withholds the destination. [Restaurant photo](http://127.0.0.1:8775/?scene=4edc9abbff73676b9a63bc1d) · [Another marked roadside bay](http://127.0.0.1:8775/?scene=ef1207ebb0cf2f79ec0eb895)

On a fresh machine, install Python 3.11, restore the private adapter and saved predictions, and prepare the two original EMT recordings. Install the UAE photo bank for the shop-front example. The general passenger model runs on a compatible NVIDIA GPU. English speech and local footage retrieval run on CPU. Hosted footage answers use a Gemini API key. Fresh calls to the deployed two-vehicle parking service require access to the team’s private Modal workspace. The laptop flag releases the local 8B worker after each inference. [UAE photos, cloud setup and attribution](docs/passenger-parking-feature.md)

## What the team built

| Component | Implementation and evidence |
| --- | --- |
| Vision · MAAI 901 | RF-DETR Large fine-tuned on EMT, followed by BoT-SORT tracking. On the five official test recordings, road-user detection AP improved from 0.303 to 0.355 and IDF1 from 0.400 to 0.443. Identity switches increased. [Training, augmentations and full results](docs/task1-vision.md) |
| Footage QA · MAAI 902 | The retriever ranks track intervals and scene evidence up to the question frame using text and temporal features. It exports original-frame citations. In hosted mode Gemini receives privacy-processed evidence and can answer or report insufficient evidence. [Retrieval and evaluation](docs/temporal-evidence.md) · [Handling footage](docs/privacy.md) |
| Passenger language grounding · MAAI 902 | A fine-tuned Qwen3-VL-8B interprets passenger descriptions and supplies visual observations. Mask2Former labels surfaces. Original pixels, detected vehicles and visible markings help locate the intended ground. The prepared gallery also uses photo-specific observations from Qwen3.6-35B-A3B. Qwen3.5-9B handles the optional two-vehicle parking route. Whisper Small transcribes English for passenger review. [8B comparison](docs/passenger-independent-comparisons.md) · [Parking experiments](docs/passenger-parking-feature.md) |

## Verify the project

Run the automated suite in the development environment with `python -m pytest tests -q`. To replay the 36 development questions through the running application, use the command below. It opens and hashes cited images and checks the original recording and time boundary for each response. Omitting `--hosted` allows local retrieval without sending images.

```powershell
python -m tools.footage_acceptance --questions experiments/temporal-development-questions.jsonl experiments/temporal-second-questions.jsonl --hosted --out runs/footage-replay
```

The release record includes the 36-question footage replay, the passenger microphone and WAV checks, location accuracy on unseen photos and the remaining participant work. [End-to-end results and remaining work](docs/project-release.md) · [Assessment criteria](docs/assessment-alignment.md) · [Five task owners and handoffs](docs/final-completion-part2.md)
