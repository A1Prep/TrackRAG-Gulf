# Open the two-black-cars passenger scene

The optional bookmark opens recording 054907, frame 781 in the passenger interface. It pairs the original development photograph from Task 2 with 23 current-frame observations from Task 1's fine-tuned detector. The importer verifies the photograph, the detector identity and the complete saved prediction files against their registered hashes.

From a full private project checkout, download the two saved prediction files for this recording. The installer reads the original development JPEG from the team's Task 2 Git commit and places a local copy in the ignored runs directory. The small release archive was verified against the full original Task 1 ZIP.

~~~powershell
gh release download passenger-feature-20261122 --repo A1Prep/TrackRAG-Gulf --pattern passenger-pair-recording-predictions.zip --dir models
python -m tools.passenger_pair_scene --predictions models/passenger-pair-recording-predictions.zip
~~~

An extracted predictions folder also works. If your checkout only has the default branch, run `git fetch origin Task-2-arjun-part-2` to retrieve the team's original development photograph. You can also supply **--image** followed by the path to the original **00781.jpg**. The importer checks the same image hash either way.

Start the app after preparing the bookmark. The launcher picks up the new photo alongside the ten existing bookmarks.

~~~powershell
.\.venv-grounding-local\Scripts\python -m tools.passenger_release start
~~~

Choose recording **054907**, frame **781** in the passenger view. Type or record “Drop me between these two black cars.” Check the transcript and submit it. In this frame there are several dark vehicles, so the app asks which two you mean.

1. Choose **Choose two vehicles**. Click the small dark sedan next to the median on the left and the sedan immediately to its right. The numbered dropdowns also work with a keyboard.
2. Try **Between the vehicles** and **Show a road location**. Their outlines overlap in this camera view. The app asks for a different placement.
3. If the patch immediately behind them matches your intent, choose the same two vehicles, select **Just behind the vehicles**, and choose **Show a road location**. The learned location model estimates a point and checks the surrounding patch against the road mask and current detections. Your action identifies the two landmarks. The model supplies the destination.
4. Inspect the displayed point and choose **Confirm this area** or **Cancel request**. You can also describe a change, or choose **Mark an area myself** to draw a different road patch. Each replacement needs its own confirmation.

This bookmark adds one still photograph for the passenger feature. The packaged footage search continues to cover the first 150 frames of each of its two recordings. Frame 781 has no attached intervening footage. The private local folder contains the photo, the scene bank and an image-free source receipt. The converter reads no target labels or example passenger answers.

A fresh private clone downloaded both released archives, fetched the original photo from Task 2, and reproduced all 23 vehicle observations and the same proposed point. It shared the project laptop's Python environment, base model and speech and road caches. A 390-pixel-wide browser also completed the selection and cancellation through the main application. [Installation and browser measurements](../experiments/dropoff/passenger-pair-install-20261122.json)
