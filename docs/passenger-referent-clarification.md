# Passenger clarification when several vehicles fit

A passenger who says “behind that white vehicle” may be looking at several cars, vans or trucks. The ranker uses the instruction and the visible detector boxes to list possible objects. The clarification policy can ask which one the passenger means. Qwen handles the instruction and the destination estimate. The passenger confirms the proposed area in the application.

Two kinds of evidence support different claims. The public benchmark measures which object the model ranked highest. The EMT pilot records which questions the policy would ask when driving-scene inputs use the project's own detector and tracker.

## Training and evaluation

The model learned from 500 commands in 295 scenes. Each training command provided the original camera image, the passenger's text and up to 64 predicted CenterNet boxes. The original Talk2Car referring-object annotation supervised the fit. All proposals that overlapped the reference by at least 0.50 intersection over union counted as positives, weighted by overlap. The remaining proposals were competing objects in the same image. When the detector missed the reference, the training target was a separate “no matching proposal” option. The model retained all 19 training misses. [Released detector data](https://github.com/kanji1011/Talk2car-Refseg/tree/cebce1943ce373857cc9be5def68ded4008c29e1/dataloader) · [Original referring-object data](https://github.com/talk2car/Talk2Car/blob/3fde52aee3341312196ab9b2f999337eff0693a0/data/commands/train_commands.json)

Each candidate is represented by its center, extent, position among the detections, detector category and colours sampled inside its box. A listwise logistic model learns how wording changes the score of each candidate. Its 5,381 fitted coefficients include interactions with 130 training-vocabulary terms and class, colour and directional agreement features. A second fit used the same images, boxes, labels and objective with the language inputs removed. Both fits completed before scoring the 125 development commands from 65 disjoint scenes.

| Public development measure | With language | Image and boxes only |
|---|---:|---:|
| Intended object ranked first | 61/125 (48.8%) | 39/125 (31.2%) |
| Intended object in the first three distinct candidates | 84/125 (67.2%) | 65/125 (52.0%) |
| Mean intersection over union of the top object | 0.416 | 0.281 |
| Correct top selection when the “no match” option can also win | 60/125 | 39/125 |
| Detector misses correctly identified | 0/2 | 0/2 |

The detector supplied a matching candidate in 123 of the 125 commands. Language gained 26 correct top choices and lost four relative to the image-only fit. The difference was 17.6 percentage points. A paired scene-level bootstrap over the 65 development scenes gave a 95% percentile interval of 8.6 to 26.7 points.

Running all 125 images through the application-facing rank method reproduced every individual top-object overlap from the benchmark. Replacing each instruction with a different image's command reduced the top-object result to 37/125. There were six same-image pairs with different annotated referents. Changing only the command changed the top box in all six pairs, and only one pair had both top boxes correct. The confidence scores are uncalibrated. Passenger ambiguity and stopping suitability require separate evaluation.

## Clarification policy

The policy considers current, spatially distinct detections with visible extent. It checks whether the passenger supplied an identifying type, side, size or colour. If several candidates remain, it can ask about their left–right or higher–lower positions in the image. With weak colour evidence under shadow, it can ask which vehicle without making a claim about its paint. The broad EMT labels for medium and large vehicles support a question about a *vehicle*. A policy response contains the actual competing track IDs. Specific subtypes, complex landmark relations and two-object instructions go to the primary visual-language workflow.

A first, input-only pilot on 19 EMT development instructions asked one erroneous question about the yellow car in “Behind the black SUV to the right of the yellow car.” The policy had overlooked the primary noun “SUV.” The revision recognizes a second subtype noun and leaves this relation to Qwen. It also requests clarification when a simple vehicle description depends on uncertain colour or coarse detections.

The revised policy asked on three of those 19 instructions and deferred the other 16. It asked which vehicle was meant by “that white vehicle,” which of two separated vehicle detections was meant by “the truck,” and which visible car was meant by “that car under the bridge.” The revised numbers are a development-informed replay on one recording. No passenger labels were used to grade these policy outcomes. The first pilot predictions and the revised decisions are retained in the private experiment bundle.

## Runtime and reproduction

Load the coefficients from the installed passenger release and call the ranker with the original image already verified by the passenger application. Internally it enforces the predicted-track schema, current recording and frame dimensions, rejects future observations and ignores earlier frames. A response from the policy is an optional clarification. It supplies no destination coordinates. The package manifest binds the released weights to their SHA256 digest. Both original fitted heads are included in the separate experiment archive.

~~~python
from trackrag.dropoff.passenger import instruction_context
from trackrag.dropoff.referent import ReferentRanker
from trackrag.dropoff.referent_policy import clarify_reference

model = ReferentRanker.load(
    "models/passenger-feature/referent.json",
    expected_sha256="d96a9a69e2e52ddbc21851d5ff6f79b344de6c8558600e54be5cb2712480a07d",
)
text = instruction_context(request)
ranking = model.rank(image, text, request["scene"]["tracks"], frame=request["scene"]["frame"])
clarification = clarify_reference(text, ranking, image.size)
~~~

The fit uses NumPy, SciPy and Pillow. The ranker and clarification policy at inference require NumPy and Pillow. Rebuild the two public splits using the dataset instructions. Export the pinned predictions and run the supervised fit into a fresh output directory.

~~~text
python -m tools.dropoff_public_candidates --splits data/dropoff-public-expanded-20261121/splits --out runs/referent-proposals.json
python -m tools.dropoff_fit_referent --splits data/dropoff-public-expanded-20261121/splits --metadata runs/passenger-landmarks/train_commands.json --proposals runs/referent-proposals.json --out runs/referent-reproduction
python -m pytest -q tests/test_dropoff_referent.py tests/test_dropoff_referent_policy.py
~~~

The original source-file digest and the canonical subset digest both appear in the [machine-readable benchmark](../experiments/dropoff/passenger-referent-results-20261122.json). Re-exported proposal JSON can have a different file digest while representing the exact same pinned detector bank. The fit stores the input file's digest in the checkpoint. [Training tool](../tools/dropoff_fit_referent.py) · [Runtime controls](../tools/dropoff_referent_controls.py) · [EMT policy pilot](../tools/dropoff_referent_policy_pilot.py)
