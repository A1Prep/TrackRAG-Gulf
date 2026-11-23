# Passenger drop-off at UAE storefronts and driveways

On 16 new UAE drives, the released app placed its first pin inside **0 of 8 visible targets** and asked for clarification on **all 8 blocked or ambiguous scenes**. After we opened the reference areas, one scripted passenger mark per visible target produced **8 of 8 confirmed pins inside those areas**. This measures two different operations. The first tests whether the app can locate the target by itself. The second tests whether it can preserve a passenger’s choice, check for road support and obtain confirmation. [Case results and first-response records](../experiments/dropoff/passenger-frontage-results-20261123.json)

## Try it in the regular app

Open the villa scene and say or type **“Could you let me out on the driveway of the first arched villa on the left?”** In our release check the app asked for clarification. Tap the entrance pavement you mean, select **Check my spot**, inspect the blue pin, then choose **Confirm this spot**. You can also answer in words. The second link opens the roundabout scene with visible parking bays. [Villa driveway](http://127.0.0.1:8775/?scene=8e586163045c8dbc54ed00c1) · [Roundabout parking](http://127.0.0.1:8775/?scene=67c8b96416f1dc9f40a21a72) · [Installation and startup](passenger-release.md)

## Automatic first responses

| Eight previously unseen drives per comparison | Released app | Experimental approach |
| --- | --- | --- |
| First group, four visible targets | 0 correct, 1 wrong pin, 3 clarifications | 0 correct, 1 wrong pin, 3 clarifications |
| First group, four blocked or ambiguous scenes | 4 clarifications | 3 clarifications, 1 wrong pin at a hospital junction |
| Second group, four visible targets | 0 correct, 3 clarifications, 1 local inference error | 0 correct, 3 clarifications, 1 local inference error |
| Second group, four blocked or ambiguous scenes | 4 clarifications | 4 clarifications |

The first experimental approach added ordinary curbside asphalt to the places Qwen could consider. It recovered two previously missed development examples, then placed a pin in the junction outside Burjeel hospital on the first independent group. The second approach localized the named landmark separately and inspected an enlarged view of the nearest marked bay. On the second group, the seven requests that reached the parking route had no eligible bay, so Qwen never received a candidate. The villa wording went to the laptop grounder and both runs failed. Neither experiment qualified for release. Their source remains available for reproduction. [Experimental code](https://github.com/A1Prep/TrackRAG-Gulf/tree/eec1f50) · [Scoring record](../experiments/dropoff/passenger-frontage-results-20261123.json)

Three different failures appear in the original images. At the roundabout, five of six candidate points were inside the marked parking area. Qwen selected one and later rejected it. At the Mercure hotel, most of the arrival area was classified as ordinary road. At the white building in the second set, the segmenter recognized the parking strip, but the vehicle exclusion and clearance checks removed every candidate. These errors occurred at different stages. [Per-photo model decisions and road-mask diagnosis](../experiments/dropoff/passenger-frontage-results-20261123.json)

## Passenger-marked recovery

The published records contain the complete signed segmentation maps and language replies for 30 of the 32 first turns. We reproduced all 30 decisions from the installed public photos and their detector inputs. The other two records preserve the local villa inference errors. [Portable inference receipts](../experiments/dropoff/passenger-frontage-receipts-20261123.jsonl) · [Detector inputs and original requests](../experiments/dropoff/passenger-frontage-requests-20261123.jsonl)

After both comparisons, we repeated the eight visible requests through the regular app. We used one small mark at the most interior point of each frozen reference area, with the same rule on every photo. Seven first responses asked for clarification. The eighth offered the wrong St Regis pin, which the scripted passenger replaced. In all eight cases the checked pin stayed inside the reference area, a confirmation with an invalid identifier was rejected, and a separate confirmation recorded the passenger’s choice. These marks came from the opened references. A human passenger study would measure how readily real riders can identify and mark the intended spot.

The updated app recognizes driveway, forecourt and villa requests. The same villa wording that had exhausted laptop memory reached the approved public-photo grounder in 5.26 seconds, asked for clarification, and accepted a passenger-marked point. On the furniture-store photo, the app rejected a mark on an occupied area and left the request open for another choice. In Edge, a typed desktop request completed through to confirmation, and a synthetic speech WAV transcribed by Whisper on an emulated phone completed through transcript review, clarification, the rejected mark and cancellation. [Recorded outcomes](../experiments/dropoff/passenger-frontage-results-20261123.json)

## Photos and source credits

Green outlines show independently marked areas. Red crosses show wrong automatic pins. Blue dots show the later checked passenger marks. An independent AI reviewer froze all expected responses and visible polygons before seeing the two sets of model outputs. The 16 photographs were deliberately selected to include clear and difficult cases. The figures describe visual placement on these drives. The driver decides where a stop is permitted.

![First eight UAE driver-view photos, wrong automatic pins and later passenger marks](passenger-frontage-1.jpg)

![Second eight UAE driver-view photos and later passenger marks](passenger-frontage-2.jpg)

© Grab and KartaView Contributors. The gallery resizes the source photos and adds outlines, pins and captions. Gallery derivatives use the same licence. The installer verifies each original against its recorded SHA-256. [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) · [Original photographs and download records](../experiments/dropoff/passenger-frontage-photos-20261123.json)

| First group | Second group |
| --- | --- |
| [Red and glass building](https://kartaview.org/details/1119285/104/) | [Lulu Exchange](https://kartaview.org/details/1257895/48/) |
| [Burjeel hospital](https://kartaview.org/details/5299897/55/) | [Orange parking sign](https://kartaview.org/details/66346/113/) |
| [Starbucks entrance](https://kartaview.org/details/3075690/358/) | [Green bins](https://kartaview.org/details/166806/131/) |
| [Roundabout parking](https://kartaview.org/details/5200729/338/) | [White canopy](https://kartaview.org/details/5053449/2/) |
| [Yellow skip](https://kartaview.org/details/976103/139/) | [Yellow forklift](https://kartaview.org/details/54900/393/) |
| [Mercure hotel](https://kartaview.org/details/6581529/932/) | [Villa driveway](https://kartaview.org/details/986841/26/) |
| [St Regis](https://kartaview.org/details/5204473/819/) | [Brixton shop](https://kartaview.org/details/6315033/178/) |
| [Just Play](https://kartaview.org/details/1269447/242/) | [White building parking strip](https://kartaview.org/details/4572137/14/) |
