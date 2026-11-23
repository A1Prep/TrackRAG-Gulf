# How often did the passenger app locate the whole bay?

On 15 UAE driver-view photographs, the revised app outlined one correctly located bay as an option for a passenger to select. Across eight requests with a visibly vacant target, none produced a single correct whole-bay suggestion on the first turn. The app did better at rejecting inaccurate passenger taps. It withheld all 31 marked road, occupied, barrier and other excluded positions in this study.

| Passenger action | Cases | Observed on the revised release |
| --- | ---: | --- |
| Ask for a visibly vacant bay | 8 | One correct outlined option for selection, six misses, one timeout. One missed request showed an option outside the requested location. |
| Ask for an unavailable bay | 7 | Six genuine clarifications and one timeout. |
| Tap a clearly intended visible bay or its unshared edge | 12 | One correctly outlined bay, five correctly located points without an outline and six withheld locations. |
| Tap a line shared by two bays | 4 | One showed a single side for clarification. Three asked for another mark. None distinguished both adjacent bays. |
| Tap an excluded location | 31 | All 31 withheld a destination. |

Four additional through-road taps that had previously been saved were replayed at their original coordinates. The revised app withheld all four. It also withheld a fifth tap at an equivocal entrance. Each replay checked the original image, submitted mark and visible-surface evidence.

### What changed after the first prospective test?

The first ten recordings formed a fixed comparison set. Before the next release was examined, five more drives were selected, reviewed and sealed. These five were already in the public app catalogue but had never been labelled in this whole-bay evaluation. All participating images had undergone the same original catalogue preparation. The subsequent occupancy correction used a failure discovered on the five-drive set, so the second measurement is a repeat on known cases.

| Five newly labelled drives | First prospective test on 8808 | Repeat after the occupancy correction on 8809 |
| --- | --- | --- |
| Three genuinely vacant target requests | Two misses, one timeout | Two misses, one timeout |
| Occupied frontage | Displayed four incorrect selectable options. Clicking one offered it for confirmation. | Withheld all four options in 0.69 seconds. |
| Through traffic behind a school bus | Returned an operationally pending result, with no pin | Returned a clarification with no pin |
| Twelve independent passenger taps | Eight correct exclusions, two correct points, one missed bay and one unresolved shared line | Same grades |

On the ten-drive repeat, the revised app removed an extra incorrect option beside a correct bay. The passenger could select the remaining bay and then confirm it. The stricter clearance check also removed a small pavement patch that had landed inside a genuine bay but had covered only 19% of it.

### Response times and repeatability

Across the 15 staged typed submissions the median response was 0.33 seconds and the maximum was 15.19 seconds. Across the 47 manual submissions the median was 0.22 seconds and the maximum was 0.91 seconds. These are local interactive measurements with prepared evidence available. They exclude downloading images and preparing evidence for a new recording.

The Chrome checks used both a desktop window and a 390-pixel mobile viewport. On one of the prospective photos, retrying the same left-side parking instruction through the page returned a correct point inside bay A. The first HTTP request had withheld a location. The retry displayed no bay outline. After a subsequent parser change on the canonical application, the request between the silver sedan and dark SUV returned clarification in 0.30 seconds. The same request had timed out in the staged 8809 study.

### What the scores measure

The eight visible targets and seven negative requests came from separate recordings. Forty-seven manual taps tested several types of locations in the same photographs. A correct outline had to contain a point inside the photographed target, cover at least 50% of the reference bay, keep at least 70% of the predicted area inside it and reach 0.45 intersection over union. One independent AI reviewer traced the photographic references. No driver study or on-road evaluation was performed. The sample measures visible correspondence in these photographs. It cannot establish stopping permission, vehicle clearance or dependable performance on unfamiliar roads.

The full results include each image source, gate and overlap score, response time, first-release prospective output, correction and journal replay. Each of the 62 staged interactions reproduced on both releases. [Full evaluation results](passenger-whole-bay-independent-results-20261123.json)

The primary protocol preserves the original questions, coordinates and grading rules. The prospective protocol records the five additional drives. One shared-stripe coordinate was corrected against the original photograph before the first result was inspected. [Ten-drive protocol](passenger-whole-bay-independent-plan-20261123.json) · [Five-drive protocol](passenger-whole-bay-prospective-plan-20261123.json) · [Pre-run correction](passenger-whole-bay-prospective-amendment-20261123.json)

### Follow-up on the road-divider check

Release 8810 checks whether a proposed parking surface visibly connects to the road ahead. Replaying all 47 original tap rectangles across the same 15 photographs produced identical pins, outlines, available choices and confirmation behavior. All 31 excluded locations were withheld again. All 47 tap records replayed against their source photographs. Median response was 0.23 seconds and the maximum was 1.20 seconds.

Three typed instructions were retested separately. In W03, the passenger could select the genuine bay, whose outline reached 0.74 intersection over union. In W07, the app still offered an area with zero overlap with the requested restaurant-side bay. Selecting either option opened a separate confirmation step. In P03, the app asked the passenger to mark the space between the silver sedan and dark SUV in 0.28 seconds, with no pin. These measurements cover the previously scored drives.

### Five checks on the final application

On the final v4 release, W07 stopped offering the unrelated restaurant-side area and asked for clarification in 0.47 seconds. Its intended bay still went undetected. W03 kept the correct outlined option in 0.39 seconds, with 0.74 intersection over union and a separate confirmation after selection. P03 again asked the passenger to mark the space between the silver sedan and dark SUV in 0.80 seconds. Tapping the W04 through lane or the W10 dashed traffic stripe produced no pin, choice or save option. All five session records and the W03 selection replayed against the original photographs. These five previously exposed cases are reported separately from the 15-drive evaluation above.
