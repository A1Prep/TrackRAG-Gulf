# Footage search on two EMT recordings

Nine of the thirteen questions named a playback time. The revised search returned every requested original frame in **9 of 9** cases, up from **3 of 9**. It also kept every result at or before the latest time the question asked about in **9 of 9**, up from **2 of 9**. The frame numbers came from each recording’s own clock.

| Recording and question | Previous result | Revised result |
| --- | --- | --- |
| 054604, paused at frame 150. “Is the yellow bus visible on the left at both 0:03 and 0:12.5?” | Gemini answered yes and mapped the times to frames 111 and 150. | Gemini answered no and cited frames 31 and 126. The bus is visible on the left in the second image. |
| 054907, paused at frame 150. “At 0:07, what structure stretches across the road above the traffic?” | The search returned no images. | Gemini identified the overpass from frame 71. |
| 054907, paused at frame 91. “Two seconds ago, was the overpass overhead or had we already passed it?” | The search returned the pause frame, 91. | Gemini described the overpass overhead in frame 71. An initial staging response declined even after it received the right image. The revised prompt tied the relative time explicitly to the pause. |

The canonical hosted run returned **eight visually supported answers, four abstentions and one service error** over thirteen questions. The earlier hosted comparison covered seven of those questions. Both versions used the configured **Gemini 3.5 Flash-Lite** model. One request for the pickup’s precise physical speed received HTTP 503 from Google after retry. A subsequent update recognized that the available data had no measurement of another vehicle’s physical speed. The same question then produced an explanation through the application with zero external calls. A separate pickup-motion question also encountered a 503 in staging and succeeded on canonical, where it cited frames 51 and 111. The pickup’s tracked centre moved from 35.8% to 13.7% of the image width.

Four local questions were checked directly against saved detector observations. The bus and pickup location questions both correctly reported a box centre on the left. At frame 126 of 054604 the tracker reported one bus. At frame 71 of 054907 it reported nine cars. All four answers cited the selected original. The combined application also declined a question containing conflicting frame and clock values and a two-time request extending past the pause.

One reviewer inspected the public originals and wrote the references before testing. Each recording is 15 seconds long at ten frames per second. These cases test named playback times, identity and limits of the available evidence on the two recordings. They cannot establish general footage-answer accuracy. The detailed results include each question, the frames retrieved, answer citations, model call status and receipt hashes. [Per-question evidence](footage-search-public-clips-20261124.json)
