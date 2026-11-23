import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

const source = await readFile(new URL("../trackrag/passenger_ui/photo-tools.js",import.meta.url));
const {PhotoTools, destinationLabel, drawDestination, positionWords} = await import("data:text/javascript;base64," + source.toString("base64"));
const car = [1340.16,1281.42];  // Centre of a real vehicle in the blind-07 phone regression.
const expected = [1332,1273,1348,1289];
for (const displayWidth of [320,375,1000,1800]) {
  for (const zoom of [1,1.5,2,3]) {
    const width = displayWidth * zoom, height = width * 2160 / 3840;
    const controls = Object.create(PhotoTools.prototype);
    controls.canvas = {width:3840,height:2160,
      getBoundingClientRect:() => ({left:-38,top:84,width,height})};
    const pointer = {clientX:-38 + car[0]*width/3840,clientY:84 + car[1]*height/2160};
    const location = controls.point(pointer);
    assert.deepEqual(controls.tapBox(location), expected);
    assert.deepEqual(controls.tapBox([0,0]), [0,0,8,8]);
    assert.deepEqual(controls.tapBox([3840,2160]), [3832,2152,3840,2160]);
    controls.active=true; controls.cursor=location; controls.box=null;
    controls.locked=() => false; controls.refresh=() => {};
    controls.key({key:"Enter", preventDefault() {}});
    assert.deepEqual(controls.box, expected);
    controls.key({key:"ArrowLeft", preventDefault() {}});
    assert.equal(controls.box, null, "Moving the cursor invalidates its previous preview");
    controls.start = [100,200];
    controls.updateBox([1600,1200],true);
    assert.deepEqual(controls.box,[100,200,1600,1200], "A drag keeps the explicitly chosen area");
  }
}
assert.equal(positionWords([3800,2000],3840,2160), "lower right");

// On the furniture-store photo, five vehicle detections overlap the estimated region.
// Its pin is the only part checked against the road mask.
const savedPoint = [2958.75,1983.75];
const savedBox = [2539.285,1825.186,3388.452,2117.020];
function paintCalls(actor, point, scale) {
  const calls = [], ctx = {};
  for (const method of ["save","restore","setLineDash","translate","scale","beginPath","arc","fill","stroke","fillRect","strokeRect"]) {
    ctx[method] = (...args) => calls.push([method,...args]);
  }
  drawDestination(ctx,point,savedBox,actor,scale);
  return calls;
}
for (const scale of [1,3.84,12]) {
  for (const actor of ["model", "unrecognized", undefined]) {
    const calls = paintCalls(actor,savedPoint,scale);
    assert.equal(destinationLabel({selected_by:actor}), "Suggested pin");
    assert.equal(calls.some(([name]) => name === "fillRect" || name === "strokeRect"), false);
    assert.deepEqual(calls.find(([name]) => name === "translate"), ["translate",...savedPoint]);
    assert.deepEqual(calls.find(([name]) => name === "scale"), ["scale",scale,scale]);
  }
  for (const actor of ["passenger", "mark_preview", "passenger_landmarks", "parking_model", "road_qualified_pair"]) {
    const calls = paintCalls(actor,savedPoint,scale);
    const [x1,y1,x2,y2] = savedBox;
    assert.deepEqual(calls.find(([name]) => name === "fillRect"), ["fillRect",x1,y1,x2-x1,y2-y1]);
    assert.deepEqual(calls.find(([name]) => name === "translate"), ["translate",...savedPoint]);
  }
}
assert.equal(destinationLabel({selected_by:"passenger"}), "Your marked area and pin");
assert.deepEqual(paintCalls("model",null,1), [], "An unplaced pin cannot implicitly highlight a coarse box");
