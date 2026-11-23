import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

const source = await readFile(new URL("../trackrag/passenger_ui/photo-tools.js",import.meta.url));
const {PhotoTools, bayAtPoint, bayChoices, boxCentre, destinationLabel, drawDestination, offeredBay, positionWords}
  = await import("data:text/javascript;base64," + source.toString("base64"));
const car = [1340.16,1281.42];  // Centre of a real vehicle in the blind-07 phone regression.
for (const displayWidth of [320,375,1000,1800]) {
  for (const zoom of [1,1.5,2,3]) {
    const width = displayWidth * zoom, height = width * 2160 / 3840;
    const controls = Object.create(PhotoTools.prototype);
    controls.canvas = {width:3840,height:2160,
      getBoundingClientRect:() => ({left:-38,top:84,width,height})};
    const pointer = {clientX:-38 + car[0]*width/3840,clientY:84 + car[1]*height/2160};
    const location = controls.point(pointer);
    const expected = controls.tapBox(location);
    assert(boxCentre(expected).every((value,i)=>Math.abs(value-car[i])<1e-8));
    assert.deepEqual(controls.tapBox([0,0]), [0,0,1,1]);
    assert.deepEqual(controls.tapBox([3840,2160]), [3839,2159,3840,2160]);
    assert.deepEqual(boxCentre(controls.tapBox([1,4])),[1,4]);
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

// Direct marking opens a local draft. A keyboard or assistive activation has no screen coordinate.
const handlers = {}, canvasEvents = {addEventListener:(name,fn)=>{handlers[name]=fn;}};
const controls = new PhotoTools(canvasEvents, {addEventListener(){}}, ()=>{}, ()=>false, {open:false});
const drafts = [];
controls.point = () => [400,250];
controls.begin = (...args) => drafts.push(args);
controls.quickMark = true;
handlers.click({button:0,detail:1});
handlers.click({button:0,detail:0});
controls.key({key:"Enter",preventDefault(){}});
assert.deepEqual(drafts, [[[400,250],true],[],[]]);
controls.quickMark = false;
handlers.click({button:0,detail:1});
controls.quickMark = true; controls.locked = () => true;
handlers.click({button:0,detail:1});
assert.equal(drafts.length,3);

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
assert.equal(destinationLabel({selected_by:"passenger"}), "Blue pin at your mark");
assert.deepEqual(paintCalls("model",null,1), [], "An unplaced pin cannot implicitly highlight a coarse box");

// Test perspective edges and overlap, rather than the enclosing rectangles.
const a={id:"bay-1",polygon_xy:[[4,8],[12,5],[18,14],[6,17]],center_xy:[9,11],passenger_tap_xy:[15,9.5]};
const b={id:"bay-2",polygon_xy:[[12,5],[20,3],[28,10],[18,14]],center_xy:[19,8],passenger_tap_xy:[15,9.5]};
assert.equal(bayAtPoint([a,b],a.center_xy,0.1),a);
assert.equal(bayAtPoint([a,b],b.center_xy,0.1),b);
assert.equal(bayAtPoint([a,b],a.passenger_tap_xy,0.1),null);
assert.equal(bayAtPoint([a,b],[28,14],0.1),null);
assert.equal(bayAtPoint([a,a],a.center_xy,0.1),null);
const choicesView={state:"needs_clarification",bay_choices:[a,b]};
assert.deepEqual(bayChoices(choicesView,30,20),[a,b]);
assert.deepEqual(bayChoices({...choicesView,state:"cancelled"},30,20),[]);
assert.deepEqual(bayChoices({...choicesView,bay_choices:[a,a]},30,20),[]);
const offer={state:"proposed",bay_outline:a,result:{destination_point_xy:a.center_xy}};
assert.equal(offeredBay(offer,30,20),a);
assert.equal(offeredBay({...offer,state:"needs_clarification"},30,20),null);
assert.equal(offeredBay({...offer,result:{destination_point_xy:b.center_xy}},30,20),null);
assert.equal(offeredBay({...offer,bay_outline:{...a,polygon_xy:[[0,0],[16,-1],[15,14]]}},30,20),null);
