import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

const source = await readFile(new URL("../trackrag/passenger_ui/photo-tools.js",import.meta.url));
const {PhotoTools, positionWords} = await import("data:text/javascript;base64," + source.toString("base64"));
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
