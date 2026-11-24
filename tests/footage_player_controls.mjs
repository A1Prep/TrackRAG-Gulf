import assert from "node:assert/strict";
import {readFileSync} from "node:fs";

// Isolate asynchronous image decoding so a late camera frame can be delivered deterministically.
const nodes=new Map(), deferred=new Map();
function makeNode(id) {
  return {id,style:{},dataset:{},hidden:false,removeAttribute(){},setAttribute(name,value){this[name]=value;},
    replaceWith(other){nodes.set("clip-frame",other);}};
}
for(const id of ["clip-frame","clip-message","clip-position","clip-screen","clip-play","clip-restart","clip-seek"]) nodes.set(id,makeNode(id));
globalThis.document={getElementById:id=>nodes.get(id),addEventListener(){}};
globalThis.Image=class {
  constructor(){Object.assign(this,makeNode(""));}
  decode(){const id=Number(this.src.split("/").at(-1));return deferred.get(id)?.promise || Promise.resolve();}
};
function hold(id){let resolve;const promise=new Promise(f=>{resolve=f;});deferred.set(id,{promise,resolve});return resolve;}
const code=readFileSync(new URL("../trackrag/passenger_ui/footage-player.js",import.meta.url));
const {FootagePlayer}=await import("data:text/javascript;base64,"+code.toString("base64"));
const player=new FootagePlayer(()=>{});
player.reset("release",{width:100,height:50});
const release=hold(6);
await player.load([{id:3,time:0},{id:6,time:.15},{id:10,time:.35}],3);
assert.equal(player.displayed.id,3);
player.play();
const delayed=player.display(1);
// A pause/search snapshots the image actually visible, even if another original is decoding.
player.pause();
release();await delayed;
assert.equal(player.displayed.id,3);
await player.display(1);
assert.match(nodes.get("clip-position").textContent,/Frame 6 · 0:00\.15/);
assert.match(nodes.get("clip-seek")["aria-valuetext"],/0.15 seconds/);
assert.match(nodes.get("clip-frame").src,/\/6$/);
await nodes.get("clip-restart").onclick();
assert.equal(player.displayed.id,3);
assert.equal(player.playing,true);
player.pause();
await player.display(1);
await player.playLeadIn();
assert.equal(player.displayed.id,3);
assert.equal(player.playing,true);
// A later frame exists in this recording, but the lead-in ends on the opened frame.
await new Promise(resolve=>setTimeout(resolve,400));
assert.equal(player.displayed.id,6);
assert.equal(player.playing,false);
player.setLocked(true);
await player.playLeadIn();
assert.equal(player.displayed.id,6);
assert.equal(player.playing,false);
player.setLocked(false);
const stale=hold(41);player.reset("first",{width:100,height:50});
const first=player.load([{id:41,time:0}],41);
player.reset("second",{width:100,height:50});
await player.load([{id:99,time:0}],99);
stale();await first;
assert.equal(player.displayed.id,99);assert.match(nodes.get("clip-frame").src,/second\/99$/);
