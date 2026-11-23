// Pin and mark coordinates always refer to the original photograph.
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const steps = [1, 1.5, 2, 3];

export function positionWords([x, y], width, height) {
  const horizontal = x < width / 3 ? "left" : x > 2 * width / 3 ? "right" : "centre";
  if (y < height / 3) return horizontal === "centre" ? "top centre" : "upper " + horizontal;
  if (y > 2 * height / 3) return horizontal === "centre" ? "bottom centre" : "lower " + horizontal;
  return horizontal === "centre" ? "centre" : "middle " + horizontal;
}

// The general grounder checks one pixel. Its larger search box can include parked vehicles.
// Show a region only when the passenger marked it or the pair route checked a road patch.
const regionSources = new Set(["passenger", "mark_preview", "parking_model", "passenger_landmarks", "road_qualified_pair"]);

export function destinationLabel(view) {
  if (view.selected_by === "passenger") return "Blue pin at your mark";
  if (view.selected_by === "passenger_landmarks") return "Spot from your vehicles";
  return regionSources.has(view.selected_by) ? "Suggested spot" : "Suggested pin";
}

export function boxCentre([left,top,right,bottom]) { return [(left+right)/2,(top+bottom)/2]; }

function validBay(bay, width, height) {
  const onPhoto = point => Array.isArray(point) && point.length === 2 && point.every(Number.isFinite)
    && point[0] >= 0 && point[0] <= width && point[1] >= 0 && point[1] <= height;
  return bay && onPhoto(bay.center_xy) && Array.isArray(bay.polygon_xy)
    && bay.polygon_xy.length >= 3 && bay.polygon_xy.every(onPhoto)
    && (bay.passenger_tap_xy === null || onPhoto(bay.passenger_tap_xy));
}

export function offeredBay(view, width, height) {
  const bay = view.bay_outline, pin = view.result?.destination_point_xy;
  return ["proposed","confirmed"].includes(view.state) && validBay(bay,width,height)
    && Array.isArray(pin) && bay.center_xy.every((value,i) => Math.abs(value-pin[i]) < 0.001) ? bay : null;
}

export function bayChoices(view, width, height) {
  const choices = view.bay_choices;
  if (view.state !== "needs_clarification" || view.result?.destination_point_xy || !Array.isArray(choices)
      || !choices.every(bay => validBay(bay,width,height) && typeof bay.id === "string" && bay.id.length)
      || new Set(choices.map(bay => bay.id)).size !== choices.length) return [];
  return choices;
}

export const bayName = (index,bay) => (bay?.extent==="visible_parking_surface" ? "Area " : "Bay ") +
  (index < 26 ? String.fromCharCode(65+index) : index+1);

// A shared painted boundary belongs to neither button. Let the passenger choose a side.
export function bayAtPoint(choices, [x,y], margin) {
  const hits = choices.filter(({polygon_xy:vertices}) => {
    let inside = false;
    for (let i=0,j=vertices.length-1; i<vertices.length; j=i++) {
      const [a,b]=vertices[i], [c,d]=vertices[j], dx=c-a, dy=d-b;
      const length = dx*dx+dy*dy;
      const t = length ? clamp(((x-a)*dx+(y-b)*dy)/length,0,1) : 0;
      if (Math.hypot(x-a-t*dx,y-b-t*dy) <= margin) return false;
      if ((b>y) !== (d>y) && x < a+(y-b)*dx/dy) inside = !inside;
    }
    return inside;
  });
  return hits.length === 1 ? hits[0] : null;
}

const bayGold = "#ffda69", tapBlue = "#8ef3f5", markerDark = "#18312e";

function bayLabels(ctx, labels, scale) {
  const width = ctx.canvas.width/scale, height = ctx.canvas.height/scale;
  const occupied = [], hitBoxes = [];
  ctx.save();
  ctx.scale(scale,scale);
  ctx.font = "600 12px system-ui, sans-serif"; ctx.textBaseline = "middle";
  for (const label of labels) {
    const x = label.point[0]/scale, y = label.point[1]/scale;
    const w = Math.ceil(ctx.measureText(label.text).width) + 14, h = 21, gap = 16;
    const above = [x-w/2,y-gap-h], below = [x-w/2,y+gap];
    const sides = [[x+gap,y-h/2],[x-gap-w,y-h/2]];
    const extraRows = [[above[0],above[1]-h-4],[below[0],below[1]+h+4]];
    const positions = (label.color === bayGold ? [above,below,...sides,...extraRows] : [below,above,...sides,...extraRows])
      .map(([a,b]) => [clamp(a,2,width-w-2),clamp(b,2,height-h-2)]);
    const [left,top] = positions.find(([a,b]) =>
      occupied.every(([l,t,r,d]) => a+w+3 < l || a > r+3 || b+h+3 < t || b > d+3)
      && labels.every(({point:[px,py]}) => px/scale < a-7 || px/scale > a+w+7 || py/scale < b-7 || py/scale > b+h+7)
    ) || positions[0];
    occupied.push([left,top,left+w,top+h]);
    hitBoxes.push({id:label.id,box:[left,top,left+w,top+h].map(value => value*scale)});
    const endX = clamp(x,left,left+w), endY = clamp(y,top,top+h);
    ctx.beginPath(); ctx.moveTo(x,y); ctx.lineTo(endX,endY);
    ctx.lineWidth = 3; ctx.strokeStyle = markerDark; ctx.stroke();
    ctx.lineWidth = 1.3; ctx.strokeStyle = label.color; ctx.stroke();
    ctx.fillStyle = "rgba(18,35,33,.94)"; ctx.fillRect(left,top,w,h);
    ctx.fillStyle = label.color; ctx.fillText(label.text,left+7,top+h/2);
  }
  ctx.restore();
  return hitBoxes;
}

// Coordinates come from this request's road-photo result. Its bounding box is not the painted bay.
export function drawBays(ctx, bays, scale, choosing, tapLabel = "Your tap") {
  ctx.save(); ctx.lineJoin = "round";
  const labels = [];
  for (const [index,bay] of bays.entries()) {
    ctx.beginPath(); ctx.moveTo(...bay.polygon_xy[0]);
    for (const vertex of bay.polygon_xy.slice(1)) ctx.lineTo(...vertex);
    ctx.closePath(); ctx.fillStyle = "rgba(255,211,79,.16)"; ctx.fill();
    ctx.lineWidth = 4.5*scale; ctx.strokeStyle = markerDark; ctx.stroke();
    ctx.lineWidth = 2.5*scale; ctx.strokeStyle = bayGold; ctx.stroke();
    const label = bay.extent === "visible_parking_surface" ? "Parking area" : "Bay centre";
    labels.push({id:bay.id,point:bay.center_xy,text:choosing ? bayName(index,bay) : label,color:bayGold});
  }
  const tap = bays.find(bay => bay.passenger_tap_xy)?.passenger_tap_xy;
  if (tap) labels.push({point:tap,text:tapLabel,color:tapBlue});
  const hitBoxes = bayLabels(ctx,labels,scale);
  for (const bay of bays) drawDestination(ctx,bay.center_xy,null,"model",scale);
  if (tap) {
    ctx.translate(...tap); ctx.scale(scale,scale);
    ctx.beginPath(); ctx.moveTo(0,-6); ctx.lineTo(6,0); ctx.lineTo(0,6); ctx.lineTo(-6,0); ctx.closePath();
    ctx.lineWidth = 4; ctx.strokeStyle = markerDark; ctx.stroke();
    ctx.lineWidth = 2; ctx.strokeStyle = tapBlue; ctx.stroke();
  }
  ctx.restore();
  return hitBoxes;
}

export function drawCheckedMark(ctx,point,scale) {
  ctx.save(); ctx.translate(...point); ctx.scale(scale,scale);
  ctx.beginPath(); ctx.moveTo(-6,-6); ctx.lineTo(6,6); ctx.moveTo(-6,6); ctx.lineTo(6,-6);
  ctx.lineCap="round"; ctx.lineWidth=4; ctx.strokeStyle="#202628"; ctx.stroke();
  ctx.lineWidth=2; ctx.strokeStyle="#ffffff"; ctx.stroke(); ctx.restore();
}

export function drawDestination(ctx, point, box, source, scale) {
  if (!point) return;
  const manual = source === "passenger" || source === "mark_preview";
  const showRegion = box && regionSources.has(source);
  const color = manual ? "#66eff1" : "#ffd34f";
  ctx.save();
  if (showRegion) {
    const [x1,y1,x2,y2] = box;
    ctx.fillStyle = manual ? "rgba(61,220,226,.15)" : "rgba(255,211,79,.10)";
    ctx.fillRect(x1, y1, x2-x1, y2-y1);
    if (source !== "passenger") ctx.setLineDash([7*scale, 4*scale]);
    for (const [stroke,width] of [["#142a25",4.5],[color,2.5]]) {
      ctx.lineWidth = width*scale; ctx.strokeStyle = stroke; ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    }
  }
  ctx.setLineDash([]); ctx.translate(point[0],point[1]); ctx.scale(scale,scale);
  ctx.beginPath(); ctx.arc(0,0,10.5,0,2*Math.PI);
  ctx.lineWidth=3.6; ctx.strokeStyle="#16352d"; ctx.stroke();
  ctx.lineWidth=1.8; ctx.strokeStyle=showRegion ? "#ffffff" : color; ctx.stroke();
  ctx.beginPath(); ctx.arc(0,0,showRegion ? 6 : 2,0,2*Math.PI);
  ctx.fillStyle=color; ctx.fill(); ctx.lineWidth=1.8; ctx.strokeStyle="#16352d"; ctx.stroke();
  ctx.restore();
}

export class PhotoTools {
  constructor(canvas, viewport, refresh, locked, pairPicker) {
    this.canvas = canvas; this.viewport = viewport; this.refresh = refresh;
    this.locked = locked; this.pairs = pairPicker;
    this.zoom = 0; this.reset();
    canvas.addEventListener("pointerdown", e => this.pointerDown(e));
    canvas.addEventListener("click", e => {
      if (!this.quickMark || e.button !== 0 || this.locked()) return;
      if (e.detail > 0) this.begin(this.point(e), true);
      else this.begin();
    });
    canvas.addEventListener("pointermove", e => {
      if (this.start && e.isPrimary && !this.locked()) this.updateBox(this.point(e), false);
    });
    canvas.addEventListener("pointerup", e => {
      if (this.start && e.isPrimary && !this.locked()) {
        this.updateBox(this.point(e), true); this.start = null;
      }
    });
    canvas.addEventListener("pointercancel", () => {
      if (this.start) { this.start = this.box = null; this.refresh(); }
    });
    canvas.addEventListener("keydown", e => this.key(e));
    viewport.addEventListener("pointerdown", e => {
      if (e.pointerType !== "mouse" || !e.isPrimary || e.button !== 0 || this.zoom === 0 ||
          this.pairs.open || (this.active && !this.panning) || !canvas.parentElement.contains(e.target)) return;
      this.drag = [e.clientX, e.clientY, viewport.scrollLeft, viewport.scrollTop];
      viewport.setPointerCapture(e.pointerId); e.preventDefault();
    });
    viewport.addEventListener("pointermove", e => {
      if (!this.drag) return;
      viewport.scrollTo(this.drag[2] + this.drag[0] - e.clientX, this.drag[3] + this.drag[1] - e.clientY);
    });
    for (const name of ["pointerup", "pointercancel", "lostpointercapture"])
      viewport.addEventListener(name, () => { this.drag = null; });
  }
  reset() {
    this.active = this.panning = this.dragged = false;
    this.start = this.box = this.cursor = this.drag = null;
    this.canvas.tabIndex = -1;
  }
  fit() {
    this.zoom = 0; this.panning = false;
    this.canvas.parentElement.style.width = "100%";
    this.viewport.scrollTo(0, 0);
  }
  frameRatio() {
    return this.canvas.width / this.canvas.getBoundingClientRect().width;
  }
  viewCenter() {
    const v = this.viewport, scale = this.frameRatio();
    return [(v.scrollLeft + v.clientWidth / 2) * scale, (v.scrollTop + v.clientHeight / 2) * scale];
  }
  setZoom(delta, target) {
    const index = clamp(this.zoom + delta, 0, steps.length - 1);
    if (index === this.zoom) return;
    const center = target || this.viewCenter();
    this.zoom = index;
    this.canvas.parentElement.style.width = (100 * steps[index]) + "%";
    if (!index) this.panning = false;
    this.centerOn(center); this.refresh();
  }
  centerOn(point) {
    const v = this.viewport, scale = this.frameRatio();
    v.scrollTo(point[0] / scale - v.clientWidth / 2, point[1] / scale - v.clientHeight / 2);
  }
  focusPhoto() {
    this.canvas.focus({preventScroll:true});
    this.viewport.scrollIntoView({block:"nearest"});
  }
  begin(point, placeMark = false) {
    this.reset(); this.active = true; this.cursor = point || this.viewCenter();
    if (placeMark) this.box = this.tapBox(this.cursor);
    this.canvas.tabIndex = 0; this.refresh(); this.focusPhoto();
  }
  point(e) {
    const c = this.canvas, rect = c.getBoundingClientRect();
    return [clamp((e.clientX - rect.left) * c.width / rect.width, 0, c.width),
      clamp((e.clientY - rect.top) * c.height / rect.height, 0, c.height)];
  }
  pointerDown(e) {
    if (this.locked() || !e.isPrimary || e.button !== 0) return;
    if (this.pairs.open) { e.preventDefault(); this.pairs.pickAt(this.point(e)); return; }
    if (!this.active || this.panning) return;
    e.preventDefault(); this.canvas.setPointerCapture(e.pointerId);
    this.start = this.cursor = this.point(e); this.box = null; this.dragged = false; this.refresh();
  }
  tapBox([x, y]) {
    const c = this.canvas;
    // Preserve the tapped centre while checking a small patch in original-image pixels.
    const r = Math.max(1, c.width / 512);
    const cx=clamp(x,.5,c.width-.5), cy=clamp(y,.5,c.height-.5);
    const rx=Math.min(r,cx,c.width-cx), ry=Math.min(r,cy,c.height-cy);
    return [cx-rx,cy-ry,cx+rx,cy+ry];
  }
  updateBox([x, y], final) {
    const [a,b] = this.start;
    this.cursor = [x, y];
    this.dragged = Math.hypot(x-a, y-b) > 3 * this.frameRatio();
    this.box = final && !this.dragged ? this.tapBox(this.cursor) :
      [Math.floor(Math.min(x,a)), Math.floor(Math.min(y,b)), Math.ceil(Math.max(x,a)), Math.ceil(Math.max(y,b))];
    if (this.box[2] <= this.box[0] || this.box[3] <= this.box[1]) this.box = null;
    this.refresh();
  }
  key(e) {
    if (this.locked() || e.ctrlKey || e.metaKey || e.altKey) return;
    if (this.quickMark && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault(); this.begin(); return;
    }
    if (!this.active) return;
    if (e.key === "Escape") { e.preventDefault(); document.getElementById("discard-box").click(); return; }
    if (this.panning) return;
    if (["Enter", " "].includes(e.key)) {
      e.preventDefault(); this.box = this.tapBox(this.cursor); this.dragged=false; this.refresh(); return;
    }
    const directions = {ArrowLeft:[-1,0], ArrowRight:[1,0], ArrowUp:[0,-1], ArrowDown:[0,1]};
    if (!directions[e.key]) return;
    e.preventDefault();
    const [dx,dy] = directions[e.key], step = (e.shiftKey ? 36 : 4) * this.frameRatio();
    const c = this.canvas;
    this.cursor = [clamp(this.cursor[0] + dx * step, 0, c.width), clamp(this.cursor[1] + dy * step, 0, c.height)];
    this.box = null;
    if (this.zoom) this.centerOn(this.cursor);
    this.refresh();
  }
  controls(ready, busy, overlayLabel, offerMark = false) {
    const get = id => document.getElementById(id);
    get("zoom-in").disabled = !ready || this.zoom === steps.length-1;
    get("zoom-out").disabled = !ready || this.zoom === 0;
    get("zoom-level").textContent = steps[this.zoom] + "×";
    get("pan").hidden = !this.active || !this.zoom;
    get("pan").disabled = busy;
    get("pan").setAttribute("aria-pressed", String(this.panning));
    get("pan").textContent = this.panning ? "Return to marking" : "Move zoomed photo";
    this.viewport.classList.toggle("pan-view", this.zoom > 0 && !this.pairs.open && (!this.active || this.panning));
    this.canvas.classList.toggle("drawing", this.active && !this.panning);
    this.canvas.classList.toggle("choosing-pair", this.pairs.open);
    this.quickMark = offerMark && ready && !busy && !this.active && !this.pairs.open && this.zoom === 0;
    this.canvas.classList.toggle("quick-mark", this.quickMark);
    this.canvas.tabIndex = this.active || this.quickMark ? 0 : -1;
    this.canvas.setAttribute("role", this.active ? "application" : this.quickMark ? "button" : "img");
    this.canvas.setAttribute("aria-label", this.active || this.quickMark ? "Mark your spot on the road photo" : overlayLabel);
    this.canvas.setAttribute("aria-describedby", this.active ? "mark-help mark-position" : "legend status");
    get("draw-tools").hidden = !this.active;
    get("use-box").disabled = busy || !ready || !this.box || this.panning;
    get("discard-box").disabled = busy;
    if (this.active && this.cursor) {
      const point = this.box ? boxCentre(this.box) : this.cursor;
      const where = positionWords(point, this.canvas.width, this.canvas.height);
      const percent = point.map((p,i)=> Math.round(100*p/(i ? this.canvas.height : this.canvas.width)));
      const message = this.box ? "Point to check" : "Cursor";
      const position = message + " · " + where + " (" + percent[0] + "% across, " + percent[1] + "% down).";
      if (get("mark-position").textContent !== position) get("mark-position").textContent = position;
    }
  }
  drawCursor(ctx, scale) {
    if (!this.active || this.box || !this.cursor || this.panning) return;
    const [x,y] = this.cursor;
    ctx.save();
    for (const [color,width] of [["#12352f", 4], ["#80f4e0", 2]]) {
      ctx.strokeStyle=color; ctx.lineWidth=width*scale; ctx.beginPath();
      ctx.arc(x,y,7*scale,0,2*Math.PI); ctx.moveTo(x-12*scale,y); ctx.lineTo(x+12*scale,y);
      ctx.moveTo(x,y-12*scale); ctx.lineTo(x,y+12*scale); ctx.stroke();
    }
    ctx.restore();
  }
}
