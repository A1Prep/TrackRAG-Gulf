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
