// Numbers connect the accessible selectors to the actual boxes on this original photograph.
const el = id => document.getElementById(id);
const slotColors = ["#8be5ff", "#ffbded"];

function outline(ctx, box, label, color, scale, selected) {
  const [x1,y1,x2,y2] = box;
  ctx.save();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#142121"; ctx.lineWidth = (selected ? 5.5 : 3) * scale;
  ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  ctx.strokeStyle = color; ctx.lineWidth = (selected ? 3 : 1.2) * scale;
  ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  ctx.font = (selected ? "bold " : "") + (11*scale) + "px sans-serif";
  const width = ctx.measureText(label).width + 7*scale, height = 16*scale;
  const x = Math.max(0,Math.min(x1,ctx.canvas.width-width));
  const y = Math.max(0,Math.min(y1-height,ctx.canvas.height-height));
  ctx.fillStyle = selected ? color : "#1d332c";
  ctx.fillRect(x,y,width,height);
  ctx.fillStyle = selected ? "#142121" : "#ffffff"; ctx.textBaseline = "middle";
  ctx.fillText(label,x+3.5*scale,y+height/2);
  ctx.restore();
}

export function drawChosenLandmarks(ctx, boxes, scale) {
  if (boxes?.length === 2) boxes.forEach((box,i)=>outline(ctx,box,i===0?"A":"B",slotColors[i],scale,true));
}

export function drawVehicleReferences(ctx, boxes, scale) {
  for (const [index,box] of (boxes || []).entries()) {
    outline(ctx,box,boxes.length===1 ? "Vehicle" : "Vehicle " + (index+1),"#8be5ff",scale,true);
  }
}

export class VehiclePairPicker {
  constructor(refresh) {
    this.refresh = refresh;
    this.selects = [el("pair-first"),el("pair-second")];
    this.position = el("pair-placement");
    this.reset();
    this.selects.forEach((select,i)=>{
      select.onchange = () => { this.nextSlot = 1-i; this.refresh(); };
    });
    this.position.onchange = this.refresh;
    el("discard-pair").onclick = () => { this.reset(); this.refresh(); };
  }
  reset() {
    this.open = false; this.rows = []; this.nextSlot = 0;
    this.position.value = "";
    this.selects.forEach(select=>select.replaceChildren());
  }
  begin(rows) {
    this.reset();
    this.rows = [...rows].sort((a,b)=>a.bbox_xyxy[0]-b.bbox_xyxy[0] ||
      a.bbox_xyxy[1]-b.bbox_xyxy[1] || a.track_id-b.track_id);
    this.selects.forEach(select=>{
      select.add(new Option("Choose a vehicle",""));
      this.rows.forEach((row,i)=>{
        const type = row.class_name.replaceAll("_"," ").toLowerCase();
        select.add(new Option((i+1)+" · "+type,String(row.track_id)));
      });
    });
    this.open = true; this.refresh();
  }
  values() {
    return this.selects.map(s=>s.value==="" ? null : Number(s.value));
  }
  hasPair() {
    const [a,b] = this.values();
    return a !== null && b !== null && a !== b && ["between","behind"].includes(this.position.value);
  }
  controls(busy) {
    el("pair-tools").hidden = !this.open;
    if (!this.open) return;
    const chosen = this.values();
    this.selects.forEach((select,i)=>{
      select.disabled = busy;
      for (const option of select.options) option.disabled = option.value!=="" && Number(option.value)===chosen[1-i];
    });
    this.position.disabled = el("discard-pair").disabled = busy;
    el("use-pair").disabled = busy || !this.hasPair();
  }
  pickAt([x,y]) {
    const contains = this.rows.filter(({bbox_xyxy:[a,b,c,d]})=>a<=x && x<=c && b<=y && y<=d);
    const area = r=>(r.bbox_xyxy[2]-r.bbox_xyxy[0])*(r.bbox_xyxy[3]-r.bbox_xyxy[1]);
    contains.sort((a,b)=>area(a)-area(b));
    const candidate = contains.find(row=>!this.values().includes(row.track_id));
    if (!candidate) return;
    const empty = this.values().indexOf(null), index = empty===-1 ? this.nextSlot : empty;
    this.selects[index].value = String(candidate.track_id);
    this.nextSlot = 1-index; this.refresh();
  }
  paint(ctx,scale) {
    const selected = this.values();
    this.rows.forEach((row,i)=>{
      if (!selected.includes(row.track_id)) outline(ctx,row.bbox_xyxy,String(i+1),"#e2ebd8",scale,false);
    });
    this.rows.forEach((row,i)=>{
      const slot = selected.indexOf(row.track_id);
      if (slot!==-1) outline(ctx,row.bbox_xyxy,(slot===0?"A · ":"B · ")+(i+1),slotColors[slot],scale,true);
    });
  }
}
