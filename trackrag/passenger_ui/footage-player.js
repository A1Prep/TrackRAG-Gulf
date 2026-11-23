// One decoded original is the authoritative frame for both the timestamp and the question.
const get = id => document.getElementById(id);
const elapsed = time => {
  const ms = Math.round(time * 1000);
  const fraction = String(ms % 1000).padStart(3,"0").replace(/0+$/,"") || "0";
  return Math.floor(ms / 60000) + ":" + String(Math.floor(ms / 1000) % 60).padStart(2,"0") + "." + fraction;
};

export class FootagePlayer {
  constructor(changed) {
    this.changed = changed;
    this.frames = []; this.cache = new Map(); this.index = -1; this.ticket = 0;
    this.locked = false; this.playing = false; this.pending = false; this.timer = null;
    get("clip-play").onclick = () => this.playing ? this.pause() : this.play();
    get("clip-seek").oninput = () => {
      this.pause(false);
      void this.display(Number(get("clip-seek").value));
    };
    document.addEventListener("visibilitychange", () => { if (document.hidden) this.pause(); });
  }
  get displayed() { return this.frames[this.index]; }
  controls() {
    get("clip-play").textContent = this.playing ? "Pause" : this.index === this.frames.length-1 && this.index > 0 ? "Replay" : "Play";
    get("clip-play").disabled = this.locked || this.frames.length < 2 || (this.pending && !this.playing);
    get("clip-seek").disabled = this.locked || this.frames.length < 2;
  }
  setLocked(value) {
    this.locked = value;
    if (value) this.pause(false);
    this.controls();
  }
  pause(notify = true) {
    clearTimeout(this.timer);
    if (this.playing) {
      ++this.ticket; this.pending = false;
    }
    this.playing = false;
    this.controls();
    if (notify && this.displayed) this.changed();
  }
  reset(key, options) {
    this.pause(false); ++this.ticket; this.pending = false; this.key = key;
    this.frames = []; this.index = -1; this.cache.clear();
    get("clip-screen").style.aspectRatio = options.width + "/" + options.height;
    get("clip-frame").hidden = true; get("clip-message").hidden = false;
    get("clip-message").textContent = "Opening the recording…"; get("clip-position").textContent = "";
    this.controls();
  }
  async cover(frameId) {
    const key=this.key;
    try { const img=await this.image(frameId); if (key===this.key && this.index<0) this.paint(img); }
    catch { /* The timeline load reports a source-frame error with its exact frame ID. */ }
  }
  image(id) {
    if (this.cache.has(id)) return this.cache.get(id);
    const photo=new Image(); photo.alt="Original road footage, frame "+id;
    photo.src="/api/footage-preview/"+this.key+"/"+id;
    const ready=photo.decode().then(()=>photo);
    this.cache.set(id,ready);
    ready.catch(()=>{ if (this.cache.get(id)===ready) this.cache.delete(id); });
    if (this.cache.size>6) this.cache.delete(this.cache.keys().next().value);
    return ready;
  }
  paint(photo) {
    const old=get("clip-frame");
    if (old!==photo) { old.removeAttribute("id"); photo.id="clip-frame"; old.replaceWith(photo); }
    photo.hidden=false; get("clip-message").hidden=true;
  }
  async load(frames,id) {
    this.frames=frames;
    get("clip-seek").max=String(frames.length-1);
    const index=frames.findIndex(row=>row.id===id);
    if (index<0) throw new Error("Frame "+id+" is unavailable in the attached original recording.");
    await this.display(index);
  }
  async display(index) {
    const row=this.frames[index];
    if (!row) return;
    const ticket=++this.ticket;
    this.pending=true; this.controls(); this.changed(false);
    try {
      const photo=await this.image(row.id);
      if (ticket!==this.ticket) return;
      this.paint(photo); this.index=index;
      get("clip-seek").value=String(index);
      get("clip-seek").setAttribute("aria-valuetext","Frame "+row.id+", "+row.time+" seconds");
      get("clip-position").textContent="Frame "+row.id+" · "+elapsed(row.time)+" / "+elapsed(this.frames.at(-1).time);
      for (const ahead of this.frames.slice(index+1,index+3)) void this.image(ahead.id).catch(()=>{});
    } catch {
      if (ticket!==this.ticket) return;
      clearTimeout(this.timer); this.playing=false; this.index=-1;
      get("clip-frame").hidden=true; get("clip-message").hidden=false;
      get("clip-message").textContent="Frame "+row.id+" could not be opened. Choose another frame.";
      get("clip-position").textContent="";
    } finally {
      if (ticket===this.ticket) { this.pending=false; this.controls(); this.changed(); }
    }
    if (ticket===this.ticket && this.playing) this.schedule();
  }
  play() {
    if (this.locked || this.pending || !this.frames.length) return;
    const start=this.index<0 || this.index===this.frames.length-1 ? 0 : this.index;
    this.playing=true; this.origin=performance.now()-1000*this.frames[start].time; this.controls();
    if (start!==this.index) void this.display(start); else this.schedule();
  }
  schedule() {
    const next=this.frames[this.index+1];
    if (!next) { this.pause(); return; }
    const delay=Math.max(0, this.origin+1000*next.time-performance.now());
    this.timer=setTimeout(()=>{ if(this.playing) void this.display(this.index+1); },delay);
  }
}
