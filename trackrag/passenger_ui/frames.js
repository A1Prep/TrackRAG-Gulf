export function addFrame(select, frame) {
  const existing = [...select.options].find(option => option.value === frame.key);
  const option = existing || document.createElement("option");
  option.value = frame.key;
  const drive = frame.recording_id.match(/^kartaview_(\d+)$/);
  option.textContent = frame.display_name || (drive ? "Street photo " + drive[1] + " · " + frame.frame_id :
    frame.recording_id.replace(/^video_/, "Recording ") + " · frame " + frame.frame_id + " · " + frame.timestamp_s.toFixed(1) + " s");
  if (frame.parking_areas_indexed) option.textContent += " · Marked bays";
  if (!existing) select.append(option);
}

export async function openFrame(select, input, forFootage = false) {
  const frameId = input.valueAsNumber;
  if (!Number.isSafeInteger(frameId) || frameId < 0) throw new Error("Enter a whole frame number.");
  const response = await fetch("/api/frames", {method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({scene:select.value,frame_id:frameId,for_footage:forFootage})});
  const frame = await response.json();
  if (!response.ok) throw new Error(frame.error || "This frame is unavailable.");
  addFrame(select, frame);
  select.value = frame.key;
  return frame;
}
