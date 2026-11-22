export function addFrame(select, frame) {
  if ([...select.options].some(option => option.value === frame.key)) return;
  const option = document.createElement("option");
  option.value = frame.key;
  option.textContent = frame.recording_id.replace(/^video_/, "Recording ") + " · frame " +
    frame.frame_id + " · " + frame.timestamp_s.toFixed(1) + " s";
  select.append(option);
}

export async function openFrame(select, input) {
  const frameId = input.valueAsNumber;
  if (!Number.isSafeInteger(frameId) || frameId < 0) throw new Error("Enter a whole frame number.");
  const response = await fetch("/api/frames", {method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({scene:select.value,frame_id:frameId})});
  const frame = await response.json();
  if (!response.ok) throw new Error(frame.error || "This frame is unavailable.");
  addFrame(select, frame);
  select.value = frame.key;
  return frame;
}
