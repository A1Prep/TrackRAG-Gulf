"""Use enlarged, current-frame visual evidence for a second landmark decision."""
from hashlib import sha256
import json
import math

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .landmarks import landmark_messages, parse_landmark
from .model import _materialize_image


GUIDE = """You are checking a passenger's drop-off reference before showing a suggested area. In the photograph and crops, cyan boxes and S labels identify the candidate landmarks a first pass selected. Numbers beginning with L label detector candidates for this frame. Read the actual pixels. Check the requested colour, object type, count, side and relation. Look for visible competing matches. Use scene context for a structure absent from the candidates. If a vehicle is selected for a building or sign, they do not match.
Return just ONE line:
MATCH if the selected landmark(s) visibly match and the passenger distinguished them adequately.
LANDMARK L3 (or LANDMARK L3 L7) to correct an identification using actual displayed detector IDs.
ASK followed by a specific short question if there are multiple plausible visible matches, or you need a distinction.
MISSING if the requested object is absent or cannot be identified from the visible evidence.
Never invent a feature or ask the passenger whether an invisible object is present. A plausible location still needs passenger confirmation."""


def _font():
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf",14)
    except OSError:
        return ImageFont.load_default(size=14)


def evidence_board(image, index, first):
    board = Image.new("RGB",(672,384),(21,24,27))
    draw = ImageDraw.Draw(board)
    font = _font()
    scene = ImageOps.contain(image,(436,244),Image.Resampling.LANCZOS)
    ox, oy = (436-scene.width)//2, (246-scene.height)//2
    board.paste(scene,(ox,oy))
    sx, sy = scene.width/image.width,scene.height/image.height
    crops, already = [],set()
    ids = {t["track_id"]:key for key,t in index.items()}
    for i, box in enumerate(first["boxes"]):
        track_ids = first.get("track_ids") or []
        label = ids[track_ids[i]] if i < len(track_ids) else None
        if label:
            already.add(label)
        x1,y1,x2,y2 = box
        draw.rectangle([ox+x1*sx,oy+y1*sy,ox+x2*sx,oy+y2*sy],outline="cyan",width=2)
        crops.append((box,f"S{i+1}"+(" / "+label if label else "")))
    def area(item):
        a,b,c,d = item[1]["bbox_xyxy"]
        return (c-a)*(d-b)
    other = [(key,t) for key,t in sorted(index.items(),key=area,reverse=True) if key not in already][:8]
    def tile(box,label,left,top,width,height):
        x1,y1,x2,y2 = box
        dx,dy = .12*(x2-x1),.12*(y2-y1)
        coords = (max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),
                  min(image.width,math.ceil(x2+dx)),min(image.height,math.ceil(y2+dy)))
        patch = ImageOps.contain(image.crop(coords),(width-4,height-21),Image.Resampling.LANCZOS)
        board.paste(patch,(left+(width-patch.width)//2,top+17+(height-20-patch.height)//2))
        draw.text((left+4,top+1),label,font=font,fill="white")
    for i,(box,label) in enumerate(crops):
        tile(box,label,440,122*i,230,122)
    for i,(key,t) in enumerate(other):
        tile(t["bbox_xyxy"],key,i*84,249,84,133)
    return board, [x[0] for x in other]


def check_landmark(resolver, request, image, first):
    _, index = landmark_messages(request)
    board, alternatives = evidence_board(image,index,first)
    original_decision = {k:v for k,v in first.items() if k in ("kind","track_ids","boxes")}
    context = {"prior_dialogue":[{"role":t["role"],"text":t["text"]} for t in request["dialogue"]],
               "passenger":request["text"],"candidate_boxes_original_pixels":original_decision,
               "image_size":list(image.size),"visible_alternative_ids":alternatives}
    messages = [{"role":"system","content":GUIDE},{"role":"user","content":[
        {"type":"image","image":"current-frame-evidence-board"},
        {"type":"text","text":json.dumps(context,ensure_ascii=False)}]}]
    with board:
        evidence = {"render_sha256":sha256(board.tobytes()).hexdigest(),"render_size":list(board.size),
                    "messages":messages,"chosen":original_decision}
        response = resolver.backend.generate(_materialize_image(messages,board),config=resolver.config,contract="landmark_v1")
    evidence.update(raw_text=response.text,backend_details=response.details)
    line = response.text.strip()
    if line == "MATCH":
        return first,evidence
    # Only proposed IDs visible in the crop panel may replace the initial set.
    visible = {k:t for k,t in index.items() if k in alternatives or t["track_id"] in first["track_ids"]}
    try:
        decision = parse_landmark(line,visible,request["scene"]["frame"])
        if decision["kind"] not in ("ask","missing","landmark"):
            raise ValueError("Visual check returned an unsupported correction")
    except ValueError as exc:
        evidence["parse_error"] = str(exc)
        return None,evidence
    return decision,evidence
