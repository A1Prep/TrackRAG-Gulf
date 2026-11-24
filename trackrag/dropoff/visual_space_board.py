"""Two independent views of original ground points for the public-photo parking study."""
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont


RECIPE = "two_view_public_pullin_observation_v1"
PAVED_RECIPE = "native_pullin_context_observation_v1"
EXTENSION_RECIPE = "native_entrance_or_row_observation_v1"
PAVED_GUIDE = (
    "Read the two VEHICLE crops along the bottom and the matching street and frontage above them. "
    "Describe what is visible, independently of the suitability of the parking location. "
    "Reply with one JSON object containing these strings: left_vehicle (paint colour and type of the vehicle in the "
    "BOTTOM LEFT crop, plus visible commercial livery), right_vehicle (same for the BOTTOM RIGHT crop), "
    "rear_vehicle (colour and type only if the ring is immediately behind the rear of a specific vehicle, else NONE), "
    "frontage (type and readable name of premises by the ring), roadside (other directly adjacent features). "
    "For each label use NONE if unrecognizable. Do not include plates, phone numbers, addresses or people."
)
EXTENSION_GUIDE = (
    "The yellow ring marks an inspected patch of ground. The two bottom panels magnify actual vehicles, "
    "ordered by their position in the image. Describe their colour and vehicle type, and the premises or "
    "other objects immediately beside the marked ground. Do not assume either vehicle flanks the ring. "
    "Return a JSON object with strings: left_vehicle (bottom left crop), right_vehicle (bottom right crop), "
    "rear_vehicle (colour/type only if the patch is clearly behind the rear of THAT vehicle, otherwise NONE), "
    "frontage (premises category and readable name), roadside (nearby objects or paving). Use NONE when unclear. "
    "Omit phone numbers, addresses, people and plates."
)
SELECT = (
    "A passenger wants to get out near this street. The upper view is the original scene. "
    "Below is an enlarged side of the same scene. Each number labels the small yellow dot at the end of its line. "
    "Select up to two dots in DISTINCT actually visible empty vehicle-sized parking spaces or pull-ins, where a driver "
    "could leave the through lane on this side. Dots must lie ON THE GROUND inside the space. "
    "Reject a dot in flowing traffic, the entrance lane of a gate, on a sidewalk, behind a barrier, next to a bumper without "
    "enough empty ground, or on bare desert or a highway shoulder. Many photos have no valid dots. "
    "A marked parking stall, a gap in a parked row, or a visibly accessible paved or unpaved lay-by can qualify. "
    "No need to pick a dot just because it is labelled. Return exactly JSON with selected as a list of integer numbers, "
    "e.g. {\"selected\":[]} when none. Do not assess permission or safe driving."
)
REVIEW = (
    "Inspect the ground at the centre of the yellow ring in this UAE street photograph. The second panel magnifies that "
    "exact part of the original. Is it inside a distinct, unoccupied, vehicle-sized place a vehicle could pull into OFF "
    "the travel lane on the camera side? Look for a visible entrance AND room in the marked space. Reject a strip behind "
    "a bumper, moving lanes, an entrance queue, the far side of a continuous curb or footway, and walkways. "
    "Return just JSON with these string fields: status (OPEN, BLOCKED, LANE, SEPARATED or UNCLEAR), "
    "edge (painted bay, recessed curb, different surface, parked row or none), "
    "left_vehicle (colour/type immediately LEFT of the dot along the same parking row, or NONE), "
    "right_vehicle (colour/type immediately RIGHT of the dot along that same row, or NONE), "
    "rear_vehicle (colour/type if the dot is immediately BEHIND the visible REAR of one vehicle, otherwise NONE), "
    "frontage (nearby business type and readable sign beside the dot), roadside (other visible landmarks directly by the dot, "
    "including distinctive colours, markings, bin, pavement, trees or divider where visible). Use NONE for absent information. "
    "No licence plates, addresses, telephone numbers or people."
)


def _panel(original, crop, size):
    image = original.crop(crop).convert("RGB")
    image.thumbnail(size,Image.Resampling.LANCZOS)
    slope = image.width/(crop[2]-crop[0]), image.height/(crop[3]-crop[1])
    return image, lambda p: ((p[0]-crop[0])*slope[0], (p[1]-crop[1])*slope[1])


def _encoded(canvas):
    target = BytesIO()
    canvas.save(target,"JPEG",quality=90,subsampling=0)
    return target.getvalue()


def selection(original, samples, side):
    w,h = original.size
    upper,_ = _panel(original,(0,0,w,h),(1200,520))
    x0,x1 = (0,.65*w) if side=="left" else (.35*w,w)
    bottom = h if any(s.get("vehicles") for s in samples) else round(.965*h)
    lower, xy = _panel(original,(round(x0),round(.34*h),round(x1),bottom),(1200,690))
    draw = ImageDraw.Draw(lower)
    font = ImageFont.load_default(size=16)
    for s in samples:
        x,y=xy(s["point"])
        if not (0<=x<lower.width and 0<=y<lower.height):
            raise ValueError("Ground point outside its inspection crop.")
        text=str(s["id"])
        lx,ly=max(1,min(x+5,lower.width-27)),max(0,y-21)
        draw.line((x,y,lx+3,ly+12),fill="black",width=3)
        draw.ellipse((x-3,y-3,x+3,y+3),fill="yellow",outline="black",width=1)
        draw.rectangle((lx,ly,lx+max(13,len(text)*10),ly+17),fill="#102a35")
        draw.text((lx+1,ly-1),text,font=font,fill="white")
    sheet=Image.new("RGB",(max(upper.width,lower.width),upper.height+lower.height+5),"#ededed")
    sheet.paste(upper,((sheet.width-upper.width)//2,0))
    sheet.paste(lower,((sheet.width-lower.width)//2,upper.height+5))
    return _encoded(sheet)


def inspection(original, sample):
    w,h=original.size
    x,y=sample["point"]
    span_x,span_y=.47*w,.48*h
    left=min(max(0,x-span_x/2),w-span_x)
    top=min(max(0,y-.62*span_y),h-span_y)
    views=[]
    for rect,size in [((0,0,w,h),(1200,590)),((round(left),round(top),round(left+span_x),round(top+span_y)),(1200,620))]:
        panel,xy = _panel(original,rect,size)
        px,py=xy((x,y)); draw=ImageDraw.Draw(panel)
        draw.ellipse((px-7,py-7,px+7,py+7),outline="#101010",width=3)
        draw.ellipse((px-5,py-5,px+5,py+5),outline="yellow",width=2)
        views.append(panel)
    witnesses = sample.get("vehicles") or []
    extra = 154 if witnesses else 0
    sheet=Image.new("RGB",(max(v.width for v in views),sum(v.height for v in views)+5+extra),"#ededed")
    sheet.paste(views[0],((sheet.width-views[0].width)//2,0))
    sheet.paste(views[1],((sheet.width-views[1].width)//2,views[0].height+5))
    if witnesses:
        pen=ImageDraw.Draw(sheet)
        for i,(_, box) in enumerate(witnesses):
            left=i*sheet.width//2
            view,_ = _panel(original,tuple(map(round,box)),(sheet.width//2,131))
            sheet.paste(view,(left+(sheet.width//2-view.width)//2,sheet.height-131))
            label = ("Vehicle crop from original, " + ("left" if i==0 else "right") if sample.get("native_option")
                     else ("Left" if i==0 else "Right")+" vehicle in the measured row")
            pen.text((left+8,sheet.height-153),label,
                     fill="black",font=ImageFont.load_default(size=16))
    return _encoded(sheet)
