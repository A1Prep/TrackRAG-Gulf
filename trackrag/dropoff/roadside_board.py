"""The same visual evidence board is rendered at collection and replay."""
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

RECIPE = "public_roadside_context_qwen35_v1"
REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
REVISION = "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2"
GUIDE = (
    "Inspect this driver-view evidence board. The top photo shows the exact yellow ring and a cyan outline if a "
    "parking stall has been traced. The middle is enlarged original frontage around this point. If present, the two "
    "bottom crops show the actual flanking vehicles from left to right. Read each supplied crop independently. "
    "Reply with a single JSON object with these string keys: status, nearer_vehicle, farther_vehicle, frontage, roadside. "
    "status is OPEN only for a visibly unused recessed area or parking space reachable from the camera's road. "
    "Use LANE, BLOCKED, SEPARATED, or UNCLEAR otherwise. A narrow strip left around a parked car is BLOCKED. "
    "For each flanking vehicle give actual colour and specific type, e.g. flatbed truck or hatchback. If crops "
    "are absent, describe only the two ends of the same parking space if visible. Use NONE for missing details. "
    "For frontage give its business category and exact sign words only if legible and directly beside the marked "
    "space. For roadside name nearby visible features at the marked position, e.g. a low divider, pavement, "
    "trees, bin, gate or gravel. No licence plates, telephone numbers or people. Short phrases only.")


def render(original, spec):
    image=original.convert("RGB")
    w,h=image.size
    x,y=spec["point"]
    panel=Image.new("RGB",(1120,1100),"#eeeae1")
    font=ImageFont.load_default(size=15)
    def fit(picture,rect):
        x0,y0,x1,y1=rect
        picture=picture.copy()
        picture.thumbnail((x1-x0,y1-y0),Image.Resampling.LANCZOS)
        pos=(x0+(x1-x0-picture.width)//2,y0+(y1-y0-picture.height)//2)
        panel.paste(picture,pos)
        return pos,picture.size
    top,(tw,th)=fit(image,(0,0,1120,570))
    draw=ImageDraw.Draw(panel)
    def on_top(p):
        return (top[0]+p[0]*tw/w,top[1]+p[1]*th/h)
    if spec["kind"]=="stall":
        poly=[on_top(p) for p in spec["polygon"]]
        draw.line(poly+[poly[0]],fill="cyan",width=2)
    px,py=on_top((x,y))
    draw.ellipse((px-8,py-8,px+8,py+8),outline="black",width=4)
    draw.ellipse((px-6,py-6,px+6,py+6),outline="yellow",width=2)
    span=.66*w
    left=max(0,min(w-span,x-span/2))
    bottom=min(h,max(.4*h,y+.03*h)); upper=max(0,bottom-.44*h)
    fit(image.crop((round(left),round(upper),round(left+span),round(bottom))),(0,592,1120,870))
    draw.text((12,573),"Frontage near the ring",font=font,fill="#101818")
    for i,(_,box) in enumerate(spec["vehicles"]):
        crop=image.crop(tuple(map(round,box)))
        fit(crop,(560*i,894,560*(i+1),1100))
        draw.text((560*i+12,873),"Vehicle "+str(i+1),font=font,fill="#101818")
    if not spec["vehicles"]:
        panel=panel.crop((0,0,1120,873))
    out=BytesIO()
    panel.save(out,format="JPEG",quality=93,subsampling=0)
    return out.getvalue()
