"""Make the images actually shown to the reference and gap verification passes."""
from PIL import Image, ImageDraw, ImageFont
from tools.passenger_road_points import render_choices


def _label(draw, xy, label, font):
    x,y=xy
    right=draw.textlength(label,font=font)+8
    draw.rectangle((x,y,x+right,y+21),fill=(18,27,35))
    draw.text((x+4,y+1),label,font=font,fill="white")


def _photo_and_tracks(original, tracks):
    full=original.copy()
    full.thumbnail((1400,820))
    sx,sy=full.width/original.width,full.height/original.height
    draw=ImageDraw.Draw(full)
    font=ImageFont.load_default(size=16)
    for track in reversed(tracks):
        x,y,r,b=track["box"]
        coords=(round(x*sx),round(y*sy),round(r*sx),round(b*sy))
        draw.rectangle(coords,outline=(244,169,63),width=2)
        _label(draw,(min(coords[0],full.width-38),max(0,coords[1]-18)),str(track["id"]),font)
    return full


def _vehicles(original, tracks):
    source=_photo_and_tracks(original,tracks)
    cols=6
    tw=max(120,source.width//cols)
    th=100
    sheet=Image.new("RGB",(tw*cols,source.height+th*((len(tracks)+cols-1)//cols)),(26,32,36))
    sheet.paste(source,(0,0))
    draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=17)
    for i,track in enumerate(tracks):
        x,y,r,b=track["box"]
        inset=(max(0,int(x-(r-x)*.08)),max(0,int(y-(b-y)*.08)),
               min(original.width,int(r+(r-x)*.08)+1),min(original.height,int(b+(b-y)*.08)+1))
        tile=original.crop(inset)
        tile.thumbnail((tw-4,th-4))
        left=(i%cols)*tw;top=source.height+(i//cols)*th
        sheet.paste(tile,(left+(tw-tile.width)//2,top+(th-tile.height)//2))
        _label(draw,(left+2,top+2),str(track["id"]),font)
    return sheet


def _inspection(original, option):
    marked=original.copy()
    draw=ImageDraw.Draw(marked)
    stroke=max(2,round(original.width/680))
    for colour,box in zip(("#e8ac47","#59e2fc"),option["anchor_boxes"]):
        draw.rectangle(tuple(box),outline=colour,width=stroke)
    x,y=option["point"]
    radius=stroke*2
    draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill="white",outline="black",width=stroke)
    boxes=[*option["anchor_boxes"],option["region_box_xyxy"]]
    x0=min(b[0] for b in boxes);y0=min(b[1] for b in boxes)
    x1=max(b[2] for b in boxes);y1=max(b[3] for b in boxes)
    dx=max(2.,(x1-x0)*.18);dy=max(2.,(y1-y0)*.45)
    crop=marked.crop((max(0,int(x0-dx)),max(0,int(y0-dy)),min(marked.width,int(x1+dx)+1),min(marked.height,int(y1+dy)+1)))
    marked.thumbnail((1152,600));crop.thumbnail((1152,460))
    sheet=Image.new("RGB",(max(marked.width,crop.width),marked.height+crop.height+3),(24,30,36))
    sheet.paste(marked,((sheet.width-marked.width)//2,0))
    sheet.paste(crop,((sheet.width-crop.width)//2,marked.height+3))
    return sheet


def render(original_path, frame, view, out):
    kind=view["type"]
    if kind=="original":
        return original_path
    if kind in ("points", "parking_components"):
        render_choices(original_path,view["options"],(frame["width"],frame["height"]),out)
        return out
    with Image.open(original_path) as infile:
        img=infile.convert("RGB")
    if kind=="vehicles":
        rendered=_vehicles(img,view["tracks"])
    elif kind=="inspection":
        rendered=_inspection(img,view["option"])
    else:
        raise ValueError("Unknown parking inference view")
    rendered.save(out,format="JPEG",quality=94)
    return out
