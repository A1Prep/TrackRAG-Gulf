"""Offer a passenger a verifiable parking alternative when part of a more specific request is unresolved."""
from io import BytesIO
import re

import cv2
import numpy as np
from PIL import Image

from . import bay_anchors, bay_clearance, bay_intent, bay_landmarks, curb_point_language as words, curb_points, local_surfaces
from . import manual_offlane, manual_surfaces
from .parking_evidence import source_input


POLICY="observed_parking_alternative_v1"
REASON="passenger_choice_of_observed_alternative"
HARD=re.compile(r"\b(?:no|not|never|without|except|unless|must|only|instead|either|or|free|paid|metered|"
                r"permit|reserved|legal|safe|allowed|wheelchair|accessible|cycle|bike|bus stop|taxi rank|"
                r"highway|motorway|lane|carriageway|zebra|crosswalk|crossing|junction|before|after|behind|"
                r"past|beyond|between|nearest|closest|farthest|furthest|first|second|last|if)\b|n['’]t\b",re.I)
GROUND=re.compile(r"\b(?:parking|park|spaces?|spots?|stalls?|bays?|recess|layby|lay-by|pullout|"
                  r"pull-in|gravel|shoulder)\b",re.I)
FAR=re.compile(r",?\s+\b(?:across (?:the (?:street|road) )?from|opposite)\s+(?P<name>[\w' &.-]{2,65})[?.!]*$")
VECTOR=re.compile(r"\b(?:just\s+)?(?:to\s+)?(?:the\s+)?(?P<side>left|right) of\s+(?P<vehicle>[\w -]+?)\s*[?.!]*$")
SIDE=re.compile(r"\b(?:on|toward|to) (?:the |my |our |your )?(left|right)(?:[- ]hand)?(?: side)?\b|"
                r"\b(left|right)[- ](?:hand )?(?:side|bay|space|curb|kerb)\b")
SIGN_COLOR=re.compile(r"\b(orange|yellow|red|blue|green)\s+(?:parking\s+|road\s+|traffic\s+)?sign\b")
SEMANTIC={"wall":6,"tree":30,"palm":30,"greenery":30,"sign":50,"fence":3}
GROUND_WORDS=words.FILLER|words.ARTICLES|{"how","could","would","please","can","empty","use","one","along","with",
     "onto","off","road","paved","unpaved","gravel","dirt","sand","spaces","spots","bays","rows","row","marked",
     "painted","stall","stalls","open","long","tall","palm","wall","tree","sign","fence","greenery","recess",
     "pullout","layby","side","parking","near","by","next","outside","beside","alongside","parked","right","left",
     "known","their","on","find","into","the","a","driver","we","an","at","clear","and","kerb","curb","roadside"}


def intent(text):
    """Parse only constraints that this observation source can check or explicitly clarify."""
    if not isinstance(text,str) or HARD.search(text) or not GROUND.search(text):
        return None
    normalized=words._normal(text).strip(" ?.!")
    far=FAR.search(normalized)
    other=far["name"].strip(" .!?") if far else None
    primary=normalized[:far.start()] if far else normalized
    if (not bay_intent.permitted_destination(primary) or
            re.search(r"\b(?:across|opposite|cross|middle|centre|center|anywhere)\b|"
                      r"\b(?:in|on|into|onto) (?:the |a |this |that )?(?:main |through |service |busy )?(?:road|street|roadway)\b",primary)):
        return None
    relative=VECTOR.search(primary)
    vehicle=words._vehicle(relative["vehicle"],requested=True) if relative else None
    if relative and not vehicle:
        return None
    base=primary[:relative.start()] if relative else primary
    signs=SIGN_COLOR.findall(base)
    if len(set(signs))>1:
        return None
    residue=SIGN_COLOR.sub("sign",base)
    terms=set(words._terms(residue))
    if terms-GROUND_WORDS:
        return None
    # The shared normalizer removes the intensifier in “right by the sign.”
    # An explicit “on the right by the sign” still constrains the actual image side.
    original_primary=re.split(r"\b(?:opposite|across)\b",text.casefold(),maxsplit=1)[0] if other else text.casefold()
    sides={match[1] or match[2] for match in SIDE.finditer(original_primary)
           if not re.match(r"\s+of\b",original_primary[match.end():])}
    if len(sides)>1:
        return None
    anchors=terms & SEMANTIC.keys()
    recess=bool(re.search(r"\b(?:recess|lay[ -]?by|pull[ -]?out)\b",text,re.I))
    if not (vehicle or anchors or recess):
        return None
    # A distant place can orient the passenger only if the independently named primary target stands on its own.
    if other and (not anchors or not re.search(r"\b(?:by|near|beside|next to|alongside)\b",base)):
        return None
    return {"direction":relative["side"] if relative else None,"vehicle":vehicle,"side":next(iter(sides),None),
            "anchors":sorted(anchors),"sign_color":signs[0] if signs else None,"other":other,"recess":recess,
            "parking_sign":bool(re.search(r"\bparking sign\b",base)),"terms":terms,"parked":bool(vehicle and vehicle[3])}


def observed_name(name,bands):
    def normal(value):
        return "".join(re.findall(r"[\w]+",value.casefold()))
    wanted=normal(re.sub(r"^the\s+","",name,flags=re.I))
    return bool(wanted and any(not words.UNCERTAIN.search(part) and normal(part)==wanted
                               for band in bands for part in re.split(r"[|,;]",band)))


def coloured_signs(classes,rgb,colour):
    """Measure the hue on the sign itself. A differently coloured wall or nearby vehicle cannot provide it."""
    h,w=classes.shape
    hsv=cv2.cvtColor(cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2HSV)
    _,labels,stats,centres=cv2.connectedComponentsWithStats(np.isin(classes,[46,49,50]).astype("uint8"))
    intervals={"orange":((6,19),),"yellow":((19,35),),"green":((35,85),),
               "blue":((94,138),),"red":((0,6),(169,180))}
    observed=[]
    for i,(_,_,bw,bh,area) in enumerate(stats[1:],1):
        if bw<4 or bh<4 or area<32:continue
        pixels=hsv[labels==i]
        hue,saturation,value=pixels.T
        desired=np.logical_or.reduce([(hue>=a)&(hue<b) for a,b in intervals[colour]])
        coverage=float(np.mean(desired & (saturation>=64) & (value>=60)))
        if coverage>=.5:
            observed.append((float(centres[i][0])/w,float(centres[i][1])/h))
    return observed


def pictures_landmark(anchor,classes,point,size):
    """Look directly above this pictured parking area, and require a coherent visible object."""
    h,w=classes.shape
    x,y=point[0]/size[0],point[1]/size[1]
    x0,x1=max(0,int((x-.12)*w)),min(w,int((x+.12)*w)+1)
    y0,y1=max(0,int((y-.67)*h)),min(h,int(y*h)+1)
    if x1<=x0 or y1<=y0:return False
    tile=classes[y0:y1,x0:x1]
    ids=(46,49,50) if anchor=="sign" else (SEMANTIC[anchor],)
    _,_,stats,_=cv2.connectedComponentsWithStats(np.isin(tile,ids).astype("uint8"))
    # A sign is a small distinct object. Walls, tree canopies and fences need larger coherent support.
    min_area=max(32,tile.size*(.001 if anchor=="sign" else .025))
    return any(bw>=3 and bh>=4 and area>=min_area for _,_,bw,bh,area in stats[1:])


def seen_vehicle(spec,option,scene):
    desired=spec["vehicle"]
    if not desired:return True
    description=option.get("descriptions") or {}
    stable=(desired[0],desired[1],desired[2],False)
    if not any(words._vehicle_match(stable,description.get(field,"")) is True for field in ("nearer_vehicle","farther_vehicle")):
        return False
    nearby=bay_anchors.closest_tracks(option,bay_anchors.vehicles(scene))
    x=option["point"][0]
    if not nearby:return False
    # The text does not bind a colour to an individual track box. Only use a direction consistent
    # with every plausible nearby vehicle, rather than guessing which box has the described colour.
    centres=[(box[0]+box[2])/2 for _,_,box in nearby]
    return x<min(centres) if spec["direction"]=="left" else x>max(centres)


def clarification(spec,option):
    from . import recessed_kerb
    if option["evidence"]["recipe"]==recessed_kerb.RECIPE:
        side=option["evidence"]["side"]
        if spec["anchors"]:
            unknown="I couldn’t identify the trees as palms. " if "palm" in spec["anchors"] else ""
            return (f"The kerb bends into a recess on the {side}, below greenery and a sign. "
                    f"{unknown}Is the marked point in that recess what you meant?")
        return f"The kerb bends into a recess on the {side}. Is the marked point in that recess what you meant?"
    if spec["vehicle"]:
        kind,colours,sizes,_=spec["vehicle"]
        name=" ".join([*sorted(sizes),*sorted(colours),kind.upper() if kind=="suv" else kind])
        material=next((word for word in ("gravel","sand","dirt","paved","unpaved") if word in spec["terms"]),None)
        missing=(f"the {material} surface" if material else "the exact drop-off area")
        if spec["parked"]:
            missing+=" or whether the vehicle is parked"
        return (f"I can see a {name} among the vehicles beside this parking area. Its marked point is to their {spec['direction']}. "
                f"I couldn’t verify {missing}. Is this the spot you meant?")
    if spec["other"]:
        name=spec["other"]
        shade=(spec["sign_color"]+" ") if spec["sign_color"] else ""
        sign="parking sign" if spec["parking_sign"] else "sign"
        return (f"I found open parking near the {shade}{sign}. I can also read “{name}” elsewhere in the photo. "
                "I couldn’t verify that the spot is across the street from it or trace the whole bay. Is this the spot you meant?")
    names={"wall":"a wall","fence":"a fence","tree":"greenery","palm":"greenery","greenery":"greenery","sign":"a sign"}
    visible=list(dict.fromkeys(names[a] for a in spec["anchors"]))
    side=" on the "+spec["side"] if spec["side"] else ""
    if "palm" in spec["anchors"] and spec["terms"] & {"long","tall"}:
        uncertain="I couldn’t verify the landmark’s size or identify the trees as palms."
    elif "palm" in spec["anchors"]:
        uncertain="I couldn’t identify the trees as palms."
    elif spec["terms"] & {"long","tall"}:
        uncertain="I couldn’t verify the landmark’s size."
    else:
        uncertain="I couldn’t verify the exact position beside the landmark."
    return (f"The photo shows open parking{side}, with {' and '.join(visible)} in that part of the picture. "
            f"{uncertain} Is this the spot you meant?")


def enclosed_points(points,stalls):
    """Keep a point only when its entire ground mark lies inside one twice-reviewed footprint."""
    contours=[np.asarray(stall["polygon"],np.float32) for stall in stalls]
    return [point for point in points if any(all(cv2.pointPolygonTest(contour,tuple(map(float,corner)),False)>=0
                                                 for corner in point["polygon"]) for contour in contours)]


def rebuild(request,photo,surface,point_packet,landmark_packet,*,public_stall_review=None):
    from . import surface_stalls, whole_bay
    clean,key,portable=source_input(request,photo)
    scene=clean["scene"]
    spec=intent(clean["text"])
    local=local_surfaces.matches_scene(scene) and scene["provenance"].get("network_policy")=="local_only"
    if not spec or (not local and (local_surfaces.local_only(scene) or public_stall_review is None)):
        return None
    frame=scene["frame"]; size=(frame["width"],frame["height"])
    bands=bay_landmarks.verify(landmark_packet,photo,size) if landmark_packet else []
    if spec["other"] and not observed_name(spec["other"],bands):
        return None
    classes=manual_surfaces.decode(surface,frame,key)
    connected=bay_clearance.visible_road_component(classes)
    if connected is None:return None
    with Image.open(BytesIO(photo)) as original:
        signs=[]
        if spec["sign_color"]:
            map_size=(classes.shape[1],classes.shape[0])
            original.draft("RGB",map_size)
            sample=original.resize(map_size,Image.Resampling.BOX).convert("RGB")
            signs=coloured_signs(classes,np.asarray(sample),spec["sign_color"])
        if spec["recess"]:
            from . import recessed_kerb, visual_space
            anchors=set(spec["anchors"])
            # A kerb recess can establish geometry alone. Each landmark the passenger added
            # still needs pictured evidence. A signed parking/vehicle claim cannot transfer to it.
            if (not local or spec["vehicle"] or spec["other"] or spec["sign_color"] or spec["parking_sign"] or
                    spec["terms"] & {"sand","dirt","unpaved","gravel","marked","painted"} or
                    anchors and not (anchors<={"tree","palm","greenery","sign"} and "sign" in anchors
                                     and anchors & {"tree","palm","greenery"})):
                return None
            options=[o for o in recessed_kerb.candidates(original,classes,visual_space.native_context(scene))
                     if (not spec["side"] or o["evidence"]["side"]==spec["side"]) and
                        all(pictures_landmark(a,classes,o["point"],size) for a in anchors)]
            if len(options)!=1:return None
            return whole_bay._make(clean,portable,options,surface,method=whole_bay.AUTO,reason=REASON,
                question=clarification(spec,options[0]),require_choice=True,explanation_policy=POLICY,
                local_photo_check=True,curb_point_evidence=None,landmark_evidence=None,
                clearance_policy=whole_bay.CLEARANCE_POLICY)
    if point_packet is None:return None
    options=curb_points.verified_options(scene,photo,surface,point_packet)
    if not local and options:
        # Public point packets checked occupancy only in the wide view. Require a second independent
        # model decision on a close crop of the ground area, authenticated against this exact original.
        _,_,_,shapes=whole_bay._layout(clean,photo,surface,public_stall_review)
        shapes=whole_bay._with_rows(clean,photo,surface,shapes,whole_bay.CLEARANCE_POLICY)
        double_reviewed=[o for o in shapes if o["evidence"]["recipe"]==surface_stalls.RECIPE and
                         bay_clearance.vehicle_sized(o,scene) and bay_clearance.on_visible_road(o,connected,frame)]
        options=enclosed_points(options,double_reviewed)
    if not options:return None
    if spec["sign_color"] and not signs:return None
    h,w=classes.shape
    distance=cv2.distanceTransform(np.pad((classes==10).astype("uint8"),1),cv2.DIST_L2,5)[1:-1,1:-1]
    valid=[]
    for option in options:
        point=option["point"]; x,y=point[0]/size[0],point[1]/size[1]
        if (spec["side"]=="left" and x>=.47 or spec["side"]=="right" and x<=.53 or
                not bay_clearance.on_visible_road(option,connected,frame) or not seen_vehicle(spec,option,scene)):
            continue
        if manual_offlane.surface_check(frame,scene["tracks"],option["box"],point,classes)[0]!="parking":
            continue
        desc=option.get("descriptions",{})
        if any(not pictures_landmark(a,classes,point,size) and
               words.observed_landmark((a,),desc,("roadside","frontage")) is not True for a in spec["anchors"]):
            continue
        if spec["parking_sign"] and words.observed_landmark(("parking","sign"),desc,("roadside",)) is not True:
            continue
        if signs and (words.observed_landmark(("sign",),desc,("roadside",)) is not True or
                      not any(abs(x-sx)<=.25 and sy<y for sx,sy in signs)):
            continue
        clearance=float(distance[min(int(y*h),h-1),min(int(x*w),w-1)])
        valid.append((clearance,option))
    if not valid:return None
    # Multiple sampled points often lie in one continuous patch; show one well-supported interior, not a fictitious row of bays.
    _,best=max(valid,key=lambda entry:entry[0])
    area={**best,"evidence":{**best["evidence"],"coverage":"interior_point_only"}}
    public_evidence={"public_stall_review":public_stall_review} if not local else {}
    return whole_bay._make(clean,portable,[area],surface,method=whole_bay.AUTO,reason=REASON,question=clarification(spec,area),
                           require_choice=True,explanation_policy=POLICY,local_photo_check=local,**public_evidence,
                           curb_point_evidence=point_packet,landmark_evidence=landmark_packet,
                           clearance_policy=whole_bay.CLEARANCE_POLICY)
