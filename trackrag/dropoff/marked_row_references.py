"""Offer pictured, paint-bounded parking interiors when a passenger names a wall or a foreground vehicle."""
from io import BytesIO
import re

import cv2
import numpy as np
from PIL import Image

from . import bay_anchors, bay_clearance, bay_intent, curb_alternatives, curb_point_language as words
from . import curb_points, local_surfaces, manual_offlane, manual_surfaces, painted_bays
from .parking_evidence import source_input


POLICY = 'pictured_marked_row_and_nearby_reference_v1'
PAINT_POINT = 'open_point_between_two_visible_parking_strokes_v1'
COLOURS = {'pink','orange','yellow','red','green','blue','white','black','gray','brown','beige'}
WALL = re.compile(r'\b(?:by|near|beside|alongside|along|next to)\s+(?:the |that |this )?(?:(\w+)\s+)?wall\b')
NEAR = re.compile(r'\b(?:on )?(?:(?:this|our|near) side of|closer to (?:us|me) than|nearer to (?:us|me) than)\s+([\w -]+)$')
SIDE = re.compile(r'\b(?:on|towards?|to) (?:the |my |our )?(left|right)(?:[- ]hand)?(?: side)?\b')
GROUND_WORDS = words.FILLER | words.ARTICLES | {'marked','painted','lined','spaces','bays','stalls','stall','in','one','any'}


def intent(text):
    if not isinstance(text,str) or curb_alternatives.HARD.search(text) or not bay_intent.permitted_destination(text):
        return None
    sentence=' '.join(words._normal(text).replace(',',' ').strip(' .!?').split())
    if not re.search(r'\b(?:marked|painted|lined)\b',sentence) or not re.search(r'\b(?:parking|bays?|spaces?|stalls?)\b',sentence):
        return None
    sides=SIDE.findall(sentence)
    if len(sides)>1:return None
    primary=' '.join(SIDE.sub(' ',sentence).split())
    wall=WALL.search(primary)
    near=NEAR.search(primary)
    if not wall and not near:
        if set(words._terms(primary))<=GROUND_WORDS:
            return {'kind':'marked','side':sides[0] if sides else None}
        return None
    if wall and near:return None
    anchor=wall or near
    remaining=primary[:anchor.start()]+' '+primary[anchor.end():]
    if set(words._terms(remaining))-GROUND_WORDS:return None
    if wall:
        colour=wall[1]
        if colour and colour not in COLOURS:return None
        return {'kind':'wall','colour':colour,'side':sides[0] if sides else None}
    vehicle=words._vehicle(near[1],requested=True)
    if not vehicle or len(vehicle[1])!=1 or vehicle[2] or vehicle[3] or not vehicle[1].issubset(COLOURS):return None
    return {'kind':'near_vehicle','vehicle':vehicle,'side':sides[0] if sides else None}


def _sample(photo, classes):
    h,w=classes.shape
    with Image.open(BytesIO(photo)) as original:
        original.draft('RGB',(w,h))
        return np.asarray(original.resize((w,h),Image.Resampling.BOX).convert('RGB'))


def _colour(mask, rgb, name):
    """Return the fraction of actual object pixels compatible with a basic colour and their image footprint."""
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    h,s,v=cv2.split(hsv)
    hue={'red':(h<=9)|(h>=169),'orange':(h>=7)&(h<=20),'yellow':(h>=20)&(h<=36),
         'green':(h>=37)&(h<=87),'blue':(h>=94)&(h<=135),'pink':(h>150)|(h<7),
         'brown':(h>=5)&(h<=21),'beige':(h>=7)&(h<=30)}
    if name=='white':selected=(s<=60)&(v>=169)
    elif name=='gray':selected=(s<=50)&(v>46)&(v<193)
    elif name=='black':selected=v<48
    elif name=='pink':selected=hue[name]&(s>18)&(s<176)&(v>=95)
    elif name=='beige':selected=hue[name]&(s>16)&(s<93)&(v>125)
    elif name=='brown':selected=hue[name]&(s>=58)&(v>=40)&(v<=170)
    else:selected=hue[name]&(s>=65)&(v>=37)
    visible=selected & mask
    return float(visible.sum()/max(1,int(mask.sum()))),visible


def wall_colour(option, classes, rgb, size, requested):
    """Require wall pixels above this bay. A warm, low-chroma wall can support an explicitly uncertain pink request."""
    h,w=classes.shape
    x,y=np.asarray(option['point'])*[w/size[0],h/size[1]]
    bbox=np.zeros_like(classes,bool)
    bbox[max(0,round(y-.44*h)):round(y),max(0,round(x-.10*w)):min(w,round(x+.10*w))]=True
    mask=(classes==6)&bbox
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'))
    if n<2:return None
    idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]);l,t,bw,bh,area=stats[idx]
    if (area<.009*w*h or bw<.07*w or bh<.07*h or y-(t+bh)>.25*h):return None
    mask=labels==idx
    if not requested:return 'observed'
    fraction,_=_colour(mask,rgb,requested)
    if fraction>=.54:return 'observed'
    if requested=='pink':
        red,green,blue=rgb.astype('int16').transpose(2,0,1)
        # Pale, sunlit masonry may be peach or pink. Never label it pink on this evidence alone.
        pale_warm=(red>=145)&(red-green>=8)&(red-blue>=12)&(red-blue<=98)&(green-blue<=48)
        if np.mean(pale_warm[mask])>=.53:return 'pale_warm'
    return None


def _visible_dividers(classes, rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype('float32')
    r,g,b=rgb.astype('float32').transpose(2,0,1)
    neutral=(np.ptp(rgb,axis=2)<55)|((r>1.12*b)&(g>1.08*b)&(abs(r-g)<45))
    ground=np.isin(classes,(10,13,24))
    distance=cv2.distanceTransform((classes!=10).astype('uint8'),cv2.DIST_L2,5)
    lines=painted_bays._dividers(gray,neutral,ground,classes,distance)
    return [line['ends'][np.argsort(line['ends'][:,1])] for line in lines if line['semantic']
            and np.ptp(line['ends'][:,1])>=.05*classes.shape[0]]


def _x_at(line, y):
    a,b=line
    return float(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1]))


def bracketing_pair(xy,lines,classes,road,occupied):
    """Find adjacent photographed dividers at the point depth without projecting past any visible endpoint."""
    x,y=xy;h,w=classes.shape
    observed=sorted(((_x_at(line,y),i) for i,line in enumerate(lines) if line[0,1]<=y<=line[1,1]))
    left=[(pos,i) for pos,i in observed if pos<x];right=[(pos,i) for pos,i in observed if pos>x]
    if not left or not right:return None
    l,i=left[-1];r,j=right[0]; a,b=lines[i],lines[j]
    width=r-l;fraction=(x-l)/width
    if not .055*w<=width<=.45*w or not .10<fraction<.90:return None
    top,bottom=max(a[0,1],b[0,1]),min(a[1,1],b[1,1])
    if bottom-top<.055*h:return None
    back=_x_at(b,top)-_x_at(a,top);front=_x_at(b,bottom)-_x_at(a,bottom)
    if back<.035*w or front<.97*back:return None
    # Reject parking symbols, fragments alongside traffic, obstructions and the two outer edges of an island.
    for yy in np.linspace(top+1,bottom-1,9):
        low,high=_x_at(a,yy),_x_at(b,yy)
        xs=np.rint(np.linspace(low+.10*(high-low),high-.10*(high-low),35)).astype(int)
        iy=round(yy)
        if (min(xs)<0 or max(xs)>=w or not road[iy,xs].all() or occupied[iy,xs].any() or
                (classes[iy,xs]==10).mean()<.84):return None
    return (i,j),abs(fraction-.5),[a.round(3).tolist(),b.round(3).tolist()]


def _occupied(scene,shape):
    h,w=shape;f=scene['frame'];occupied=np.zeros(shape,bool);scale=[w/f['width'],h/f['height']]
    for t in scene['tracks']:
        if t['frame_id']!=f['frame_id']:continue
        l,top,r,b=np.asarray(t['bbox_xyxy'])*np.tile(scale,2)
        occupied[max(0,int(top)):min(h,int(np.ceil(b))+1),max(0,int(l)):min(w,int(np.ceil(r))+1)]=True
    return occupied


def _checked_point_choices(spec,scene,classes,rgb,road,points):
    f=scene['frame'];size=(f['width'],f['height']);h,w=classes.shape
    lines=_visible_dividers(classes,rgb);groups={};occupied=_occupied(scene,classes.shape)
    for option in points:
        xy=np.asarray(option['point'])*[w/size[0],h/size[1]]
        if not _side(xy[0]/w,spec['side']):continue
        if manual_offlane.surface_check(f,scene['tracks'],option['box'],option['point'],classes)[0]!='parking':continue
        match=bracketing_pair(xy,lines,classes,road,occupied)
        if match is None:continue
        wall=wall_colour(option,classes,rgb,size,spec['colour']) if spec['kind']=='wall' else None
        if spec['kind']=='wall' and not wall:continue
        shape,rank,strokes=match
        proof={'recipe':PAINT_POINT,'coverage':'interior_point_only','paint_in_map':strokes}
        if wall:proof['wall_colour_support']=wall
        area={**option,'evidence':proof}
        if shape not in groups or rank<groups[shape][0]:groups[shape]=(rank,area)
    return [entry[1] for entry in sorted(groups.values(),key=lambda entry:entry[1]['point'])]


def _side(x,side):
    return not side or side=='right' and x>.53 or side=='left' and x<.47


def colour_vehicle(spec,points,scene,classes,rgb):
    """Bind the neutral vehicle description to one current body, excluding overlapping vehicles and lights."""
    wanted=spec['vehicle'];colour=next(iter(wanted[1]));f=scene['frame'];h,w=classes.shape
    options=[p for p in points if any(words._vehicle_match(wanted,p['descriptions'].get(name,'')) is True
                                      for name in ('nearer_vehicle','farther_vehicle'))]
    if not options:return None
    vehicles=bay_anchors.vehicles(scene);hits=[]
    plausible={t[0] for o in options for t in bay_anchors.closest_tracks(o,vehicles)}
    size=np.asarray([f['width'],f['height']]);scale=np.asarray([w,h])/size
    for tid,kind,box in vehicles:
        l,t,r,b=box;bw,bh=r-l,b-t
        if (tid not in plausible or bw<.045*size[0] or bh<.045*size[1] or
                not _side((l+r)/(2*size[0]),spec['side'])):continue
        name=wanted[0]
        if name=='vehicle':classes_for_type=(54,55,59,61)
        elif name in {'car','suv','sedan','hatchback','coupe','taxi','van','minivan'}:classes_for_type=(55,)
        elif name in {'pickup','truck','flatbed'}:classes_for_type=(55,61)
        elif name=='bus':classes_for_type=(54,)
        else:classes_for_type=()
        if not classes_for_type:continue
        mask=np.zeros(classes.shape,bool)
        # Paint should span the body. Exclude the roofline, windscreen and tyres as well as
        # pixels actually shared with other tracked vehicles, which can donate another colour.
        x0,y0,x1,y1=np.rint(np.array([l+.06*bw,t+.33*bh,r-.06*bw,t+.82*bh])*np.tile(scale,2)).astype(int)
        x0,x1=max(0,x0),min(w,x1);y0,y1=max(0,y0),min(h,y1)
        if x1-x0<9 or y1-y0<5:continue
        mask[y0:y1,x0:x1]=True
        for other,_,bb in vehicles:
            if other==tid:continue
            a,c,e,g=np.rint(np.asarray(bb)*np.tile(scale,2)).astype(int)
            mask[max(0,c):min(h,g+1),max(0,a):min(w,e+1)]=False
        mask &= np.isin(classes,classes_for_type)
        if mask.sum()<max(24,.28*(x1-x0)*(y1-y0)):continue
        fraction,visible=_colour(mask,rgb,colour)
        spans=np.linspace(x0,x1,4).round().astype(int)
        broad=sum(visible[y0:y1,a:e].sum()>=.12*max(1,mask[y0:y1,a:e].sum())
                  and mask[y0:y1,a:e].sum()>=.07*mask.sum() for a,e in zip(spans[:-1],spans[1:]))
        if fraction>=.20 and broad>=2:hits.append((tid,box))
    return hits[0] if len(hits)==1 else None


def _measured_row_choices(spec,scene,classes,photo,road,bound=None):
    from . import painted_row_interiors
    f=scene['frame'];w,h=f['width'],f['height']
    with Image.open(BytesIO(photo)) as original:
        areas=painted_row_interiors.candidates(original,classes,scene)
    options=[]
    for option in areas:
        x,y=option['point']
        if (not _side(x/w,spec['side']) or not bay_clearance.on_visible_road(option,road,f) or
                manual_offlane.surface_check(f,scene['tracks'],option['box'],option['point'],classes)[0]!='parking'):
            continue
        proof={**option['evidence'],'coverage':'interior_point_only'}
        if bound:
            tid,(l,t,r,b)=bound
            # Only the near side as it appears in the image, without a claim about physical depth.
            if y<=b+.012*h or not l-.4*(r-l)<=x<=r+.4*(r-l):continue
            proof['vehicle_reference_track']=tid
        options.append({**option,'evidence':proof})
    return sorted(options,key=lambda o:(o['point'][1],o['point'][0]))


def rebuild(request,photo,surface,packet):
    from . import whole_bay
    clean,key,source=source_input(request,photo);scene=clean['scene'];spec=intent(clean['text'])
    if (not spec or packet is None or not local_surfaces.matches_scene(scene) or
            scene['provenance'].get('network_policy')!='local_only'):return None
    frame=scene['frame'];classes=manual_surfaces.decode(surface,frame,key)
    road=bay_clearance.visible_road_component(classes)
    if road is None:return None
    confirmed=curb_points.verified_options(scene,photo,surface,packet)
    if not confirmed:return None
    rgb=_sample(photo,classes)
    if spec['kind']=='marked':
        choices=_checked_point_choices(spec,scene,classes,rgb,road,confirmed)
        if not choices:choices=_measured_row_choices(spec,scene,classes,photo,road)
        if not choices:return None
        side=' on the '+spec['side'] if spec['side'] else ''
        question=f'I found parking between visible painted lines{side}. I can mark the interior, but the full outline is uncertain. '
    elif spec['kind']=='wall':
        choices=_checked_point_choices(spec,scene,classes,rgb,road,confirmed)
        if not choices:return None
        ambiguous=any(o['evidence']['wall_colour_support']=='pale_warm' for o in choices)
        colour='light, warm-coloured' if ambiguous else spec['colour'] or 'visible'
        question=f'I found parking between painted lines beside the {colour} wall. '
        if ambiguous:question+='I couldn’t verify that the wall is pink in this photo. '
    else:
        bound=colour_vehicle(spec,confirmed,scene,classes,rgb)
        if not bound:return None
        choices=_measured_row_choices(spec,scene,classes,photo,road,bound)
        if not choices:return None
        vehicle=spec['vehicle'];name=next(iter(vehicle[1]))+' '+(vehicle[0].upper() if vehicle[0]=='suv' else vehicle[0])
        side=' on the '+spec['side'] if spec['side'] else ''
        question=(f'I can see a {name} and painted parking spaces{side}. '
                  'The marked parking appears on this side of the vehicle in the photo. ')
    # A menu must not imply a unique area when there are many non-equivalent options.
    if len(choices)>3:return None
    question+='Is this the spot you meant?' if len(choices)==1 else 'Which marked spot did you mean?'
    return whole_bay._make(clean,source,choices,surface,method=whole_bay.AUTO,reason=curb_alternatives.REASON,
                           question=question,require_choice=True,explanation_policy=POLICY,local_photo_check=True,
                           marked_row_point_evidence=packet,clearance_policy=whole_bay.CLEARANCE_POLICY)
