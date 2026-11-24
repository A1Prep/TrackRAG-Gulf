"""Resolve everyday curb, row and landmark details from measured scene evidence."""
from io import BytesIO
import re

import cv2
import numpy as np
from PIL import Image

from . import single_vehicle


# IDs use the pinned Mask2Former Mapillary Vistas v1.2 label set.
LANDMARKS = {
    'bin': (51,), 'bins': (51,), 'trash can': (51,), 'trash cans': (51,),
    'rubbish bin': (51,), 'rubbish bins': (51,), 'garbage bin': (51,), 'garbage bins': (51,),
    'bench': (33,), 'benches': (33,), 'hydrant': (38,), 'fire hydrant': (38,),
    'mailbox': (40,), 'traffic light': (48,), 'traffic lights': (48,),
    'traffic sign': (49,50), 'road sign': (49,50), 'triangular road sign': (49,50),
    'triangular traffic sign': (49,50), 'triangular sign': (49,50), 'street light': (44,),
}
SIDE = re.compile(r'\b(left|right)(?:[- ]hand)?(?: side)?\b',re.I)
ROW = re.compile(r'^(?:parked )?(?:in|along) (?:parking |parked )?row$')
KERB = re.compile(r'^(?:parked )?(?:along|at|on|near|by) (?:curb|kerb)(?:side)?$')


def _triangle(contour):
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 16:
        return False
    polygon = cv2.approxPolyDP(contour, .065*perimeter, True)
    return (len(polygon) == 3 and cv2.isContourConvex(polygon)
            and .68 <= cv2.contourArea(contour)/max(1,cv2.contourArea(polygon)) <= 1.45)


def _triangular_panel(original, contour, map_size):
    """A triangular sign and its rectangular information plate often share one semantic component."""
    if original is None:
        return False
    x,y,w,h = cv2.boundingRect(contour)
    with Image.open(BytesIO(original)) as photo:
        sx,sy = photo.width/map_size[0],photo.height/map_size[1]
        crop = photo.crop((int(x*sx),int(y*sy),int(np.ceil((x+w)*sx)),int(np.ceil((y+h)*sy)))).convert('RGB')
        crop.thumbnail((200,240),Image.Resampling.LANCZOS)
        rgb = np.asarray(crop,dtype=np.float32)
    r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
    red = ((r>53) & (r>1.23*g) & (r>1.17*b)).astype('uint8')
    semantic = np.zeros((h,w),np.uint8)
    cv2.drawContours(semantic,[contour-np.int32([x,y])],-1,1,-1)
    red &= cv2.resize(semantic,(red.shape[1],red.shape[0]),interpolation=cv2.INTER_NEAREST)
    ink = cv2.morphologyEx(red,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    for part in cv2.findContours(ink,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        _,_,pw,ph = cv2.boundingRect(part)
        if (.42 <= pw/red.shape[1] and .65 <= pw/ph <= 1.82 and _triangle(part)
                and cv2.contourArea(part) >= .18*pw*ph):
            return True
    return False


def _near_landmark(classes, name, box, frame, side, original):
    """A sign across the scene or a bin on the other side cannot identify this car."""
    mask = np.isin(classes, LANDMARKS[name]).astype('uint8')
    h,w = mask.shape
    scale = np.array([w/frame['width'], h/frame['height']])
    l,t,r,b = np.asarray(box) * np.tile(scale,2)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,cw,ch = cv2.boundingRect(contour)
        if cw < 3 or ch < 3 or cv2.contourArea(contour) < 10:
            continue
        mx = x + cw/2
        if side and ((mx < w/2) != (side == 'left')):
            continue
        gap = max(0, l-x-cw, x-r)
        # Objects mounted above the car can be close in the picture. Ground-level
        # landmarks must reach the same depth range as the car itself.
        elevated = 'sign' in name or 'light' in name
        if gap <= .90*(r-l) and t-(1.25 if elevated else .22)*(b-t) <= y+ch <= b+.38*(b-t):
            if not name.startswith('triangular') or _triangle(contour) or _triangular_panel(original,contour,(w,h)):
                return True
    return False


def matching(request, ref, specs, readings, classes, *, original=None):
    """Return a match only when each extra detail can be checked. Ground constraints follow separately."""
    if (request['scene'].get('provenance',{}).get('network_policy') != 'local_only'
            or ref.get('ground_side') or ref.get('unsupported') or not ref.get('qualifiers')):
        return []
    frame = request['scene']['frame']
    sides = {s.lower() for s in SIDE.findall(request['text'])}
    if len(sides) > 1:
        return []
    named_side = next(iter(sides), None)
    results = []
    for spec,reading in zip(specs,readings,strict=True):
        colour,body = reading[:2]
        if (body == 'unknown' or not single_vehicle._compatible(ref['kind'],body)
                or ref.get('color') and ref['color'] != colour):
            continue
        ground = set()
        matched_landmark = False
        side_ambiguous = False
        for q in ref['qualifiers']:
            field,value = q['field'],q['value']
            if field == 'unknown' and value == 'which side the extra detail refers to':
                side_ambiguous = True
                continue
            if field == 'appearance' and q.get('source') == 'body_identity':
                words = value.split()
                if 'dark' in words and colour == 'black':
                    words.remove('dark')
                if not words:
                    continue
                detail = ' '.join(words)
                if ROW.fullmatch(detail):
                    ground.add('row')
                    continue
                if KERB.fullmatch(detail):
                    ground.add('kerb')
                    continue
            if (field == 'place' and q.get('relation') in {'near','by','beside','alongside','next to'}
                    and value in LANDMARKS and _near_landmark(classes,value,spec['box'],frame,named_side,original)):
                matched_landmark = True
                continue
            break
        else:
            if side_ambiguous and (not named_side or not matched_landmark):
                continue
            if 'kerb' in ground and named_side:
                ground.add(named_side)
            elif named_side and not single_vehicle._side(spec['box'],frame['width'],named_side):
                continue
            results.append((spec,reading,ground))
    return results


def ground_confirms(evidence, required):
    if 'row' in required and evidence.get('aligned_neighbour_id') is None:
        return False
    if 'kerb' in required and evidence.get('curb_side') not in {'left','right'}:
        return False
    return all(side not in required or evidence.get('curb_side') == side for side in ('left','right'))
