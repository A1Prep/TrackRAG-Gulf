"""Native-image corroboration for two types of off-lane point near parked vehicles.

Regular block paving is compared with the adjoining asphalt across an observed
aisle stripe. A separate side-entry case requires learned parking, visible paint,
two measured kerbs and a clear lateral opening. Neither yields a whole-stall outline.
"""
from itertools import combinations

import cv2
import numpy as np
from PIL import Image

from .bay_clearance import visible_road_component
from .curb_recess import _vehicles


RECIPE = "native_surface_and_parking_row_contacts_v1"
_GROUND = (10, 13, 24)


def _vegetated(rgb):
    pixels=np.asarray(rgb,dtype="int16")
    r,g,b=pixels[...,0],pixels[...,1],pixels[...,2]
    return np.mean((g>r+10)&(g>b+8)&(g>.55*(r+b)))>.065


def _texture(photo, x, y, scale):
    """Check repeated non-collinear surface detail in original pixels."""
    px, py = np.rint([x / scale[0], y / scale[1]]).astype(int)
    rx = max(23, min(76, round(photo.width / 65)))
    ry = max(18, round(.75 * rx))
    if px < rx or py < ry or px+rx >= photo.width or py+ry >= photo.height:
        return (0., 0.)
    crop = photo.crop((px-rx, py-ry, px+rx, py+ry)).convert("RGB")
    if _vegetated(crop):
        return None, None
    a = np.asarray(crop.convert("L"), dtype="float32")
    high = a - cv2.GaussianBlur(a, (0, 0), 3.2)
    magnitude = float(np.sqrt(np.mean(high * high)))
    if magnitude < .7:
        return magnitude, 0.
    height, width = a.shape
    a = high * np.outer(np.hanning(height), np.hanning(width))
    ac = np.fft.fftshift(np.fft.irfft2(np.abs(np.fft.rfft2(a)) ** 2, s=a.shape)).astype("float32")
    ac /= max(float(ac[height//2, width//2]), 1.e-8)
    yy, xx = np.ogrid[-height//2:height//2, -width//2:width//2]
    peaks = (ac >= cv2.dilate(ac, np.ones((5, 5), np.uint8))-.000001)
    keep = peaks & (yy >= 0) & (xx*xx+yy*yy > 49) & (abs(xx) < .30*width) & (yy < .29*height)
    indices = sorted(zip(*np.where(keep)), key=lambda ij: -ac[ij])[:8]
    lattice = 0.
    for i, j in combinations(indices, 2):
        v = np.array([i[1]-width//2, i[0]-height//2])
        u = np.array([j[1]-width//2, j[0]-height//2])
        cross = abs(v[0]*u[1]-v[1]*u[0])
        if cross > .48 * np.linalg.norm(v) * np.linalg.norm(u):
            lattice = max(lattice, min(float(ac[i]), float(ac[j])))
    return magnitude, lattice


def _runs(a):
    if not len(a):
        return []
    cuts = np.r_[0, np.flatnonzero(a[:-1] != a[1:])+1, len(a)]
    return [(int(a[i]), int(i), int(j)) for i, j in zip(cuts, cuts[1:])]


def _stripe(x, y, side, labels, free, road):
    w = labels.shape[1]
    limit = min(x if side < 0 else w-1-x, round(.21*w))
    xs = x + side*np.arange(limit)
    if not len(xs):
        return None
    runs = _runs(labels[y, xs])
    travelled = []
    for index, (label, i, j) in enumerate(runs):
        if label not in _GROUND or not free[y, xs[i:j]].all():
            return None
        if label != 24:
            travelled.append(label)
            continue
        if i < .015*w or j-i < 2 or j-i > .025*w or not travelled or index+1 >= len(runs):
            return None
        after, start, end = runs[index+1]
        minimum = max(9, round(.033*w))
        if after != 13 or end-start < minimum or not road[y, xs[start:start+minimum]].all():
            return None
        if not free[y, xs[start:start+minimum]].all():
            return None
        # Put the comparison crop well onto the roadway. A sample centred too
        # near the bright dividing line measures the line as if it were asphalt.
        return int(xs[(i+j)//2]), int(xs[min(end-1, start+round(1.45*minimum))])
    return None


def _kerb(x, y, labels):
    steps = min(y, round(.14 * labels.shape[0]))
    column = labels[y-np.arange(steps), x]
    boundary = np.flatnonzero(~np.isin(column, _GROUND))
    if not len(boundary):
        return None
    first = int(boundary[0])
    near = column[first:first+3]
    if (first < 3 or near[0] not in (2,15) or np.count_nonzero(np.isin(near,(2,15))) < 2
            or not np.isin(near[:2],(2,15,24)).all()):
        return None
    return y-first


def _has_frontage(near, distant, candidate, labels):
    # The tyre contacts must run beside the actual kerb, on its road side. A
    # central lane divider with two queued vehicles has no such nearby boundary.
    for car in (near, distant):
        l, _, r, b = car.box
        xx = int(np.clip((r if (l+r)/2 < candidate[0] else l), 0, labels.shape[1]-1))
        span = max(4, round(.07*car.width))
        xx = int(np.clip(xx + (span if xx < candidate[0] else -span), 0, labels.shape[1]-1))
        ys = np.arange(max(0,int(b-.43*car.height)), min(labels.shape[0],int(b+.16*car.height)+1))
        if not len(ys) or not np.isin(labels[ys, xx], (2,15)).any():
            return False
    return True


def _entrance(x, y, outside, needed, labels, free, road):
    """Require a contiguous vehicle-relative opening, using a common approach column.

    Three isolated rays can pass through a narrow gap between bollards or planters.
    Count every adjoining source-map row from the actual exterior lane to the
    proposed patch. Foreground occlusion or a material barrier breaks the passage.
    """
    h = labels.shape[0]
    a,b=sorted((x, outside))
    lo,hi=max(0,y-needed),min(h,y+needed+1)
    ground=np.isin(labels[lo:hi,a:b+1],_GROUND)
    possible=(ground & free[lo:hi,a:b+1]).all(axis=1) & road[lo:hi,outside]
    possible &= labels[lo:hi,outside] == 13
    for state,start,end in _runs(possible):
        if state and lo+start<=y<lo+end and end-start>=needed:
            return [lo+start,lo+end-1]
    return None


def _output(x, y, scale, radius, ids, kind, details, exact):
    def native(a, b):
        return [round((a+.5)/scale[0]-.5, 3), round((b+.5)/scale[1]-.5, 3)]
    point = list(exact) if exact is not None else native(x, y)
    patch = [native(x-radius,y-radius),native(x+radius,y-radius),
             native(x+radius,y+radius),native(x-radius,y+radius)]
    return {"point": point, "box": patch[0]+patch[2], "support_patch": patch,
            "evidence": {"recipe":RECIPE, "extent":"interior_point_only", "surface_type":kind,
                         "flanking_track_ids":list(ids), **details}}


def _paved_pair(im, labels, cars, free, road, scale, exact):
    h,w = labels.shape
    found = []
    for left, right in combinations(cars, 2):
        lo, hi = left.box[2], right.box[0]
        span = hi-lo
        if (span < max(.028*w,.30*min(left.width,right.width)) or span > .22*w
                or abs(left.box[3]-right.box[3]) <= .11*min(left.height,right.height)
                or max(left.height,right.height) > 3.8*min(left.height,right.height)):
            continue
        lc, rc = (left.box[0]+left.box[2])/2, (right.box[0]+right.box[2])/2
        side = -1 if right.box[3] > left.box[3] else 1
        if exact is None:
            xs = [int(lo+span*q) for q in (.52,.40,.64,.30,.73)]
        else:
            xs = [int(exact[0]*scale[0])]
        for x in xs:
            if not lo+.07*span < x < hi-.07*span:
                continue
            contact = left.box[3] + (x-lc)*(right.box[3]-left.box[3])/(rc-lc)
            y = int(round(contact+.055*min(left.height,right.height))) if exact is None else int(exact[1]*scale[1])
            if (not 4 <= y < h-4 or abs(y-contact) > .27*min(left.height,right.height)
                    or labels[y,x] not in (10,13) or not free[y,x]):
                continue
            core = max(2,min(6,round(.055*min(left.height,right.height))))
            section=np.s_[y-core:y+core+1,x-core:x+core+1]
            if not free[section].all() or not np.isin(labels[section],(10,13)).all():
                continue
            kerbs = [_kerb(x+d, y, labels) for d in (-core,0,core)]
            if None in kerbs or not _has_frontage(left,right,(x,y),labels):
                continue
            marks = [_stripe(x,y+d,side,labels,free,road) for d in (-core,0,core)]
            if None in marks:
                continue
            # Two different surfaces must be visible across the same observed aisle edge.
            rx=max(3,round(max(23,min(76,round(im.width/65)))*scale[0]))
            ry=max(3,round(.75*rx))
            outside=marks[1][1]
            required=max(12,int(np.ceil(.27*min(left.height,right.height))))
            entrance=_entrance(x,y,outside,required,labels,free,road)
            if entrance is None:continue
            lens=np.s_[max(0,y-ry):y+ry+1,max(0,outside-rx):outside+rx+1]
            if np.mean((labels[lens]==13)&free[lens]) < .98:
                continue
            mag, repeat = _texture(im,x,y,scale)
            out_mag, out_repeat = _texture(im, outside, y, scale)
            if (mag is None or out_mag is None or mag < 2.0 or repeat < .15
                    or mag < 1.27*out_mag or repeat < out_repeat+.015):
                continue
            ids = [left.track,right.track]
            found.append(_output(x,y,scale,core,ids,"kerbside_block_paving",
                                 {"signed_center_class":int(labels[y,x]), "joint_repeat_strength":round(repeat,4),
                                  "neighbouring_road_repeat_strength":round(out_repeat,4),
                                  "native_texture_rms":[round(mag,3),round(out_mag,3)],
                                  "observed_entrance_rows":entrance,"required_entrance_rows":required,
                                  "aisle_mark_grid":[[a[0],y+d] for a,d in zip(marks,(-core,0,core))],
                                  "kerb_grid":[[x+d,v] for d,v in zip((-core,0,core),kerbs)]},exact))
            break
    return found


def _parking_slice(x, y, sign, labels):
    h,w=labels.shape
    xs=x+sign*np.arange(min(x if sign<0 else w-x-1,round(.30*w)))
    for tag,a,_ in _runs(labels[y,xs]):
        if tag not in (10,24):
            return (int(xs[a]),tag) if a>2 else None
    return None


def _side_entry(im, labels, cars, free, road, scale, exact):
    h,w=labels.shape
    parking = labels == 10
    if np.count_nonzero(parking) < 80:
        return []
    near_park=cv2.dilate(parking.astype("uint8"),np.ones((5,5),np.uint8)).astype(bool)
    allowed = ((parking | ((labels == 24) & near_park)) & free).astype("uint8")
    n,components,stats,_=cv2.connectedComponentsWithStats(allowed,connectivity=8)
    result=[]
    for component in range(1,n):
        lx,ty,cw,ch,area=map(int,stats[component])
        if (cw<.07*w or ch<.027*h or ch>.20*h or cw<2.1*ch or area<.36*cw*ch):
            continue
        if exact is not None and not lx<=exact[0]*scale[0]<lx+cw:
            continue
        mask=components[ty:ty+ch,lx:lx+cw]==component
        paint=(labels[ty:ty+ch,lx:lx+cw]==24)&mask
        if paint.sum()<max(12,.008*area):continue
        yy,xx=np.nonzero(paint); gx,gy=xx+lx,yy+ty
        if gx.max()-gx.min()<.42*cw:continue
        # The learned markings must also be visibly lighter than the immediately
        # adjoining parking surface in the authenticated original image.
        crop=im.crop((max(0,int(lx/scale[0])),max(0,int(ty/scale[1])),
                      min(im.width,int(np.ceil((lx+cw)/scale[0]))),min(im.height,int(np.ceil((ty+ch)/scale[1])))))
        if min(crop.size)<1 or _vegetated(crop.convert("RGB")):continue
        gray=np.asarray(crop.resize((cw,ch),Image.Resampling.BOX).convert("L"),dtype="float32")
        bg=cv2.dilate(paint.astype("uint8"),np.ones((7,7),np.uint8)).astype(bool)&~paint&mask
        if not bg.any() or np.percentile(gray[paint],75)<np.median(gray[bg])+3.:continue
        xs=[int(lx+cw*f) for f in (.55,.64,.45,.73,.36)]
        if exact is not None:xs=[int(exact[0]*scale[0])]
        offered=None
        for x in xs:
            if not lx<=x<lx+cw:continue
            column=np.flatnonzero(mask[:,x-lx])+ty
            if len(column)<5:continue
            y=int(round((column[0]+column[-1])/2)) if exact is None else int(exact[1]*scale[1])
            if not ty<=y<ty+ch or labels[y,x]!=10 or components[y,x]!=component:continue
            above=_kerb(x,y,labels)
            lower=next((q for q in range(y+1,min(h,y+int(.10*h))) if labels[q,x] not in (10,24)),None)
            if above is None or lower is None or labels[lower,x] != 2 or lower-y<4:continue
            back=[c for c in cars if c.box[0] < x < c.box[2] and 0 < y-c.box[3] < .60*c.height
                  and cw>.65*c.width and lower-above>.31*c.height]
            if not back:continue
            for direction in (-1,1):
                probes=[(y+d,_parking_slice(x,y+d,direction,labels),
                         _parking_slice(x,y+d,-direction,labels)) for d in (-3,0,3)]
                if any(out is None or end is None or out[1]!=13 or end[1] not in (54,55,56,57,60,61,62,63)
                       for _,out,end in probes):continue
                if not all(road[row,out[0]] and free[row,min(x,out[0]):max(x,out[0])+1].all()
                           for row,out,_ in probes):continue
                open_length=max(8,round(.023*w))
                access=[(row,out[0]+direction*np.arange(open_length)) for row,out,_ in probes]
                if any(aa.min()<0 or aa.max()>=w or not (labels[row,aa]==13).all()
                       or not road[row,aa].all() or not free[row,aa].all() for row,aa in access):continue
                endx=probes[1][2][0]
                side=[c for c in cars if c not in back and c.box[1]<y<c.box[3]
                      and c.box[0]-.025*w<=endx<=c.box[2]+.025*w]
                if not side:continue
                radius=max(1,min(4,(lower-above)//8))
                if not free[y-radius:y+radius+1,x-radius:x+radius+1].all():continue
                flange=min(back,key=lambda c:abs(y-c.box[3]))
                stopper=min(side,key=lambda c: min(abs(endx-c.box[0]),abs(endx-c.box[2])))
                required=max(12,int(np.ceil(.62*(lower-above))),
                             int(np.ceil(.24*min(flange.height,stopper.height))))
                entrance=_entrance(x,y,int(access[1][1][-1]),required,labels,free,road)
                if entrance is None:continue
                offered=_output(x,y,scale,radius,[flange.track,stopper.track],"painted_side_entry",
                                {"signed_center_class":10,"parking_component_pixels":area,
                                 "optically_corroborated_paint_cells":int(paint.sum()),
                                 "observed_entrance_rows":entrance,"required_entrance_rows":required,
                                 "lateral_road_contacts_grid":[[v[0],q] for q,v,_ in probes],
                                 "near_and_far_kerb_grid":[[x,above],[x,lower]]},exact)
                break
            if offered:break
        if offered:result.append(offered)
    return result


def _run(original, classes, context, exact=None):
    if (not isinstance(classes,np.ndarray) or classes.ndim!=2 or min(classes.shape)<48
            or not isinstance(context,dict) or tuple(context.get("size",()))!=original.size
            or "current_vehicles" not in context or "current_boxes" not in context):
        return []
    h,w=classes.shape;size=np.asarray(original.size);scale=np.array([w,h])/size
    try:
        cars,boxes=_vehicles(context,scale,size,h,w)
    except (ValueError,TypeError):
        return []
    if len(cars)<2 or not np.any(classes==2):return []
    road=visible_road_component(classes)
    if road is None:return []
    free=np.ones((h,w),bool)
    for l,t,r,b in boxes:free[int(t):int(np.ceil(b))+1,int(l):int(np.ceil(r))+1]=False
    result=_paved_pair(original,classes,cars,free,road,scale,exact)
    result.extend(_side_entry(original,classes,cars,free,road,scale,exact))
    return sorted(result,key=lambda o:(o["point"],o["evidence"]["flanking_track_ids"]))


def candidates(original, classes, context):
    return _run(original,classes,context)


def marked(original, classes, context, mark):
    """Return evidence for this exact small mark only, without snapping to a neighbouring point."""
    if not isinstance(mark,(tuple,list)) or len(mark)!=4:return None
    try:
        l,t,r,b=map(float,mark)
    except (ValueError,TypeError):
        return None
    w,h=original.size
    if not (0<=l<r<=w and 0<=t<b<=h and r-l<=.012*w and b-t<=.012*h):return None
    point=[(l+r)/2,(t+b)/2]
    for option in _run(original,classes,context,point):
        x0,y0,x1,y1=option["box"]
        if x0<=l<=r<=x1 and y0<=t<=b<=y1:return option
    return None
