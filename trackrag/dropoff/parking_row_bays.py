"""Measure parking spaces bounded by repeated road studs visible in the source photograph."""
from itertools import combinations

import cv2
import numpy as np
from PIL import Image

from . import painted_bays


RECIPE = 'repeated_optical_parking_dividers_v1'
GROUND = (10, 13, 24, 36)


def _work_view(original, classes):
    width = min(1440, original.width)
    size = (width, round(original.height * width / original.width))
    rgb = np.asarray(original.convert('RGB').resize(size, Image.Resampling.LANCZOS))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    labels = cv2.resize(classes, size, interpolation=cv2.INTER_NEAREST)
    return gray, labels, np.array(original.size) / size


def _stud_points(gray, labels):
    h, w = gray.shape
    parking = labels == 10
    if parking.sum() < .001 * gray.size:
        return np.empty((0, 4))
    distance = cv2.distanceTransform((~parking).astype('uint8'), cv2.DIST_L2, 3)
    allowed = np.isin(labels, GROUND) & (distance < max(5, w * .022))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(11, (w//60)|1), max(7, (w//170)|1)))
    contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    bright = (contrast >= np.maximum(20., .20 * gray)) & allowed
    count, components, stats, centers = cv2.connectedComponentsWithStats(bright.astype('uint8'))
    points = []
    for i in range(1, count):
        x, y, width, height, area = stats[i]
        if not (max(2, .0015*w) <= width <= .027*w and 1 <= height <= .012*w
                and 1.12 <= width/height <= 9 and .40*width*height <= area <= .00017*w*w):
            continue
        local = contrast[y:y+height, x:x+width][components[y:y+height, x:x+width] == i]
        score = float(np.quantile(local, .75))
        if score >= 9:
            points.append([*centers[i], float(width), score])
    # Extremely textured frames should abstain instead of fitting chance alignments to noise.
    if len(points) > 560:
        points = sorted(points, key=lambda item:item[3], reverse=True)[:560]
    return np.asarray(sorted(points), dtype=np.float64).reshape(-1, 4)


def _stud_lines(points, labels):
    """Fit only chains of separately visible, consistently spaced optical highlights."""
    if len(points) < 8:
        return []
    h,w = labels.shape
    xy = points[:,:2]
    found = {}
    for i, origin in enumerate(xy):
        offsets = xy - origin
        for j in np.flatnonzero((offsets[:,0] >= .010*w) & (offsets[:,0] <= .081*w)
                                & (np.abs(offsets[:,1]) <= .72*offsets[:,0])
                                & (points[:,2] >= .38*points[i,2]) & (points[:,2] <= 2.65*points[i,2])):
            step = np.linalg.norm(offsets[j])
            if (min(points[i,2],points[j,2]) < .055*step or max(points[i,2],points[j,2]) > .59*step):
                continue
            axis = offsets[j]/step
            normal = np.array([-axis[1],axis[0]])
            along = offsets@axis
            perpendicular = np.abs(offsets@normal)
            tolerance = min(max(1.6, .040*step), 3.8)
            aligned = np.flatnonzero(
                (perpendicular<=tolerance) & (points[:,2]>=max(2.,.045*step)) & (points[:,2]<=.61*step))
            slots = {}
            for k in aligned:
                n = round(along[k]/step)
                err = abs(along[k]/step - n)
                if err < .19 and abs(n) <= 16 and (n not in slots or err < slots[n][0]):
                    slots[n]=(err,k)
            if 0 not in slots or 1 not in slots:
                continue
            lo,hi=0,1
            while lo-1 in slots: lo-=1
            while hi+1 in slots: hi+=1
            chosen=tuple(slots[n][1] for n in range(lo,hi+1))
            if len(chosen)<4 or chosen in found:
                continue
            support=xy[list(chosen)]; span=np.linalg.norm(support[-1]-support[0]);
            if span<.065*w or np.std(points[list(chosen),2])>.48*np.median(points[list(chosen),2]):
                continue
            center=support.mean(axis=0); _,_,axes=np.linalg.svd(support-center,full_matrices=False); direction=axes[0];
            if direction[0]<0:direction=-direction
            pos=(support-center)@direction
            ends=np.array([center+direction*pos.min(),center+direction*pos.max()]);
            found[chosen]={'ends':ends,'axis':direction,'count':len(chosen),'period':step,'members':set(chosen),
                           'width':float(np.median(points[list(chosen),2])), 'contrast':float(np.median(points[list(chosen),3]))}
    result=[]
    for _,line in sorted(found.items(),key=lambda item:(-item[1]['count'],-np.linalg.norm(np.diff(item[1]['ends'],axis=0)),item[0])):
        if any(len(line['members'] & previous['members']) >= 2 for previous in result):
            continue
        result.append(line)
    return result


def _stripe_support(gray, a, b, *, flank=4.):
    delta=b-a
    length=np.linalg.norm(delta)
    if length<4:
        return np.zeros(1,dtype=bool)
    normal=np.array([-delta[1],delta[0]])/length
    path=np.linspace(a,b,max(7, min(150,round(length))))
    samp=painted_bays._samples
    peak=np.max([samp(gray,path+n*normal) for n in (-2.,-1.,0.,1.,2.)],axis=0)
    sides=np.maximum(samp(gray,path-flank*normal),samp(gray,path+flank*normal))
    return peak-sides>=4


def _capped_stems(gray, labels, boxes):
    """Follow observed paint from a genuine transverse end cap to a curb on the same ground."""
    h,w=gray.shape
    curb=np.isin(labels,(2,15)); ground=np.isin(labels,GROUND)
    if not boxes.size or not curb.any():
        return []
    if not any((r-l)>.065*w and (b-t)>.045*h for l,t,r,b in boxes):
        return []
    boosted=cv2.createCLAHE(clipLimit=2.5,tileGridSize=(12,12)).apply(gray)
    lines=cv2.createLineSegmentDetector().detect(boosted)[0]
    if lines is None:
        return []
    ends=lines.reshape(-1,2,2).astype('float64')
    vectors=ends[:,1]-ends[:,0]
    spans=np.linalg.norm(vectors,axis=1)
    axes=vectors/np.maximum(1.,spans)[:,None]
    mid=ends.mean(axis=1)
    ground_mid=painted_bays._samples(ground,mid)
    cap_ids=np.flatnonzero(ground_mid&(spans>=max(7.,.0060*w))&(spans<=.070*w))
    long_ids=np.flatnonzero(ground_mid&(spans>=max(20.,.024*w))&(spans<=.60*w))
    curb_distance=cv2.distanceTransform((~curb).astype('uint8'),cv2.DIST_L2,3)
    float_gray=boosted.astype('float32')
    short=[]
    for i in cap_ids:
        cap=ends[i];
        if (painted_bays._samples(ground,np.linspace(*cap,7)).mean()>=.85
                and _stripe_support(float_gray,*cap,flank=max(3.5,.003*w)).mean()>=.53):
            short.append(i)
    if len(short)<3:
        return []
    caps=np.asarray(short)
    found=[]
    for i in long_ids:
        a,b=ends[i]; d=axes[i].copy(); c=mid[i]
        if painted_bays._samples(ground,np.linspace(a,b,11)).mean()<.91:
            continue
        if _stripe_support(float_gray,a,b).mean()<.60:
            continue
        # The stripe must approach a signed curb; an ordinary lane edge follows it.
        near=painted_bays._samples(curb_distance,np.array([a,b]))
        if abs(near[0]-near[1]) < .10*spans[i]:
            continue
        if near[0]<near[1]:d=-d
        n=np.array([-d[1],d[0]])
        reach=min(.56*w, max(.20*w,3.4*spans[i]))
        distances=np.arange(0,reach,2.)
        path=c+distances[:,None]*d
        touch=np.flatnonzero(painted_bays._samples(curb,path))
        if not len(touch) or touch[0]<4:
            continue
        k=touch[0];back=path[max(k-1,0)]
        if (painted_bays._samples(ground,path[:k]).mean()<.97
                or not (2<back[0]<w-2 and 2<back[1]<h-2)):
            continue
        delta=mid[caps]-c
        along=delta@d; perp=np.abs(delta@n);angle=np.abs(axes[caps]@d)
        possible=np.flatnonzero((along < .32*spans[i])&(along>-.38*w)&(angle<.978)&(angle>.12)
                                &(perp < .64*spans[caps]*np.sqrt(np.maximum(0.,1-angle**2))+3.))
        for j in possible:
            ci=caps[j];cap_axis=axes[ci];denom=cap_axis@n
            crossing=mid[ci] - cap_axis*((mid[ci]-c)@n)/denom
            if np.linalg.norm(crossing-mid[ci])>.66*spans[ci]+2:
                continue
            if not .080*w<np.linalg.norm(back-crossing)<.56*w:
                continue
            span=np.linalg.norm(back-crossing); ray=np.linspace(crossing,back,27)
            if painted_bays._samples(ground,ray[1:-1]).mean()<.96:
                continue
            contrast=_stripe_support(float_gray,crossing,back,flank=max(4.,.0035*w))
            if contrast.mean()<.52 or min(np.mean(part) for part in np.array_split(contrast,3))<.25:
                continue
            # At the mouth the stem terminates, while the short crossing stroke continues.
            before=_stripe_support(float_gray,crossing-d*min(.10*span,21.),crossing-d*4,flank=4.)
            if before.mean()>.64:
                continue
            found.append({'front':crossing,'back':back,'axis':d,'cap_axis':cap_axis,'cap':ends[ci],
                          'strength':float(contrast.mean()),'length':span})
    # Two edges or several fragments of one physical stripe provide only one observation.
    kept=[]
    for arm in sorted(found,key=lambda v:(-v['strength'],-v['length'])):
        if any(np.linalg.norm(arm['front']-old['front'])<max(10.,.020*w) and
               np.linalg.norm(arm['back']-old['back'])<.055*w for old in kept):
            continue
        kept.append(arm)
    return kept


def _overlap(polygon, box):
    left, top, right, bottom = box
    corners = np.float32([[left,top],[right,top],[right,bottom],[left,bottom]])
    return cv2.intersectConvexConvex(polygon, corners)[0]


def _semantic_stems(gray, labels, boxes):
    """Measure continuous painted dividers between the aisle and a matching back boundary."""
    h,w=labels.shape
    parking=labels==10
    if parking.sum()<.0025*w*h:
        return []
    nearby=cv2.distanceTransform((~parking).astype('uint8'),cv2.DIST_L2,3)
    supported=np.isin(labels,(10,24)) & (nearby<.015*w)
    if supported.sum()<.003*w*h:
        return []
    stepped=cv2.createCLAHE(clipLimit=2.4,tileGridSize=(12,12)).apply(gray)
    original=stepped.astype('float32')
    detected=cv2.createLineSegmentDetector().detect(stepped)[0]
    if detected is None:
        return []
    curb_distance=cv2.distanceTransform((~np.isin(labels,(2,15))).astype('uint8'),cv2.DIST_L2,3)
    ground_mask=np.isin(labels,GROUND)
    sam=painted_bays._samples
    measured=[]
    for x0,y0,x1,y1 in detected.reshape(-1,4):
        a,b=np.array([x0,y0]),np.array([x1,y1]); line=b-a; length=np.linalg.norm(line)
        if not max(15.,.026*w)<length<.60*w or sam(supported,np.linspace(a,b,9)).mean()<.62:
            continue
        axis=line/length;normal=np.array([-axis[1],axis[0]])
        center=(a+b)/2
        offsets=np.arange(-round(.50*w),round(.50*w),2)
        ray=center+offsets[:,None]*axis
        original_mask=sam(supported,ray)
        ground=sam(ground_mask,ray)
        in_frame=(ray[:,0]>2)&(ray[:,0]<w-3)&(ray[:,1]>2)&(ray[:,1]<h-3)
        mask=cv2.morphologyEx(original_mask.astype('uint8').reshape(1,-1),cv2.MORPH_CLOSE,
                              np.ones((1,5),np.uint8))[0].astype(bool)&ground&in_frame
        groups=np.split(np.flatnonzero(mask),np.flatnonzero(np.diff(np.flatnonzero(mask))>1)+1)
        middle=len(offsets)//2
        run=next((s for s in groups if len(s)>=3 and s[0]-2<=middle<=s[-1]+2),None)
        if run is None or (offsets[run[-1]]-offsets[run[0]])<.065*w:
            continue
        end=np.array([ray[run[0]],ray[run[-1]]]); span=np.linalg.norm(end[1]-end[0]);
        if span>max(5*length,.30*w) or span>.65*w:
            continue
        beyond=np.array([end[0]-axis*np.arange(4,25,3)[:,None],end[1]+axis*np.arange(4,25,3)[:,None]])
        outer_road=[np.mean(sam(labels,line)==13) for line in beyond]
        front_i=int(np.argmax(outer_road)); back_i=1-front_i
        hood=False
        if outer_road[front_i]<.70:
            # On a near parking bay, the camera bonnet can hide the aisle terminal of one
            # stripe. Its partner must still establish the aisle, and the extent is partial.
            ego=[np.mean(sam(labels,line)==64) for line in beyond]
            front_i=int(np.argmax(ego));back_i=1-front_i;hood=True
            if ego[front_i]<.85:
                continue
        if outer_road[back_i]>.65:
            continue
        rear=end[back_i];curb=sam(curb_distance,rear[None])[0]<max(6.,.014*w)
        obstacle=any(l-5<=rear[0]<=r+5 and t-5<=rear[1]<=bottom+5 for l,t,r,bottom in boxes)
        if not curb and not obstacle:
            continue
        best=None
        shifts=[(s,s) for s in (-3.,0.,3.)]
        if hood:
            # Opposite sides of a wide painted stripe converge under perspective. Trace its
            # axis through the observed bright core with separate offsets at the two ends.
            bound=max(4.,.006*w)
            shifts=[(a,b) for a in (-bound,0.,bound) for b in (-bound,0.,bound)]
        for off0,off1 in shifts:
            segment=end+np.float32([[off0],[off1]])*normal
            for flank in ((6.,10.,15.) if hood else (4.5,8.)):
                values=_stripe_support(original,*segment,flank=flank)
                support=float(values.mean());endsupport=min(np.mean(v) for v in np.array_split(values,4))
                if (support>=.65 and endsupport>=.36 and
                        (best is None or support>best[0])):
                    best=(support,segment)
        if best is None:
            continue
        support,segment=best; front,back=segment[front_i],segment[back_i]
        proposed={'front':front,'back':back,'axis':(back-front)/span,'length':span,
                  'strength':support,'back_curb':bool(curb),'back_vehicle':bool(obstacle),'front_hood':hood}
        measured.append(proposed)
    stems=[]
    for stem in sorted(measured,key=lambda s:-s['strength']):
        if not any(np.linalg.norm(stem['front']-t['front'])<.018*w and
                   np.linalg.norm(stem['back']-t['back'])<.028*w for t in stems):
            stems.append(stem)
    return stems


def _clear_part(polygon, boxes):
    """Clip a peripheral occlusion, never carve a nominal bay around an occupied centre."""
    area = cv2.contourArea(polygon)
    original = polygon.copy()
    for box in boxes:
        covered = _overlap(polygon,box)
        if covered < .05:
            continue
        left, top, right, bottom = box
        if covered > .31*area or cv2.pointPolygonTest(original, ((left+right)/2, bottom), False) > 0:
            return None
        portions = [painted_bays._clip_edge(polygon, axis, val, less) for axis,val,less in
                    [(0,left-.5,True),(0,right+.5,False),(1,top-.5,True),(1,bottom+.5,False)]]
        polygon = max(portions, key=lambda p:cv2.contourArea(p) if len(p)>=3 else 0)
        if len(polygon)<3 or cv2.contourArea(polygon) < .67*area:
            return None
    return polygon


def _interior(poly, labels, *, require_parking=True):
    x,y,w,h=cv2.boundingRect(poly)
    if x<0 or y<0 or x+w>labels.shape[1] or y+h>labels.shape[0]:
        return None
    mask=np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(mask,np.rint(poly-[x,y]).astype('int32'),1)
    inset=cv2.erode(mask,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    local=labels[y:y+h,x:x+w]
    cells=local[inset>0]
    if len(cells)<65 or np.isin(cells,(8,23)).any() or np.isin(cells,GROUND).mean()<.98:
        return None
    if require_parking and (cells==10).mean()<.57:
        return None
    blocked=(inset>0)&~np.isin(local,GROUND)
    if np.any(blocked):
        _,_,stats,_=cv2.connectedComponentsWithStats(blocked.astype('uint8'))
        if max(stats[1:,cv2.CC_STAT_AREA])>=max(8,.01*len(cells)):
            return None
    return cells


def _parking_center(polygon,labels):
    moment=cv2.moments(polygon)
    center=np.array([moment['m10'],moment['m01']])/moment['m00']
    x,y,w,h=cv2.boundingRect(polygon)
    if labels[round(center[1]),round(center[0])]==10:
        return center
    mask=np.zeros((h,w),np.uint8)
    cv2.fillConvexPoly(mask,np.rint(polygon-[x,y]).astype('int32'),1)
    mask[labels[y:y+h,x:x+w]!=10]=0
    inside=cv2.distanceTransform(np.pad(mask,1),cv2.DIST_L2,5)[1:-1,1:-1]
    pts=np.column_stack(np.where(inside>=max(2.,.55*inside.max()))[::-1])+[x,y]
    if len(pts)==0:
        return None
    nearest=pts[np.argmin(np.linalg.norm(pts-center,axis=1))]
    return nearest if np.linalg.norm(nearest-center)<.26*max(w,h) else None


def _semantic_options(stems, labels, boxes, scale):
    h,w=labels.shape
    output=[]
    for a,b in combinations(stems,2):
        one_hood=bool(a.get('front_hood') or b.get('front_hood'))
        # Strong convergence of the side stripes is valid near the camera. A single such
        # partly occluded bay needs an actual T-headed partner and two curb-facing ends.
        paired_terminals=(one_hood and bool(a.get('cap') is not None or b.get('cap') is not None)
                          and a['back_curb'] and b['back_curb'])
        if ((a.get('front_hood') and b.get('front_hood')) or a['axis']@b['axis']<(.70 if paired_terminals else .95)
                or not (a['back_curb'] or b['back_curb'])):
            continue
        fronts=np.array([a['front'],b['front']]);backs=np.array([a['back'],b['back']]);
        length=max(a['length'],b['length']); df=fronts[1]-fronts[0];db=backs[1]-backs[0]
        fw=np.linalg.norm(df);bw=np.linalg.norm(db)
        if (min(fw,bw)<max(12.,.013*w) or max(fw,bw)>(1.45 if paired_terminals else .86)*length or not .42<fw/bw<2.65
                or df@db/(fw*bw)<.79 or min(a['length'],b['length'])<.44*length):
            continue
        q=np.float32([fronts[0],backs[0],backs[1],fronts[1]])
        if not cv2.isContourConvex(q) or cv2.contourArea(q)<.0014*w*h:
            continue
        neighbours=0;interior_divider=False
        normal=np.array([-a['axis'][1],a['axis'][0]])
        depth=max(3.,.12*abs(df@normal))
        for t in stems:
            if t is a or t is b or t['axis']@a['axis']<.96:
                continue
            if np.linalg.norm(t['front']-fronts[0])<.012*w or np.linalg.norm(t['front']-fronts[1])<.012*w:
                continue
            sample=np.linspace(t['front'],t['back'],12)[2:-2]
            if sum(cv2.pointPolygonTest(q,tuple(p),True)>depth for p in sample)>=3:
                interior_divider=True
                break
            delta=t['front']-fronts[0]; k=delta@df/(fw*fw)
            if ((-2.8<k<-.32 or 1.32<k<3.8) and np.linalg.norm(delta-k*df)<max(9.,.31*fw)
                    and .43*length<t['length']<1.65*length):
                neighbours+=1
        if interior_divider or not (neighbours or paired_terminals):
            continue
        poly=_clear_part(q,boxes)
        if poly is None:
            continue
        cells=_interior(poly,labels,require_parking=False)
        if cells is None or (cells==10).mean()<(.75 if paired_terminals else .44) or np.isin(cells,(10,24)).mean()<.85:
            continue
        center=_parking_center(poly,labels)
        if center is None:
            continue
        ratio=cv2.contourArea(poly)/cv2.contourArea(q)
        coord=lambda p:np.asarray(p,dtype=float).round(3).tolist()
        vertices=poly*scale
        output.append({'polygon':coord(vertices),'point':coord(center*scale),
                       'box':coord(np.r_[vertices.min(axis=0),vertices.max(axis=0)]),
                       'evidence':{'recipe':RECIPE,'boundary_kind':'aisle_to_curb_parking_row',
                                   'coverage':'measured_stall' if not one_hood and ratio>.97 and a['back_curb'] and b['back_curb'] else 'visible_portion',
                                   'frontage_xy':coord(backs.mean(axis=0)*scale),
                                   'painted_boundaries':[],'boundary_widths':[],'boundary_opposite_closed':[],
                                   'optical_dividers':[coord(np.array([s['front'],s['back']])*scale) for s in (a,b)],
                                   'two_flank_paint_support':[round(s['strength'],3) for s in (a,b)],
                                   'rear_curb_seen':[a['back_curb'],b['back_curb']], 'external_dividers':neighbours,
                                   'aisle_occluded_by_camera':one_hood,
                                   'visible_parking_cells':int((cells==10).sum()),'interior_cells':len(cells),
                                   'remaining_fraction':round(ratio,3)}})
    return output


def _capped_options(stems, labels, boxes, scale):
    h,w=labels.shape
    result=[]
    for first,second in combinations(stems,2):
        f,g=first['front'],second['front']; a,b=first['back'],second['back']
        front=g-f;back=b-a;fw=np.linalg.norm(front);bw=np.linalg.norm(back)
        shorter=min(first['length'],second['length']);longer=max(first['length'],second['length'])
        minimum_terminal_gap = max(12., .016*w, .045*shorter)
        if (first['axis']@second['axis']<.95 or shorter<.60*longer or min(fw,bw)<minimum_terminal_gap
                or max(fw,bw)>.73*longer or not .48<fw/bw<2.20):
            continue
        row=front/fw; curb_row=back/bw
        if row@curb_row<.89 or any(abs(s['cap_axis']@row)<.91 for s in (first,second)):
            continue
        # The stems point into the curb and the end caps follow the aisle. Lane-edge paint
        # parallel to a median cannot have the required projected angle at both terminals.
        if max(abs(row@first['axis']),abs(row@second['axis']))>.982:
            continue
        external=[];middle=False
        for other in stems:
            if other is first or other is second or other['axis']@first['axis']<.94:
                continue
            front_delta=other['front']-f;back_delta=other['back']-a
            tx=front_delta@row/fw;ty=back_delta@curb_row/bw
            if (np.linalg.norm(front_delta-tx*front)>max(6.,.18*fw)
                    or np.linalg.norm(back_delta-ty*back)>max(7.,.19*bw)
                    or abs(tx-ty)>.65 or abs(other['cap_axis']@row)<.91):
                continue
            if .14<tx<.86 and .14<ty<.86:
                middle=True
            if (-2.6<tx<-.26 or 1.26<tx<3.6) and .45*shorter<other['length']<1.8*longer:
                external.append(other)
        if middle or not external:
            continue
        original=np.float32([f,a,b,g])
        if not cv2.isContourConvex(original) or cv2.contourArea(original)<.0013*w*h:
            continue
        # A vehicle's visible ground contact corroborates the row's scale and location.
        # Its actual occupied rectangle remains excluded from the entire suggested outline.
        near_car=False
        for l,t,r,bottom in boxes:
            contact=((l+r)/2,bottom)
            if (r-l)>.06*w and bottom-t>.045*h and cv2.pointPolygonTest(original,contact,True)>-.30*longer:
                near_car=True
                break
        if not near_car:
            continue
        poly=_clear_part(original,boxes)
        if poly is None:
            continue
        cells=_interior(poly,labels,require_parking=False)
        if cells is None or ((cells==10).mean()<.40 and np.mean(np.isin(cells,(10,13)))<.82):
            continue
        moment=cv2.moments(poly);center=np.array([moment['m10'],moment['m01']])/moment['m00']
        if labels[round(center[1]),round(center[0])] not in (10,13):
            continue
        vertices=poly*scale;ratio=cv2.contourArea(poly)/cv2.contourArea(original)
        coords=lambda arr:np.asarray(arr,dtype=float).round(3).tolist()
        # Save genuinely observed short cap strokes, not an unbroken painted line over every
        # gap in the much longer worn optical stems. Shared polygon edges still bind interiors.
        result.append({'polygon':coords(vertices),'point':coords(center*scale),
                       'box':coords(np.r_[vertices.min(axis=0),vertices.max(axis=0)]),
                       'evidence':{'recipe':RECIPE,'boundary_kind':'curb_facing_capped_row',
                                   'coverage':'measured_stall' if ratio>.97 else 'visible_portion',
                                   'frontage_xy':coords((a+b)/2*scale),
                                   'painted_boundaries':[coords(s['cap']*scale) for s in (first,second)],
                                   'boundary_widths':[round(2.3*scale.mean(),3)]*2,'boundary_opposite_closed':[False,False],
                                   'optical_dividers':[coords(np.array([s['front'],s['back']])*scale) for s in (first,second)],
                                   'two_flank_paint_support':[round(s['strength'],3) for s in (first,second)],
                                   'external_capped_dividers':len(external),'remaining_fraction':round(ratio,3),
                                   'visible_parking_cells':int((cells==10).sum()),'interior_cells':len(cells)}})
    return result


def _pair(first, second, lines, labels, boxes, scale):
    a,b=first['ends'],second['ends']
    u=first['axis']+second['axis']; u/=np.linalg.norm(u)
    if first['axis']@second['axis']<.985:
        return None
    normal=np.array([-u[1],u[0]])
    ac,bc=a.mean(axis=0),b.mean(axis=0)
    sep=abs((bc-ac)@normal)
    la=np.linalg.norm(a[1]-a[0]); lb=np.linalg.norm(b[1]-b[0])
    h,w=labels.shape
    if (sep<max(3., .005*w) or sep>.38*max(la,lb) or min(la,lb)<.40*max(la,lb)
            or abs((bc-ac)@u)>.74*max(la,lb) or max(first['period'],second['period'])>1.75*min(first['period'],second['period'])):
        return None
    if len(first['members'] & second['members']) or min(first['count'],second['count'])<4 or max(first['count'],second['count'])<6:
        return None
    # A third row must have the same optical direction and act as an external neighbour.
    # An intermediate divider invalidates this pair: it would encompass multiple spaces.
    neighbours=0
    for other in lines:
        if other is first or other is second or other['axis']@u<.985:
            continue
        oc=other['ends'].mean(axis=0)
        t=(oc-ac)@normal/((bc-ac)@normal)
        if abs((oc-(ac+bc)/2)@u)>.80*max(la,lb):
            continue
        if .19<t<.81:
            return None
        if (-2.7<t<-.30 or 1.30<t<3.7) and min(first['period'],second['period'])*.50<other['period']<max(first['period'],second['period'])*2.0:
            neighbours+=1
    if neighbours==0:
        return None
    # End the footprint at the visible end studs with at most a fraction of their measured spacing.
    da=first['axis']*first['period']*.24
    db=second['axis']*second['period']*.24
    ea=np.array([a[0]-da,a[1]+da]); eb=np.array([b[0]-db,b[1]+db]);
    original=np.float32([ea[0],ea[1],eb[1],eb[0]])
    if not cv2.isContourConvex(original) or cv2.contourArea(original)<.0008*w*h:
        return None
    poly=_clear_part(original,boxes)
    if poly is None:
        return None
    cells=_interior(poly,labels)
    if cells is None:
        return None
    moments=cv2.moments(poly)
    center=np.array([moments['m10'],moments['m01']])/moments['m00']
    if labels[round(center[1]), round(center[0])] != 10:
        return None
    vertices=poly*scale
    coords=lambda p:np.asarray(p,dtype=float).round(3).tolist()
    # A row of discrete studs is not a continuous painted stripe. Manual taps outside a
    # measured contour must never be snapped to a fictional line in the unmarked interval.
    return {'polygon':coords(vertices),'point':coords(center*scale),
            'box':coords(np.r_[vertices.min(axis=0),vertices.max(axis=0)]),
            'evidence':{'recipe':RECIPE,'boundary_kind':'repeated_pavement_studs','coverage':'visible_portion',
                        'frontage_xy':coords(center*scale),
                        'painted_boundaries':[],'boundary_widths':[],'boundary_opposite_closed':[],
                        'stud_boundaries':[coords(a*scale),coords(b*scale)],'stud_counts':[first['count'],second['count']],
                        'external_stud_rows':neighbours,'observed_period_pixels':coords([first['period']*scale.mean(),second['period']*scale.mean()]),
                        'remaining_fraction':round(cv2.contourArea(poly)/cv2.contourArea(original),3),
                        'visible_parking_cells':int((cells==10).sum()),'interior_cells':len(cells)}}


def candidates(original, classes, context):
    """Return source-pixel polygons of unoccupied parking compartments with optical divider evidence."""
    if not isinstance(original,Image.Image) or tuple(context['size']) != original.size:
        raise ValueError('Parking row photograph must match the current source dimensions')
    classes=np.asarray(classes)
    if classes.ndim!=2 or classes.dtype!=np.uint8 or min(classes.shape)<8:
        raise ValueError('Expected authenticated two-dimensional semantic classes')
    gray,labels,scale=_work_view(original,classes)
    boxes=np.asarray(context['current_boxes'],dtype=float).reshape(-1,4)/np.tile(scale,2)
    points=_stud_points(gray,labels)
    lines=_stud_lines(points,labels)
    options=[]
    for a,b in combinations(lines,2):
        option=_pair(a,b,lines,labels,boxes,scale)
        if option:
            options.append(option)
    stems=_capped_stems(gray,labels,boxes)
    options.extend(_capped_options(stems,labels,boxes,scale))
    if not any(o['evidence']['boundary_kind']=='repeated_pavement_studs' for o in options):
        supported=_semantic_stems(gray,labels,boxes)
        if any(s['front_hood'] for s in supported):
            # Only one strong, terminally capped measurement per physical line may supply
            # the far partner of an aisle edge occluded by the bonnet.
            caps=[]
            for stem in sorted(stems,key=lambda s:-s['strength']):
                if any(np.linalg.norm(other['back']-stem['back'])<.025*labels.shape[1]
                       and other['axis']@stem['axis']>.98 for other in caps):
                    continue
                caps.append(dict(stem,back_curb=True,back_vehicle=False,front_hood=False))
            supported.extend(caps)
        options.extend(_semantic_options(supported,labels,boxes,scale))
    distinct=[]
    for option in options:
        proposed=np.float32(option['polygon'])
        if any(cv2.intersectConvexConvex(previous,proposed)[0]>.18*min(cv2.contourArea(previous),cv2.contourArea(proposed))
               for previous,_ in distinct):
            continue
        distinct.append((proposed,option))
    return sorted((o for _,o in distinct),key=lambda o:tuple(o['point']))
