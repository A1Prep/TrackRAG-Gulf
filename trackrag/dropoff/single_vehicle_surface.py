"""Find an unoccupied parking surface touching the camera-facing end of a detected vehicle."""
import cv2
import numpy as np


def beside_patches(classes, box, occupied, size):
    """Find substantial, visible parking surfaces on either side at the vehicle's contact depth.

    A travel lane, an isolated pixel and the space projecting out from a bumper are
    excluded. The caller still needs an independent check of the requested vehicle and
    its orientation before a patch can be offered to a passenger.
    """
    height, width = classes.shape
    sx, sy = width / size[0], height / size[1]
    left, top, right, bottom = np.asarray(box, dtype=float) * [sx, sy, sx, sy]
    bw, bh = right - left, bottom - top
    if bw < max(14, .025 * width) or bh < max(12, .025 * height):
        return []
    surface = classes == 10
    for x0, y0, x1, y1 in occupied:
        ax = max(0, int(np.floor(x0 * sx)))
        ay = max(0, int(np.floor(y0 * sy)))
        bx = min(width, int(np.ceil(x1 * sx)))
        by = min(height, int(np.ceil(y1 * sy)))
        surface[ay:by, ax:bx] = False
    candidates = []
    for side, edge, direction in (("left",left,-1), ("right",right,1)):
        x0 = max(0, int(np.floor(edge - (.94 if direction < 0 else 0) * bw)))
        x1 = min(width, int(np.ceil(edge + (.94 if direction > 0 else 0) * bw)))
        y0 = max(0, int(np.floor(bottom - .22 * bh)))
        y1 = min(height, int(np.ceil(bottom + .18 * bh)))
        if x1-x0 < max(7,.44*bw) or y1-y0 < max(5,.18*bh):
            continue
        area = surface[y0:y1, x0:x1]
        _, labels = cv2.connectedComponents(np.ascontiguousarray(area, dtype=np.uint8), connectivity=8)
        yy, xx = np.indices(area.shape)
        lateral = (xx + x0 + .5 - edge) * direction
        depth = yy + y0 + .5
        contact = (lateral <= .22 * bw) & (np.abs(depth-bottom) <= .15 * bh)
        found = []
        for label in np.unique(labels[contact & area]):
            region = labels == label
            iy, ix = np.nonzero(region)
            if (np.count_nonzero(region & contact) < max(6,.008*bw*bh)
                    or np.ptp(ix) < .48*bw or np.ptp(iy) < .22*bh):
                continue
            interior = cv2.distanceTransform(np.pad(region,1).astype("uint8"), cv2.DIST_L2,5)[1:-1,1:-1]
            allowed = region & (lateral >= .26*bw) & (lateral <= .70*bw) & (interior >= max(4,.085*min(bw,bh)))
            if not np.any(allowed):
                continue
            py, px = np.unravel_index(np.argmax(np.where(allowed,interior,-1)), area.shape)
            radius = max(2, int(interior[py,px]/np.sqrt(2)) - 1)
            found.append((float(interior[py,px]), {
                "side":side, "point":[float((px+x0+.5)/sx), float((py+y0+.5)/sy)],
                "box":[float((px+x0-radius)/sx),float((py+y0-radius)/sy),
                       float((px+x0+radius+1)/sx),float((py+y0+radius+1)/sy)],
                "parking_seed_cells":int(np.count_nonzero(region & contact)),
                "connected_cells":int(region.sum()), "kind":"adjacent_parking",
            }))
        if found:
            candidates.append(max(found, key=lambda item:item[0])[1])
    return candidates


def _spans(row):
    positions = np.flatnonzero(np.diff(np.r_[False, row, False]))
    return zip(positions[::2], positions[1::2])


def behind_patch(classes, box, occupied, size):
    """Return a parking rectangle; the visual reader must separately confirm the rear orientation.

    A road-labelled extension requires a parking seed at the bumper and a marked
    boundary on the lane side opposite a curb or raised sidewalk. A road mask
    by itself cannot produce a candidate.
    """
    h, w = classes.shape
    sx, sy = w / size[0], h / size[1]
    left, top, right, bottom = np.asarray(box, dtype=float) * [sx, sy, sx, sy]
    bw, bh = right - left, bottom - top
    if bw < max(14, .025*w) or bh < max(12, .025*h):
        return None
    cx = (left+right)/2
    x0 = max(0,int(cx-2.45*bw)); x1 = min(w, int(cx+2.45*bw)+1)
    y0 = max(0,int(np.floor(bottom))); y1 = min(h,int(bottom+1.25*bh)+1)
    if y1-y0 < 6 or x1-x0 < 8:
        return None
    local = classes[y0:y1,x0:x1]
    mask = local == 10
    free = np.ones_like(mask)
    for a,b,c,d in occupied:
        lx,ty=max(0,int(np.floor(a*sx))-x0),max(0,int(np.floor(b*sy))-y0)
        rx,by=min(x1-x0,int(np.ceil(c*sx))-x0),min(y1-y0,int(np.ceil(d*sy))-y0)
        if rx>lx and by>ty:free[ty:by,lx:rx]=False
    for j,row in enumerate(local):
        for lx,rx in _spans((row==10)|(row==13)):
            if not .59*bw <= rx-lx <= 3.3*bw or lx==0 or rx==len(row):
                continue
            a,b=int(row[lx-1]),int(row[rx])
            if (a==24 and b in (2,15)) or (b==24 and a in (2,15)):
                mask[j,lx:rx]=True
    mask &= free
    _, components = cv2.connectedComponents(np.ascontiguousarray(mask, dtype=np.uint8),connectivity=8)
    yy,xx=np.indices(mask.shape);xx=xx+x0+.5;yy=yy+y0+.5
    bumper = (local==10) & mask & (yy<=bottom+max(4,.19*bh)) & (np.abs(xx-cx)<.50*bw)
    labels,frequency=np.unique(components[bumper],return_counts=True)
    winners=[]
    for label,count in zip(labels,frequency):
        if not label or count<max(8,round(.009*bw*bh)):continue
        comp=components==label
        if np.count_nonzero(comp)<.15*bw*bh:continue
        pad=np.pad(comp,1,constant_values=False)
        interior=cv2.distanceTransform(pad.astype('uint8'),cv2.DIST_L2,5)[1:-1,1:-1]
        # Keep a visible separation from the bumper and avoid extrapolating far down the street.
        eligible=comp & (yy>=bottom+.32*bh) & (yy<=bottom+.92*bh) & (interior>=max(3,.055*bw))
        if not eligible.any():continue
        score=interior/min(bw,bh)-.27*np.abs((yy-bottom)/bh-.56)
        iy,ix=np.unravel_index(np.argmax(np.where(eligible,score,-np.inf)),score.shape)
        radius=max(2,min(int(np.floor(interior[iy,ix]/np.sqrt(2)))-1,round(.15*bw),round(.20*bh)))
        patch=comp[iy-radius:iy+radius+1,ix-radius:ix+radius+1]
        if patch.shape!=(2*radius+1,2*radius+1) or not patch.all():continue
        out={'point':[(ix+x0+.5)/sx,(iy+y0+.5)/sy],
             'box':[(ix+x0-radius)/sx,(iy+y0-radius)/sy,(ix+x0+radius+1)/sx,(iy+y0+radius+1)/sy],
             'parking_seed_cells':int(count),'connected_cells':int(comp.sum()),
             'kind':'parking' if np.all(local[iy-radius:iy+radius+1,ix-radius:ix+radius+1]==10) else 'marked_lane_edge_to_curb'}
        winners.append((score[iy,ix],out))
    return max(winners,key=lambda pair:pair[0])[1] if winners else None
