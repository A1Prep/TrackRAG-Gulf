"""Prospective, opt-in facing/oblique parking-row geometry; no stopping-safety claim.

The road bitmask alone never certifies a roadside.  A separate semantic sidewalk
prediction or visible parking/curb paint must witness the back of the gap.  All
coordinates at this boundary are original-image pixels.  No source identifier,
destination annotation or scene-dependent threshold is accepted.
"""
from hashlib import sha256
import json
import math

import cv2
import numpy as np

from . import pair_geometry
from .contracts import validate_box
from .projected_pair import MOTOR, Ineligible, need
from .road_cache import ROAD_RECIPE_SHA256, SIDEWALK_RECIPE_SHA256, mask_shape, mask_sha256


VERSION = "side_facing_roadside_gap_v1"


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _image_hash(image, size):
    if image is None:
        return None
    need(image.size == tuple(size), "roadside_image_dimensions_differ")
    return sha256(image.convert("RGB").tobytes()).hexdigest()


def _geometry(observations, size):
    need(len(observations) == 2 and observations[0]["track_id"] != observations[1]["track_id"]
         and all(t["class_name"] in MOTOR for t in observations), "two_distinct_current_vehicles_required")
    boxes = [validate_box(t["bbox_xyxy"], *size) for t in observations]
    left, right = sorted(boxes, key=lambda b: (b[0]+b[2])/2)
    heights = [b[3]-b[1] for b in boxes]
    need(max(heights) <= 3*min(heights), "incompatible_anchor_scales")
    intersection = max(0, min(left[2],right[2])-max(left[0],right[0])) * max(0,min(left[3],right[3])-max(left[1],right[1]))
    need(intersection <= .5*min((b[2]-b[0])*(b[3]-b[1]) for b in boxes), "selected_vehicles_substantially_occlude_each_other")
    height = min(heights)
    if left[2] < right[0]:
        # A facing parking row can have indistinguishable contact heights.  Its
        # observable gap is between the inner box sides, around the contact line.
        x0,x1 = left[2],right[0]
        y0,y1 = left[3],right[3]
        mode = "inner_sides_and_contact_band"
        polygon = [[x0,y0-.2*height],[x1,y1-.2*height],
                   [x1,min(size[1],y1+.4*height)],[x0,min(size[1],y0+.4*height)]]
    else:
        # Oblique rows expose a slanted ground strip even when detector x-ranges
        # overlap.  Keep the full swept contact polygon and subtract all objects.
        far,near = sorted(boxes,key=lambda b:b[3])
        need(near[3]-far[3] >= .05*height, "no_visible_between_gap")
        polygon = [[far[0],far[3]],[far[2],far[3]],[near[2],near[3]],[near[0],near[3]]]
        mode = "oblique_contact_sweep"
    bounds = [min(p[0] for p in polygon),max(0,min(p[1] for p in polygon)),
              max(p[0] for p in polygon),min(size[1],max(p[1] for p in polygon))]
    validate_box(bounds,*size)
    return {"contact_polygon_xy":polygon,"regression_inputs_xyxy":boxes,
            "center_corridor_xyxy":bounds,"orientation_geometry":mode,
            "minimum_anchor_height":height}


def _inside_polygon(polygon, xs, ys, dx, dy):
    # cv2's polygon test is independent of interpolation/rounded raster vertices.
    contour = np.asarray(polygon,dtype=np.float32)
    a,b,c,d = min(p[0] for p in polygon),min(p[1] for p in polygon),max(p[0] for p in polygon),max(p[1] for p in polygon)
    out = np.zeros((len(ys),len(xs)),bool)
    for y in np.flatnonzero((ys>=b)&(ys+dy<=d)):
        for x in np.flatnonzero((xs>=a)&(xs+dx<=c)):
            out[y,x] = all(cv2.pointPolygonTest(contour,(float(px),float(py)),False)>=0
                           for px,py in ((xs[x],ys[y]),(xs[x]+dx,ys[y]),(xs[x],ys[y]+dy),(xs[x]+dx,ys[y]+dy)))
    return out


def _border_component(binary):
    count,labels = cv2.connectedComponents(binary.astype(np.uint8),connectivity=8)
    ids = np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])
    return np.isin(labels,ids[ids!=0]) if count>1 else np.zeros_like(binary)


def _row_context(observations,tracks):
    """Find an additional distinct contact aligned with the observed vehicle row.

    This is only a paint-context check: a still picture cannot establish parking
    state.  Duplicate/nested detections of either selected car do not count.
    """
    boxes=sorted([o["bbox_xyxy"] for o in observations],key=lambda b:(b[0]+b[2])/2)
    a,b=boxes;ax,bx=(a[0]+a[2])/2,(b[0]+b[2])/2
    height=min(a[3]-a[1],b[3]-b[1])
    selected={o["track_id"] for o in observations}
    for t in tracks:
        if t["track_id"] in selected or t["class_name"] not in MOTOR:continue
        box=t["bbox_xyxy"];x=(box[0]+box[2])/2;area=(box[2]-box[0])*(box[3]-box[1])
        duplicated=any(max(0,min(box[2],v[2])-max(box[0],v[0]))*max(0,min(box[3],v[3])-max(box[1],v[1]))>.5*area for v in boxes)
        expected=a[3]+(x-ax)/max(1,bx-ax)*(b[3]-a[3])
        if (not duplicated and .2*height<=box[3]-box[1]<=3*height
                and abs(box[3]-expected)<=.6*height and (x<ax or x>bx)):
            return {"additional_track_id":t["track_id"],"additional_box_xyxy":box}
    return None


def _paint_witness(image, mask, occupied, corridor, rear, radius, *, row_context=False):
    if image is None:
        return np.zeros_like(mask), []
    rgb = np.asarray(image.convert("RGB").resize((mask.shape[1],mask.shape[0])))
    hsv = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    # Photometric stripe evidence must precede raster/Hough geometry.  Thresholding
    # bright pixels alone turns uniformly pale pavement into arbitrary "lines".
    kernel=np.ones((7,7),np.uint8)
    bright=cv2.morphologyEx(hsv[:,:,2],cv2.MORPH_TOPHAT,kernel)>25
    chromatic=cv2.morphologyEx(hsv[:,:,1],cv2.MORPH_TOPHAT,kernel)>40
    paint = (((hsv[:,:,1]<65)&(hsv[:,:,2]>110)&bright) |
             ((hsv[:,:,0]>=15)&(hsv[:,:,0]<=40)&(hsv[:,:,1]>90)&(hsv[:,:,2]>100)&(bright|chromatic)))
    # A generic untracked nonroad object is never a witness.  Paint must lie on
    # actual predicted road beside a border-connected exterior, not an interior
    # vehicle hole/raised island, and form two different line directions.
    exterior = _border_component(~mask & ~occupied)
    edge_band = cv2.dilate(exterior.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    local = cv2.dilate(corridor.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
    # A curb/sidewalk is behind the row; a genuine parking bay's front paint is
    # allowed in front of wheel contacts.  The latter requires a distinct aligned
    # third vehicle as context and remains bounded to the same local ground span.
    pixels = paint & mask & ~occupied & local & (np.ones_like(rear) if row_context else rear)
    lines = cv2.HoughLinesP(pixels.astype(np.uint8)*255,1,np.pi/180,threshold=6,
                            minLineLength=6,maxLineGap=2)
    if lines is None:
        return np.zeros_like(mask), []
    segments = [list(map(int,line)) for line in np.asarray(lines).reshape(-1,4)[:100]]
    witness = np.zeros_like(mask,np.uint8)
    accepted=[]
    for i,a in enumerate(segments):
        va=np.array(a[2:])-a[:2]
        for b in segments[i+1:]:
            vb=np.array(b[2:])-b[:2]
            cosine=abs(float(np.dot(va,vb)))/(np.linalg.norm(va)*np.linalg.norm(vb))
            if cosine > math.cos(math.radians(25)):
                continue
            distance=min(np.linalg.norm(np.array(p)-q) for p in (a[:2],a[2:]) for q in (b[:2],b[2:]))
            if distance > max(3,radius):
                continue
            boundary=any(edge_band[y,x] for x,y in (a[:2],a[2:],b[:2],b[2:]))
            if not boundary and not row_context:
                continue
            cv2.line(witness,tuple(a[:2]),tuple(a[2:]),1,1)
            cv2.line(witness,tuple(b[:2]),tuple(b[2:]),1,1)
            accepted.extend((a,b))
    return witness.astype(bool), accepted


def eligible_cells(mask,size,observations,tracks,*,image=None,sidewalk_mask=None):
    need(isinstance(mask,np.ndarray) and mask.dtype==np.bool_ and mask.shape==mask_shape(size),"unsupported_road_mask")
    need(mask.any(),"no_supported_neighborhood")
    _image_hash(image,size)
    geometry = _geometry(observations,size)
    h,w=mask.shape;dx,dy=size[0]/w,size[1]/h
    xs,ys=np.arange(w)*dx,np.arange(h)*dy
    occupied=np.zeros_like(mask)
    for t in tracks:
        a,b,c,d=validate_box(t["bbox_xyxy"],*size)
        occupied |= ((xs<c)&(xs+dx>a))[None,:] & ((ys<d)&(ys+dy>b))[:,None]
    corridor = _inside_polygon(geometry["contact_polygon_xy"],xs,ys,dx,dy)
    free=mask & corridor & ~occupied
    need(free.any(),"no_unoccupied_between_ground")
    radius=max(2,math.ceil(.3*geometry["minimum_anchor_height"]/dy))
    # Independent roadside evidence must be at/behind the vehicle contact band,
    # not a foreground island on the camera side of an otherwise through-lane gap.
    boxes=geometry["regression_inputs_xyxy"]
    left,right=sorted(boxes,key=lambda b:(b[0]+b[2])/2)
    cx0,cx1=(left[0]+left[2])/2,(right[0]+right[2])/2
    contact=left[3]+np.clip(((xs+.5*dx)-cx0)/max(dx,cx1-cx0),0,1)*(right[3]-left[3])
    rear = ((ys+.5*dy)[:,None] <= contact[None,:]+.1*geometry["minimum_anchor_height"])
    local = cv2.dilate(corridor.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
    witness=np.zeros_like(mask);kinds=[];segments=[]
    if sidewalk_mask is not None:
        need(isinstance(sidewalk_mask,np.ndarray) and sidewalk_mask.dtype==np.bool_ and sidewalk_mask.shape==mask.shape,
             "unsupported_sidewalk_evidence")
        need(not (sidewalk_mask & mask).any(),"contradictory_semantic_surfaces")
        # Closed interior semantic components could be raised islands.  Require
        # connection to the observed exterior, then actual adjacency to road.
        sidewalk=_border_component(sidewalk_mask & ~occupied)
        adjacent=cv2.dilate(mask.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        witness |= sidewalk & adjacent & local & rear
        if witness.any():kinds.append("independent_sidewalk_semantics")
    row_context=_row_context(observations,tracks)
    paint,segments=_paint_witness(image,mask,occupied,corridor,rear,radius,row_context=row_context is not None)
    if paint.any():
        witness |= paint
        kinds.append("visible_nonparallel_parking_paint_with_row" if row_context else "visible_nonparallel_boundary_paint")
    need(witness.any(),"no_independent_roadside_evidence")
    distance=cv2.distanceTransform((~witness).astype(np.uint8),cv2.DIST_L2,5)
    eligible=free & (distance<=radius)
    need(eligible.any(),"no_ground_adjacent_to_independent_roadside")
    evidence={"witness_kinds":kinds,"witness_sha256":mask_sha256(witness),
              "sidewalk_mask_sha256":mask_sha256(sidewalk_mask) if sidewalk_mask is not None else None,
              "sidewalk_recipe_sha256":SIDEWALK_RECIPE_SHA256 if sidewalk_mask is not None else None,
              "image_pixels_sha256":_image_hash(image,size),"paint_segments_mask_xy":segments,
              "aligned_row_context":row_context,
              "maximum_witness_distance_mask_cells":radius}
    return eligible,geometry,witness,evidence


def estimate(fitted_geometry,mask,size,observations,tracks,text,*,image=None,sidewalk_mask=None):
    eligible,geometry,witness,evidence=eligible_cells(mask,size,observations,tracks,image=image,sidewalk_mask=sidewalk_mask)
    boxes=geometry["regression_inputs_xyxy"]
    forecast=pair_geometry.predict(fitted_geometry.pair_regression,boxes,size)
    neighbors=[]
    if forecast is None:
        point,region,neighbors=fitted_geometry.predict_with_evidence(boxes,size,text)
        estimator="learned_anchor_envelope_and_retrieved_offsets"
    else:
        point,region=forecast;estimator="learned_horizontal_pair_regression"
    region=validate_box(region,*size)
    point=np.asarray(point,dtype=float)
    need(point.shape==(2,) and np.isfinite(point).all(),"no_prediction_from_trained_pair_head")
    windows=np.lib.stride_tricks.sliding_window_view(eligible,(3,3)).all(axis=(2,3))
    rows,cols=np.nonzero(windows)
    need(len(rows)>0,"no_supported_neighborhood")
    rows,cols=rows+1,cols+1;h,w=mask.shape;dx,dy=size[0]/w,size[1]/h
    scale=np.maximum([region[2]-region[0],region[3]-region[1]],[3*dx,3*dy])
    cost=(((cols+.5)*dx-point[0])/scale[0])**2+(((rows+.5)*dy-point[1])/scale[1])**2
    i=int(cost.argmin());cy,cx=int(rows[i]),int(cols[i])
    proposed=[(cx+.5)*dx,(cy+.5)*dy];box=[(cx-1)*dx,(cy-1)*dy,(cx+2)*dx,(cy+2)*dy]
    wy,wx=np.nonzero(witness);j=int(((wx-cx)**2+(wy-cy)**2).argmin())
    return {**geometry,**evidence,"guard":VERSION,"point":proposed,"box":box,
            "learned_point":point.tolist(),"learned_region":region,"learned_estimator":estimator,
            "retrieved_training_examples":neighbors,"learned_projection_cost":float(cost[i]),
            "supported_three_by_three_centers":len(rows),"road_mask_sha256":mask_sha256(mask),
            "road_recipe_sha256":ROAD_RECIPE_SHA256,"mask_shape":list(mask.shape),
            "witness_mask_cell_xy":[int(wx[j]),int(wy[j])],"candidate_cell_xy":[cx,cy]}


def validate_patch(patch,observations,tracks,size,point,box):
    geometry=_geometry(observations,size)
    validate_box(box,*size)
    need(patch.get("guard")==VERSION and patch.get("point")==point and patch.get("box")==box
         and all(patch.get(k)==v for k,v in geometry.items()),"curb_facing_geometry_changed")
    need(box[0]<=point[0]<box[2] and box[1]<=point[1]<box[3],"curb_facing_point_outside_patch")
    need(not any(box[0]<t["bbox_xyxy"][2] and box[2]>t["bbox_xyxy"][0]
                 and box[1]<t["bbox_xyxy"][3] and box[3]>t["bbox_xyxy"][1] for t in tracks),"curb_facing_patch_occupied")


def validate_mask(patch,mask,size,observations,tracks,*,image=None,sidewalk_mask=None):
    point,box=patch.get("point"),patch.get("box")
    validate_patch(patch,observations,tracks,size,point,box)
    eligible,geometry,witness,evidence=eligible_cells(mask,size,observations,tracks,image=image,sidewalk_mask=sidewalk_mask)
    need(patch.get("road_mask_sha256")==mask_sha256(mask)
         and patch.get("road_recipe_sha256")==ROAD_RECIPE_SHA256 and patch.get("mask_shape")==list(mask.shape)
         and all(patch.get(k)==v for k,v in evidence.items()),"curb_facing_surface_evidence_changed")
    h,w=mask.shape;dx,dy=size[0]/w,size[1]/h
    cells=np.array(box)/[dx,dy,dx,dy];rounded=np.rint(cells).astype(int)
    a,b,c,d=rounded
    need(np.allclose(cells,rounded,atol=1.e-8,rtol=0) and 0<=a<c<=w and 0<=b<d<=h
         and c-a==3 and d-b==3 and eligible[b:d,a:c].all(),"curb_facing_patch_not_supported")
    cx,cy=a+1,b+1
    need(point==[(cx+.5)*dx,(cy+.5)*dy] and patch.get("candidate_cell_xy")==[cx,cy],"curb_facing_point_changed")
    wx,wy=patch.get("witness_mask_cell_xy",[-1,-1])
    need(type(wx) is int and type(wy) is int and 0<=wx<w and 0<=wy<h and witness[wy,wx],"curb_facing_witness_changed")


def reproduce(fitted_geometry,patch,mask,size,observations,tracks,text,*,image=None,sidewalk_mask=None):
    """Replay the learned winner before confirmation; a different valid cell is not interchangeable."""
    expected=estimate(fitted_geometry,mask,size,observations,tracks,text,image=image,sidewalk_mask=sidewalk_mask)
    need(_digest(expected)==_digest(patch),"curb_facing_fitted_winner_changed")
    return True
