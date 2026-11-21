"""A region head trained on annotated ground regions bracketed by two visible detections."""
import numpy as np

from .contracts import validate_box


def coordinates(boxes, size):
    if len(boxes) != 2:
        raise ValueError("Two distinct boxes are required")
    normalized = np.array([validate_box(b,*size) for b in boxes])/np.tile(size,2)
    left,right = sorted(normalized,key=lambda b:b[0])
    gap = right[0]-left[2]
    if gap <= 0:
        return None
    lh,rh = left[3]-left[1],right[3]-right[1]
    depth = max(.04,.5*(lh+rh))
    origin = np.array([left[2],.5*(left[3]+right[3])])
    scale = np.array([gap,depth])
    x = np.array([1.,gap,origin[0],origin[1],lh,rh,left[2]-left[0],right[2]-right[0],
                  (right[3]-left[3])/depth,np.log((lh+.015)/(rh+.015))])
    return x,origin,scale


def fit(rows):
    x, y = [], []
    for e in rows:
        f,o,s = coordinates(e["anchors"],e["size"])
        p = np.array(e["point"])/e["size"]
        b = np.array(e["box"])/np.tile(e["size"],2)
        distances = np.maximum(np.r_[p-b[:2],b[2:]-p],.003)
        x.append(f)
        y.append(np.r_[(p-o)/s, np.log(distances/np.tile(s,2))])
    if len(x) < 24:
        raise ValueError("Too few bracketed ground examples to fit a two-landmark head")
    x,y = np.stack(x),np.stack(y)
    coefs = []
    for target in y.T:
        weights = np.ones(len(x))
        for _ in range(5):
            xw = x.T*weights
            c = np.linalg.solve(xw@x+np.diag([.001]+[2.]*(x.shape[1]-1)),xw@target)
            residual = np.abs(x@c-target)
            weights = np.minimum(1.,max(.12,1.5*np.median(residual))/np.maximum(residual,1.e-7))
        coefs.append(c)
    return {"kind":"visible_horizontal_gap_regression_v1","coefficients":np.column_stack(coefs).tolist(),
            "training_count":len(rows)}


def predict(model, boxes, size):
    recipe = coordinates(boxes,size)
    if recipe is None:
        return None
    f,o,s = recipe
    c = np.asarray(model["coefficients"])
    if model.get("kind") != "visible_horizontal_gap_regression_v1" or c.shape != (10,6) or not np.isfinite(c).all():
        raise ValueError("Invalid two-landmark region checkpoint")
    v = f@c
    pt = np.clip(o+s*v[:2],0,1-1/np.array(size))
    dist = np.tile(s,2)*np.exp(np.clip(v[2:],-6,4))
    box = np.clip(np.r_[pt,pt]+dist*[-1,-1,1,1],0,1)
    return (pt*size).tolist(),(box*np.tile(size,2)).tolist()
