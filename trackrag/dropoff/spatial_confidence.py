"""Estimate native mask support for two frozen, image-conditioned destination candidates."""
import math

import numpy as np


HEADS = ("instruction_conditioned","language_blind")
SELECTIVE_THRESHOLD = 0.30
L2 = 2.0


def _candidate(row, size):
    if row is None or row.get("status","geometry")!="geometry":
        return None
    w,h=size
    x,y=row["destination_point_xy"]
    l,t,r,b=row["destination_box_xyxy"]
    p,road,cells=row["peak_mass"],row["point_road_probability"],row["eligible_grid_cells"]
    if not (w>0 and h>0 and 0<=l<=x<r<=w and 0<=t<=y<b<=h
            and 0<p<=1 and 0<=road<=1 and type(cells) is int and 0<cells<=2048):
        raise ValueError("Unbound or invalid native spatial candidate")
    return np.array((x/w,y/h,l/w,t/h,r/w,b/h,math.log(p),road,cells/2048),dtype=float)


def features(candidates, size):
    current=[_candidate(candidates.get(name),size) for name in HEADS]
    rows=[]
    for i,a in enumerate(current):
        if a is None:
            rows.append(None)
            continue
        b=current[1-i]
        if b is None:
            b=a
        area=(a[4]-a[2])*(a[5]-a[3])
        other_area=(b[4]-b[2])*(b[5]-b[3])
        shared=max(min(a[4],b[4])-max(a[2],b[2]),0)*max(min(a[5],b[5])-max(a[3],b[3]),0)
        joint=np.array([a[0],a[1],a[4]-a[2],a[5]-a[3],math.log(area),a[6],a[7],a[8],
                        a[0]-b[0],a[1]-b[1],math.hypot(a[0]-b[0],a[1]-b[1]),
                        shared/(area+other_area-shared),a[6]-b[6]],dtype=float)
        identity=float(i==0)
        rows.append(np.r_[identity,joint,identity*joint])
    return rows


def fit(x, y):
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    if x.ndim!=2 or x.shape[1]!=27 or y.shape!=(len(x),) or not np.isfinite(x).all() or set(y)!={0.,1.}:
        raise ValueError("Calibration requires finite per-candidate features and binary native-mask outcomes")
    mean=x.mean(axis=0)
    scale=x.std(axis=0)
    scale[scale<1e-5]=1.
    a=np.c_[np.ones(len(x)),(x-mean)/scale]
    beta=np.zeros(a.shape[1])
    penalty=np.ones(a.shape[1])*L2
    penalty[0]=0
    def objective(b):
        scores=a@b
        return np.sum(np.logaddexp(0.,scores)-y*scores)+.5*np.dot(penalty,b*b)
    for _ in range(40):
        chance=1/(1+np.exp(-np.clip(a@beta,-40.,40.)))
        gradient=a.T@(chance-y)+penalty*beta
        hessian=a.T@((chance*(1-chance))[:,None]*a)+np.diag(penalty)
        step=np.linalg.solve(hessian,gradient)
        if np.max(np.abs(step))<1e-8:
            break
        rate=1.
        loss=objective(beta)
        while rate>1e-7 and objective(beta-rate*step)>loss:
            rate*=.5
        beta-=rate*step
    return {"kind":"candidate_mask_hit_logistic_v1","mean":mean.tolist(),"scale":scale.tolist(),"coefficients":beta.tolist()}


def choose(model, candidates, size):
    vectors=features(candidates,size)
    mean,scale,beta=[np.asarray(model[k],dtype=float) for k in ("mean","scale","coefficients")]
    if (model.get("kind")!="candidate_mask_hit_logistic_v1" or mean.shape!=(27,) or scale.shape!=(27,)
            or beta.shape!=(28,) or not all(np.isfinite(v).all() for v in (mean,scale,beta)) or np.any(scale<=0)):
        raise ValueError("Invalid spatial calibration model")
    chances={}
    for name,row in zip(HEADS,vectors,strict=True):
        if row is not None:
            logit=beta[0]+np.dot((row-mean)/scale,beta[1:])
            chances[name]=float(1/(1+np.exp(-np.clip(logit,-40.,40.))))
    best=max(chances,key=chances.get) if chances else None
    return {"head":best,"mask_hit_probabilities":chances,"meets_selected_threshold":bool(best and chances[best]>=SELECTIVE_THRESHOLD)}
