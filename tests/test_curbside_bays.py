import cv2
import numpy as np
from PIL import Image
import pytest

from trackrag.dropoff import curbside_bays, painted_bays


def parallel_space(*, mirrored=False, ends=True, curb=True, yellow=False, obstruction=False):
    """A synthetic camera sees two transverse T marks, a kerb slope and an adjacent through lane."""
    w,h=840,480
    im=np.full((h,w,3),[96,99,100],np.uint8)
    classes=np.full((h,w),13,np.uint8)
    for y in range(h):
        edge=round(563+(y-294)*2.05)
        if curb and 0<=edge<w:
            classes[y,edge:edge+4]=2
            classes[y,edge+4:]=15
            im[y,edge:]=[137,134,131]
    far=(451,294)
    near=(578,382)
    for x,y in (far,near):
        right=round(563+(y-294)*2.05)
        cv2.line(im,(x,y),(right-2,y),(176,176,169),3,cv2.LINE_AA)
        if ends:
            cv2.line(im,(x-11,y-8),(x+14,y+10),(179,179,173),4,cv2.LINE_AA)
    if yellow:
        cv2.line(im,(516,288),(649,385),(248,196,22),11,cv2.LINE_AA)
    boxes=[[528,310,557,341]] if obstruction else []
    if mirrored:
        im,classes=im[:,::-1].copy(),classes[:,::-1].copy()
        boxes=[[w-1-r,t,w-1-l,b] for l,t,r,b in boxes]
    return Image.fromarray(im),classes,{"size":[w,h],"current_boxes":boxes}


@pytest.mark.parametrize("mirrored",[False,True])
def test_white_end_markings_and_curb_allow_one_parallel_bay_on_road_label(mirrored):
    im,classes,ctx=parallel_space(mirrored=mirrored)
    opts=curbside_bays.candidates(im,classes,ctx)
    assert len(opts)==1
    option=opts[0]
    assert option["evidence"]["visible_parking_cells"]==0
    poly=np.float32(option["polygon"])
    inside=(im.width-1-550.,333.) if mirrored else (550.,333.)
    lane=(im.width-1-345.,333.) if mirrored else (345.,333.)
    assert cv2.pointPolygonTest(poly,inside,False)>0
    assert cv2.pointPolygonTest(poly,lane,False)<0
    frame={"width":im.width,"height":im.height}
    box=lambda xy:[xy[0]-2,xy[1]-2,xy[0]+2,xy[1]+2]
    assert painted_bays.matching_options(opts,box(inside),inside,frame,classes)==opts
    assert not painted_bays.matching_options(opts,box(lane),lane,frame,classes)


@pytest.mark.parametrize("setting",[{"curb":False},{"ends":False},{"yellow":True},{"obstruction":True}])
def test_through_road_ordinary_stripes_occupied_and_coloured_edge_are_withheld(setting):
    im,classes,ctx=parallel_space(**setting)
    assert not curbside_bays.candidates(im,classes,ctx)


def test_crossing_through_footprint_is_always_rejected():
    im,classes,ctx=parallel_space()
    classes[334:343,526:622]=23
    assert not curbside_bays.candidates(im,classes,ctx)
