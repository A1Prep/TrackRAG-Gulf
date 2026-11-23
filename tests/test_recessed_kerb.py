import numpy as np
import pytest
from PIL import Image

from trackrag.dropoff import recessed_kerb


def road(fault=None):
    h,w=400,700
    y,x=np.indices((h,w))
    baseline=404+1.43*(y-206)
    offset=np.interp(y,[0,207,224,236,266,282,300,399],[0,0,43,98,100,58,0,0])
    if fault=="straight":
        offset[:]=0
    elif fault=="permanent_extra_lane":
        offset[y>238]=100
    outer=baseline+offset
    labels=np.where(x<outer,13,np.where(x<outer+8,2,15)).astype(np.uint8)
    labels[:156]=27
    rgb=np.full((h,w,3),[90,92,94],np.uint8)
    rgb[labels==2]=[197,184,169]
    rgb[labels==15]=[185,151,134]
    rgb[labels==27]=[157,182,208]
    boxes=[]
    aperture=(y>=207)&(y<=300)&(abs(x-baseline)<4)
    if fault=="painted_traffic_lane":
        labels[aperture]=24
        rgb[aperture]=240
    elif fault=="unlabelled_lane_paint":
        rgb[aperture]=240
    elif fault=="offset_optical_kerb":
        cap=x-(baseline-.035*w)
        mouth=(y>=205)&(y<=304)
        rgb[mouth & (cap>-3) & (cap<=1)]=[215,207,201]
        rgb[mouth & (cap>1) & (cap<7)]=[55,53,50]
    elif fault=="barrier":
        labels[aperture]=6
        rgb[aperture]=[179,153,109]
    elif fault=="curb_in_front_of_recess":
        labels[aperture]=2
        rgb[aperture]=[198,180,164]
    elif fault=="planting":
        rgb[(x>baseline)&(x<outer)]=[56,132,57]
    elif fault=="crossing":
        labels[232:264,435:585]=23
    elif fault=="blocked_mouth":
        boxes=[[401,208,501,299]]
    elif fault=="no_visible_edge":
        rgb[:]=89
    return Image.fromarray(rgb),labels,{"size":[w,h],"current_boxes":boxes}


def test_pocket_beyond_both_kerb_approaches_preserves_the_selected_pixel():
    im,classes,ctx=road()
    [only]=recessed_kerb.candidates(im,classes,ctx)
    assert only["evidence"]["side"]=="right"
    assert only["evidence"]["coverage"]=="interior_point_only"
    x,y=only["point"]
    assert recessed_kerb.marked(im,classes,ctx,[x-2,y-2,x+4,y+4])["point"]==[x+1,y+1]
    assert recessed_kerb.marked(im,classes,ctx,[343,y-2,347,y+2]) is None
    assert recessed_kerb.marked(im,classes,ctx,[300,y-8,x+8,y+8]) is None
    # Identical evidence on the other side produces the same result mirrored.
    flipped=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    reflected=recessed_kerb.candidates(flipped,np.fliplr(classes),ctx)
    assert len(reflected)==1 and reflected[0]["evidence"]["side"]=="left"
    assert np.allclose(reflected[0]["point"],[im.width-1-x,y],atol=.01)


@pytest.mark.parametrize("fault",["straight","permanent_extra_lane","painted_traffic_lane","unlabelled_lane_paint","offset_optical_kerb",
                                 "barrier","curb_in_front_of_recess","planting","crossing","blocked_mouth","no_visible_edge"])
def test_extra_asphalt_alone_never_establishes_a_recess(fault):
    im,classes,ctx=road(fault)
    assert not recessed_kerb.candidates(im,classes,ctx)
    assert recessed_kerb.marked(im,classes,ctx,[488,248,492,252]) is None
