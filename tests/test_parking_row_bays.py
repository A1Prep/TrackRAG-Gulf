import cv2
import numpy as np
from PIL import Image, ImageDraw

from trackrag.dropoff import parking_row_bays, painted_bays


def parking_row(*, rows=4, parking=True, scale=1):
    size=(800,480)
    image=Image.new('RGB',size,(52,55,60))
    labels=np.full((size[1],size[0]),13,dtype='uint8')
    labels[:180]=27
    if parking:
        labels[232:386,250:755]=10
    draw=ImageDraw.Draw(image)
    for left,right,y in [(284,580,250),(310,630,278),(347,683,313),(393,745,356)][:rows]:
        for x in np.linspace(left,right,9):
            py=y-.025*(x-left)
            draw.ellipse([x-4,py-2,x+4,py+1],fill=(210,215,220))
    if scale != 1:
        image=image.resize((round(size[0]*scale),round(size[1]*scale)),Image.Resampling.BICUBIC)
    context={'size':list(image.size),'current_boxes':[]}
    return image,labels,context


def test_studded_parking_rows_return_actual_ground_and_separate_neighbours():
    im,labels,ctx=parking_row()
    options=parking_row_bays.candidates(im,labels,ctx)
    assert len(options)==3
    for i,option in enumerate(options):
        evidence=option['evidence']
        assert evidence['boundary_kind']=='repeated_pavement_studs' and evidence['external_stud_rows']>=1
        assert min(evidence['stud_counts'])>=6 and evidence['coverage']=='visible_portion'
        assert evidence['painted_boundaries']==[]
        assert cv2.pointPolygonTest(np.float32(option['polygon']),tuple(option['point']),True)>5
        x,y=option['point'];frame={'width':im.width,'height':im.height}
        assert painted_bays.matching_options(options,[x-1,y-1,x+1,y+1],[x,y],frame,labels)==[option]
        for other in options[i+1:]:
            assert cv2.intersectConvexConvex(np.float32(option['polygon']),np.float32(other['polygon']))[0]<1


def test_no_optical_row_single_divider_crossing_or_through_lane_can_offer():
    for rows,parking in [(0,True),(1,True),(4,False)]:
        image,labels,context=parking_row(rows=rows,parking=parking)
        assert parking_row_bays.candidates(image,labels,context)==[]
    image,labels,context=parking_row()
    labels[230:390,245:765]=8
    assert parking_row_bays.candidates(image,labels,context)==[]


def test_rows_with_occupied_footprints_do_not_leave_an_automatic_bay():
    image,labels,context=parking_row()
    context['current_boxes']=[[285,194,725,380]]
    assert parking_row_bays.candidates(image,labels,context)==[]


def test_source_pixel_scaling_and_actual_vehicle_boundaries_are_preserved():
    image,labels,context=parking_row(scale=2)
    context['current_boxes']=[[1180,445,1580,750]]
    options=parking_row_bays.candidates(image,labels,context)
    assert options
    assert any(option['point'][0]>800 and option['point'][1]>480 for option in options)
    obstacle=np.float32([[1180,445],[1580,445],[1580,750],[1180,750]])
    assert all(cv2.intersectConvexConvex(np.float32(option['polygon']),obstacle)[0]<1 for option in options)


def optical_row(*, caps=True, curb=True, close_vehicle=True):
    image=Image.new('RGB',(800,480),(107,109,107))
    labels=np.full((480,800),13,dtype='uint8')
    draw=ImageDraw.Draw(image)
    if curb:
        labels[:218]=15
        labels[208:218]=2
        draw.rectangle((0,208,799,217),fill=(139,139,135))
    for fx,fy,bx in [(84,318,345),(177,334,440),(274,351,545),(381,370,667)]:
        draw.line([(fx,fy),(bx,217)],fill=(137,138,133),width=3)
        if caps:
            draw.line([(fx-13,fy-2),(fx+13,fy+2)],fill=(148,149,144),width=3)
    box=[25,198,115,283] if close_vehicle else [620,60,716,120]
    draw.rectangle(box,fill=(80,83,85))
    return image,labels,{'size':list(image.size),'current_boxes':[box]}


def test_optical_stall_row_needs_real_end_caps_common_curb_and_nearby_vehicle():
    image,labels,ctx=optical_row()
    detected=parking_row_bays.candidates(image,labels,ctx)
    assert detected
    for row in detected:
        e=row['evidence']
        assert e['boundary_kind']=='curb_facing_capped_row' and e['visible_parking_cells']==0
        assert e['coverage']=='measured_stall' and e['external_capped_dividers']>=1
        assert len(e['optical_dividers'])==len(e['painted_boundaries'])==2
    for args in ({'caps':False},{'curb':False},{'close_vehicle':False}):
        image,labels,ctx=optical_row(**args)
        assert parking_row_bays.candidates(image,labels,ctx)==[]


def test_signed_crosswalk_and_current_occupant_veto_optical_row():
    image,labels,ctx=optical_row()
    labels[218:395,65:680]=23
    assert parking_row_bays.candidates(image,labels,ctx)==[]
    image,labels,ctx=optical_row()
    ctx['current_boxes'].append([79,210,700,390])
    assert parking_row_bays.candidates(image,labels,ctx)==[]


def test_straight_painted_stalls_are_bounded_by_signed_aisle_and_kerb():
    image=Image.new('RGB',(800,480),(73,77,81))
    labels=np.full((480,800),13,dtype='uint8')
    draws=ImageDraw.Draw(image)
    fronts=[(264,237),(298,260),(337,287),(384,319),(438,357)]
    backs=[(535,226),(570,246),(611,271),(661,300),(721,334)]
    cv2.fillConvexPoly(labels,np.int32([*fronts,*reversed(backs)]),10)
    cv2.polylines(labels,[np.int32(backs)],False,2,5)
    for front,back in zip(fronts[1:-1],backs[1:-1]):
        draws.line([front,back],fill=(189,192,195),width=3)
    ctx={'size':list(image.size),'current_boxes':[]}
    options=parking_row_bays.candidates(image,labels,ctx)
    assert options and all(o['evidence']['boundary_kind']=='aisle_to_curb_parking_row' for o in options)
    for option in options:
        e=option['evidence']
        assert all(e['rear_curb_seen']) and e['coverage']=='measured_stall' and e['external_dividers']>0
        assert e['aisle_occluded_by_camera'] is False
    labels[labels==2]=13
    assert parking_row_bays.candidates(image,labels,ctx)==[]
