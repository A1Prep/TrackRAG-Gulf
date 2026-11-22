"""Software-only geometry and canvas checks for the image marking variant."""
from copy import deepcopy

from PIL import Image
import pytest

from tools.dropoff_legible_marks import draw_legible_candidates, smart_spatial_size


@pytest.mark.parametrize("width,height,expected",[(1600,900,(384,672)),(2560,1440,(384,672)),(900,1600,(672,384)),(512,512,(512,512))])
def test_generic_qwen_aligned_size(width,height,expected):
    assert smart_spatial_size(height,width)==expected


def test_marks_scale_geometry_exactly_and_only_text_moves_at_edges(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(1600,900),'gray').save(source)
    options=[{'candidate_id':'R01','point_original_px':[1599.5,899.5],'bbox_original_px':[1590,890,1600,900]},
             {'candidate_id':'R02','point_original_px':[0.5,0.5],'bbox_original_px':[0,0,10,10]}]
    original=deepcopy(options)
    output=tmp_path/'marks.png'
    report=draw_legible_candidates(source,options,{'width':1600,'height':900},output)
    assert options==original
    rendered=Image.open(output)
    assert rendered.size==(672,384)
    for row,candidate in zip(report['candidates'],options):
        assert row['point_model_px']==[candidate['point_original_px'][0]*672/1600,candidate['point_original_px'][1]*384/900]
        assert row['patch_model_px']==[candidate['bbox_original_px'][0]*672/1600,candidate['bbox_original_px'][1]*384/900,
                                       candidate['bbox_original_px'][2]*672/1600,candidate['bbox_original_px'][3]*384/900]
        x1,y1,x2,y2=row['label_model_bbox']
        assert 0<=x1<x2<=672 and 0<=y1<y2<=384
        assert row['font_size']==20
        px,py=row['point_model_px']
        assert rendered.getpixel((min(671,int(px)),min(383,int(py))))==(0,255,255)
    assert 'PIL_pre_resampling' in report['kind']
    assert report['labels_used'] is False and report['instruction_used'] is False
    with pytest.raises(FileExistsError):draw_legible_candidates(source,options,{'width':1600,'height':900},output)


def test_invalid_original_geometry_rejected_without_clipping(tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(100,100)).save(source)
    option={'candidate_id':'R01','point_original_px':[101,20],'bbox_original_px':[10,10,30,30]}
    with pytest.raises(ValueError):draw_legible_candidates(source,[option],{'width':100,'height':100},tmp_path/'out.png')


@pytest.mark.parametrize('args',[(0,100),(100,0),(True,100),(1,1000),(100,100,200,100)])
def test_invalid_resize_parameters_rejected(args):
    with pytest.raises(ValueError):smart_spatial_size(*args)
