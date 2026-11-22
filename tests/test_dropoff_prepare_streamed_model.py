"""Preparation boundaries; no network, real weights or GPU operations."""
import sys
from types import SimpleNamespace

import pytest

from tools import dropoff_prepare_streamed_model as preparation


def fake_processor(monkeypatch,calls):
    class Processor:
        @classmethod
        def from_pretrained(cls,*args,**kwargs):
            calls.append((args,kwargs));return cls()
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoProcessor=Processor))


def test_preparation_downloads_quantized_weights_but_only_original_processor(monkeypatch,tmp_path):
    downloads=[];processors=[]
    monkeypatch.setattr(preparation.shutil,'disk_usage',lambda _:SimpleNamespace(free=20*1024**3))
    monkeypatch.setattr(preparation,'checkpoint_identity',lambda *a,**k:{'verified_weights':k['verify_weights']})
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=lambda *a,**k:downloads.append((a,k))))
    fake_processor(monkeypatch,processors)
    monkeypatch.setenv('HF_HUB_DISABLE_XET','1')
    result=preparation.prepare(tmp_path/'model',tmp_path/'processor')
    assert len(downloads)==2
    assert downloads[0][0][0]==preparation.STREAMED_MODEL_ID
    assert downloads[0][1]['revision']==preparation.STREAMED_REVISION
    assert '*.safetensors' in downloads[0][1]['allow_patterns']
    assert downloads[1][0][0]==preparation.BASE_MODEL_ID
    assert downloads[1][1]['revision']==preparation.BASE_REVISION
    assert not any('safetensors' in pattern or 'bin' in pattern for pattern in downloads[1][1]['allow_patterns'])
    assert processors[0][1]['local_files_only'] is True
    assert result['checkpoint']['verified_weights'] and result['model_training_executed'] is False


def test_offline_preparation_never_calls_download(monkeypatch,tmp_path):
    def forbidden(*a,**k):raise AssertionError('Unexpected download')
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=forbidden))
    monkeypatch.setattr(preparation,'checkpoint_identity',lambda *a,**k:{'verified_weights':True})
    calls=[];fake_processor(monkeypatch,calls)
    result=preparation.prepare(tmp_path/'model',tmp_path/'processor',offline=True)
    assert result['offline'] and calls[0][1]['local_files_only']


def test_low_disk_stops_before_requesting_model(monkeypatch,tmp_path):
    monkeypatch.setattr(preparation.shutil,'disk_usage',lambda _:SimpleNamespace(free=100*1024**2))
    def forbidden(*a,**k):raise AssertionError('Unexpected download')
    monkeypatch.setitem(sys.modules,'huggingface_hub',SimpleNamespace(snapshot_download=forbidden))
    with pytest.raises(RuntimeError,match='disk headroom'):
        preparation.prepare(tmp_path/'model',tmp_path/'processor')


def test_offline_rejects_bad_checkpoint_before_processor(monkeypatch,tmp_path):
    def bad(*a,**k):raise ValueError('Pinned bytes differ')
    monkeypatch.setattr(preparation,'checkpoint_identity',bad)
    calls=[];fake_processor(monkeypatch,calls)
    with pytest.raises(ValueError,match='Pinned bytes differ'):
        preparation.prepare(tmp_path/'model',tmp_path/'processor',offline=True)
    assert calls==[]
