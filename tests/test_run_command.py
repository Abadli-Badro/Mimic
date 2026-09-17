"""Single-command orchestration for final animation and optional overlay."""
import json
import numpy as np
import pytest
from typer.testing import CliRunner


@pytest.mark.parametrize('format', ['glb','bvh'])
def test_run_command_uses_one_extraction_and_optional_overlay(tmp_path,monkeypatch,format):
    from interfaces.cli.main import app
    from mimic import pipeline
    from mimic.export import merge_animation
    from mimic.extraction import overlay
    calls=[]
    model=tmp_path/'model.glb';model.touch()
    monkeypatch.setattr(pipeline,'output_file',lambda name:tmp_path/name)
    def extract(video,path):
        calls.append('extract');path.write_bytes(b'raw');return path
    def smooth(source,path,*args):
        assert args[2]==7
        calls.append('smooth');path.write_bytes(b'smooth');return path
    def solve(source,path):
        calls.append('solve')
        np.savez(path,fps=30.,num_frames=3,rot_pelvis=np.tile([0.,0,0,1],(3,1)))
        return path
    def merge(model,source,destination):
        assert source.name=='skeleton.glb' and source.exists()
        calls.append('merge');destination.write_bytes(b'animated');return destination
    def visualize(video,landmarks,destination):
        assert landmarks.name=='landmarks.npz'
        calls.append('overlay');destination.write_bytes(b'overlay');return destination
    monkeypatch.setattr(pipeline,'do_extract',extract)
    monkeypatch.setattr(pipeline,'do_smooth',smooth)
    monkeypatch.setattr(pipeline,'do_solve',solve)
    monkeypatch.setattr(merge_animation,'merge_animation',merge)
    monkeypatch.setattr(overlay,'generate_overlay_video',visualize)
    args=['run','video.mp4','--format',format,'--overlay','--max-missing-frames','7']
    if format=='glb':args+=['--model',str(model)]
    result=CliRunner().invoke(app,args)
    assert result.exit_code==0,result.output
    assert calls==['extract','smooth','solve']+(['merge'] if format=='glb' else [])+['overlay']
    output_name = 'video_animated' if format == 'glb' else 'video'
    report=json.loads((tmp_path/f'{output_name}_status.json').read_text())
    assert report['status']=='complete'
    assert 'overlay' in report
    assert list((tmp_path/'intermediate').iterdir()) == []


def test_run_rejects_model_for_bvh_before_processing():
    from interfaces.cli.main import app
    result=CliRunner().invoke(app,['run','video.mp4','--format','bvh','--model','rig.glb'])
    assert result.exit_code!=0
    assert 'only to GLB' in result.output
