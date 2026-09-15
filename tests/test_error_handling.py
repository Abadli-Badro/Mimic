"""Failure-path regression tests: reject bad input before publishing output."""
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pygltflib
import pytest
from typer.testing import CliRunner
from mimic.errors import MimicError
from mimic.config import VideoLimits
from mimic.artifacts import atomic_output, load_npz
from mimic.validation import hierarchy, rotations, landmarks
from mimic.extraction.quality import validate_pose_quality
from mimic.extraction.video_reader import get_video_info, read_frames
from mimic.export.gltf_writer import write_gltf
from mimic.export.bvh_writer import write_bvh
from mimic.export import gltf_validation
from test_animation_regressions import pose


@pytest.mark.parametrize('parents', [
    {}, {'root':'root'}, {'root':None,'child':'missing'},
    {'a':None,'b':None}, {'a':'b','b':'a'}, {'root':None,'a':'b','b':'a'},
])
def test_invalid_hierarchies(parents):
    with pytest.raises(MimicError, match='invalid_hierarchy'):
        hierarchy(parents)


@pytest.mark.parametrize('track', [np.zeros((2,4)), np.full((2,4),np.nan),
                                  np.full((2,4),np.inf), np.ones((2,3)), np.zeros((0,4))])
def test_invalid_tracks_do_not_replace_existing_output(tmp_path, track):
    path=tmp_path/'test.bvh';path.write_text('previous valid result')
    with pytest.raises(MimicError):
        write_bvh({'pelvis':track},30,path)
    assert path.read_text()=='previous valid result'


def test_unknown_bone_rejected(tmp_path):
    with pytest.raises(MimicError,match='unknown_bones'):
        write_bvh({'tail':np.array([[0,0,0,1.]])},30,tmp_path/'test.bvh')
    assert not (tmp_path/'test.bvh').exists()


def test_track_lengths_must_match():
    with pytest.raises(MimicError):
        rotations({'root':np.tile([0.,0,0,1],(3,1)), 'child':np.tile([0.,0,0,1],(2,1))})


def test_anatomically_wrong_parent_rejected(tmp_path):
    from mimic.retargeting.retarget import MIXAMO_HIERARCHY
    parents=dict(MIXAMO_HIERARCHY);parents['mixamorig:Head']='mixamorig:LeftFoot'
    with pytest.raises(MimicError,match='incompatible_skeleton'):
        write_gltf({'mixamorig:Hips':np.array([[0.,0,0,1]])},30,tmp_path/'a.glb',hierarchy=parents)


def test_atomic_write_failure_keeps_previous_file(tmp_path):
    path=tmp_path/'animation.glb';path.write_bytes(b'old')
    with pytest.raises(OSError):
        with atomic_output(path) as temporary:
            temporary.write_bytes(b'partial')
            raise OSError('disk full')
    assert path.read_bytes()==b'old'
    assert list(tmp_path.iterdir())==[path]


def test_pickle_npz_rejected(tmp_path):
    path=tmp_path/'bad.npz';np.savez(path,payload=np.array([{}],dtype=object))
    with pytest.raises(MimicError,match='invalid_archive'):
        load_npz(path)


def test_corrupt_npz_rejected(tmp_path):
    path=tmp_path/'bad.npz';path.write_bytes(b'not a zip')
    with pytest.raises(MimicError,match='invalid_archive'):
        load_npz(path)


def test_missing_landmark_fields():
    with pytest.raises(MimicError,match='missing_fields'):
        landmarks({'fps':30.})


@pytest.mark.parametrize('kind,code',[('no_pose','no_human_pose'),('collapsed','poor_pose_quality'),
                                     ('short','insufficient_motion'),('gap','tracking_lost')])
def test_pose_quality_failures(kind,code):
    world=np.stack([pose()]*120);vis=np.ones((120,33))
    if kind=='no_pose':vis[:]=0
    if kind=='collapsed':world[:]=0
    if kind=='short':world=world[:2];vis=vis[:2]
    if kind=='gap':vis[40:80]=0
    with pytest.raises(MimicError) as error:
        validate_pose_quality(world,vis,30)
    assert error.value.code==code


def test_empty_edges_allowed_and_preserved():
    world=np.stack([pose()]*30);vis=np.ones((30,33));vis[:10]=0;vis[20:]=0
    assert validate_pose_quality(world,vis,30)==(10,19)


def test_video_size_checked_before_decoder(tmp_path,monkeypatch):
    import mimic.extraction.video_reader as reader
    path=tmp_path/'large.mp4';path.write_bytes(b'12345')
    monkeypatch.setattr(reader.cv2,'VideoCapture',lambda p:pytest.fail('decoder should not run'))
    with pytest.raises(MimicError,match='video_too_large'):
        get_video_info(path,replace(VideoLimits(),max_file_bytes=4))


class FakeCapture:
    def __init__(self, metadata, frames=()):
        self.metadata=metadata;self.frames=iter(frames);self.released=False
    def isOpened(self):return True
    def get(self,key):return self.metadata[key]
    def release(self):self.released=True
    def read(self):
        frame=next(self.frames,None)
        return (frame is not None,frame)


def fake_video(tmp_path,monkeypatch,**overrides):
    import mimic.extraction.video_reader as reader
    cv=reader.cv2
    metadata={cv.CAP_PROP_FPS:30.,cv.CAP_PROP_FRAME_WIDTH:32.,cv.CAP_PROP_FRAME_HEIGHT:32.,cv.CAP_PROP_FRAME_COUNT:10.}
    for key,value in overrides.items():metadata[getattr(cv,key)]=value
    path=tmp_path/'video.mp4';path.write_bytes(b'fake')
    captures=[]
    def create(p):
        cap=FakeCapture(metadata,[np.zeros((32,32,3),dtype=np.uint8)]*2);captures.append(cap);return cap
    monkeypatch.setattr(cv,'VideoCapture',create)
    return path,captures


@pytest.mark.parametrize('overrides', [{'CAP_PROP_FPS':0.},{'CAP_PROP_FPS':float('nan')},
    {'CAP_PROP_FRAME_WIDTH':0.},{'CAP_PROP_FRAME_COUNT':0.},{'CAP_PROP_FRAME_WIDTH':100000.},
    {'CAP_PROP_FRAME_COUNT':4000.}])
def test_bad_video_metadata_releases_decoder(tmp_path,monkeypatch,overrides):
    path,caps=fake_video(tmp_path,monkeypatch,**overrides)
    with pytest.raises(MimicError):get_video_info(path)
    assert all(c.released for c in caps)


def test_truncated_decode_rejected_and_released(tmp_path,monkeypatch):
    path,caps=fake_video(tmp_path,monkeypatch)
    with pytest.raises(MimicError,match='truncated_video'):read_frames(path)
    assert all(c.released for c in caps)


def test_decoded_memory_limit(tmp_path,monkeypatch):
    path,caps=fake_video(tmp_path,monkeypatch)
    with pytest.raises(MimicError,match='video_memory_limit'):
        read_frames(path,limits=replace(VideoLimits(),max_decoded_bytes=10))
    assert all(c.released for c in caps)


def test_multiple_people_rejected_instead_of_selecting_first(tmp_path,monkeypatch):
    from mimic.extraction import pose_extractor as module
    model=tmp_path/'model.task';model.write_bytes(b'fake')
    class Detector:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def detect_for_video(self,*args):
            return SimpleNamespace(pose_world_landmarks=[[],[]],pose_landmarks=[[],[]])
    def create(options):
        assert options.num_poses==2
        return Detector()
    monkeypatch.setattr(module.vision.PoseLandmarker,'create_from_options',create)
    with pytest.raises(MimicError,match='multiple_people'):
        module.extract_landmarks([np.zeros((32,32,3),dtype=np.uint8)],model,30.)


def test_gltf_cycle_and_multiple_parents():
    for nodes in [[pygltflib.Node(children=[1]),pygltflib.Node(children=[0])],
                  [pygltflib.Node(children=[2]),pygltflib.Node(children=[2]),pygltflib.Node()]]:
        with pytest.raises(MimicError,match='invalid_hierarchy'):
            gltf_validation.graph(pygltflib.GLTF2(nodes=nodes))


def test_bad_glb_container(tmp_path):
    path=tmp_path/'bad.glb';path.write_bytes(b'invalid')
    with pytest.raises(MimicError,match='invalid_glb'):
        gltf_validation.load_glb(path)


def test_bad_gltf_buffer_bounds():
    g=pygltflib.GLTF2(buffers=[pygltflib.Buffer(byteLength=4)],
                       bufferViews=[pygltflib.BufferView(buffer=0,byteLength=16)])
    g.set_binary_blob(b'1234')
    with pytest.raises(MimicError,match='invalid_buffer_view'):gltf_validation.buffers(g)


def test_bad_timeline(tmp_path):
    path=tmp_path/'animation.glb'
    write_gltf({'mixamorig:Hips':np.tile([0.,0,0,1],(3,1))},30,path)
    g=pygltflib.GLTF2.load(str(path));blob=bytearray(g.binary_blob())
    blob[:12]=np.array([0,.2,.1],dtype='<f4').tobytes();g.set_binary_blob(bytes(blob))
    with pytest.raises(MimicError,match='invalid_timeline'):gltf_validation.animation(g)


def test_cli_expected_failure_has_no_traceback(tmp_path):
    from interfaces.cli.main import app
    result=CliRunner().invoke(app,['convert',str(tmp_path/'missing.mp4')])
    assert result.exit_code==1
    assert 'Error:' in result.output and 'extraction' in result.output
    assert 'Traceback' not in result.output


def test_failed_run_reports_failure_without_replacing_export(tmp_path,monkeypatch):
    import json
    from mimic import pipeline
    monkeypatch.setattr(pipeline,'output_file',lambda name:tmp_path/name)
    def fail(*args):raise MimicError('no_human_pose','No person detected', 'extraction')
    monkeypatch.setattr(pipeline,'do_extract',fail)
    output=tmp_path/'existing.glb';output.write_bytes(b'old result')
    with pytest.raises(MimicError):pipeline.run(tmp_path/'video.mp4',output)
    report=json.loads((tmp_path/'intermediate/video/status.json').read_text())
    assert report['status']=='failed' and report['code']=='no_human_pose'
    assert output.read_bytes()==b'old result'
