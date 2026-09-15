"""Regression checks for reconstructed motion, hierarchy serialization and bind frames."""
from pathlib import Path

import numpy as np
import pygltflib
from scipy.spatial.transform import Rotation

from mimic.processing.rotation_solver import (
    BONE_ORDER, BONE_HIERARCHY, solve_rotations, _compute_bone_direction,
)
from mimic.retargeting.skeleton import create_mediapipe_skeleton
from mimic.retargeting.retarget import retarget
from mimic.export.bvh_writer import write_bvh
from mimic.export.gltf_writer import write_gltf
from mimic.export.merge_animation import merge_animation
from mimic.processing.smoothing import fill_landmark_gaps


def pose():
    lm = np.zeros((33, 3))
    for left, right, x, y in [(23,24,.1,0), (11,12,.2,.5), (13,14,.5,.5),
                              (15,16,.75,.5), (19,20,.85,.5),
                              (25,26,.1,-.43), (27,28,.1,-.85),
                              (7,8,.08,.7)]:
        lm[left] = [x,y,0]
        lm[right] = [-x,y,0]
    lm[0] = [0,.7,.1]
    for heel, toe, ankle in [(29,31,27),(30,32,28)]:
        lm[heel] = lm[ankle] + [0,0,-.05]
        lm[toe] = lm[ankle] + [0,0,.15]
    return lm


def worlds(rotations):
    result = {}
    for b in BONE_ORDER:
        result[b] = result.get(BONE_HIERARCHY[b], Rotation.identity()) * Rotation.from_quat(rotations[b])
    return result


def test_t_pose_has_identity_rotations():
    result = solve_rotations(pose()[None])["rotations"]
    for quats in result.values():
        np.testing.assert_allclose(Rotation.from_quat(quats).magnitude(), 0, atol=1e-7)


def test_rigid_body_turn_does_not_double_parent_rotation():
    turn = Rotation.from_euler("xyz", [20,70,30], degrees=True)
    lm = turn.apply(pose())
    result = solve_rotations(lm[None])["rotations"]
    skel = create_mediapipe_skeleton()
    for b, world in worlds(result).items():
        np.testing.assert_allclose(world.apply(skel.bones[b].rest_direction)[0],
                                   _compute_bone_direction(b,lm), atol=1e-7)
    for b in BONE_ORDER[1:]:
        np.testing.assert_allclose(Rotation.from_quat(result[b]).magnitude(), 0, atol=1e-7)


def test_occluded_limb_holds_last_local_rotation():
    lm = np.stack([pose(), pose()])
    lm[0,13] += [0,-.2,.1]
    lm[1,13] = np.nan
    visibility = np.ones((2,33)); visibility[1,13] = 0
    result = solve_rotations(lm, visibility)["rotations"]
    for b in ["left_upper_arm", "left_lower_arm"]:
        np.testing.assert_allclose(result[b][0],result[b][1])
    assert all(np.isfinite(q).all() for q in result.values())


def test_gap_fill_does_not_pull_pose_to_origin():
    lm = np.ones((5,33,3)); lm[2] = 0
    vis = np.ones((5,33)); vis[2] = 0
    np.testing.assert_allclose(fill_landmark_gaps(lm,vis),1)


def test_bvh_channels_roundtrip_in_declared_order(tmp_path):
    rotations = {b: Rotation.from_euler("XYZ", [i+1, i/2, -i/3], degrees=True).as_quat()[None]
                 for i,b in enumerate(BONE_ORDER)}
    out = tmp_path / "pose.bvh"
    write_bvh(rotations,30,out)
    lines = out.read_text().splitlines()
    declared = [line.split()[1] for line in lines if line.startswith(("ROOT ","JOINT "))]
    values = np.array([float(v) for v in lines[-1].split()])[3:].reshape(-1,3)
    for b, euler in zip(declared,values):
        error = Rotation.from_euler("XYZ",euler,degrees=True).inv() * Rotation.from_quat(rotations[b][0])
        assert error.magnitude() < 1e-7


def read_track(g, channel):
    acc = g.accessors[g.animations[0].samplers[channel.sampler].output]
    bv = g.bufferViews[acc.bufferView]
    return np.frombuffer(g.binary_blob(),dtype="<f4",count=acc.count*4,
                         offset=(bv.byteOffset or 0)+(acc.byteOffset or 0)).reshape(-1,4)


def test_glb_keeps_offsets_and_unanimated_parents(tmp_path):
    out = tmp_path / "pose.glb"
    write_gltf({"mixamorig:LeftArm":np.array([[0,0,0,1.]])},30,out)
    g = pygltflib.GLTF2.load(str(out))
    names = {n.name:i for i,n in enumerate(g.nodes)}
    assert names["mixamorig:LeftArm"] in g.nodes[names["mixamorig:LeftShoulder"]].children
    assert np.linalg.norm(g.nodes[names["mixamorig:LeftArm"]].translation) > 0
    assert len(g.nodes) == len(BONE_ORDER)


def test_merge_preserves_bind_pose_and_transfers_world_delta(tmp_path):
    source = tmp_path / "source.glb"
    model = tmp_path / "model.glb"
    out = tmp_path / "merged.glb"
    tracks = {b:np.tile([0.,0.,0.,1.],(2,1)) for b in BONE_ORDER}
    tracks["left_upper_arm"][1] = Rotation.from_euler("z",40,degrees=True).as_quat()
    write_gltf(retarget(tracks)["rotations"],30,source)
    root_rest = Rotation.from_euler("y",25,degrees=True)
    arm_rest = Rotation.from_euler("z",-90,degrees=True)
    g = pygltflib.GLTF2(nodes=[
        pygltflib.Node(name="mixamorig_Hips",rotation=root_rest.as_quat().tolist(),children=[1]),
        pygltflib.Node(name="mixamorig_LeftArm",rotation=arm_rest.as_quat().tolist()),
    ], skins=[pygltflib.Skin(joints=[0,1])], scenes=[pygltflib.Scene(nodes=[0])],scene=0)
    g.save_binary(str(model))
    merge_animation(model,source,out)
    merged = pygltflib.GLTF2.load(str(out))
    animated = {ch.target.node:Rotation.from_quat(read_track(merged,ch))
                for ch in merged.animations[0].channels}
    np.testing.assert_allclose(animated[1][0].as_matrix(),arm_rest.as_matrix(),atol=1e-6)
    actual = animated[0][1]*animated[1][1]
    expected = Rotation.from_euler("z",40,degrees=True)*root_rest*arm_rest
    np.testing.assert_allclose(actual.as_matrix(),expected.as_matrix(),atol=1e-6)


def test_pipeline_resamples_rotations_and_dispatches_bvh(tmp_path, monkeypatch):
    from mimic import pipeline
    landmarks = np.stack([pose()] * 31)
    solved = solve_rotations(landmarks)
    raw = tmp_path / 'raw.npz'; raw.touch()
    filtered = tmp_path / 'smooth.npz'; filtered.touch()
    rotation_path = tmp_path / 'rotations.npz'
    tracks = {f'rot_{b}':q for b,q in solved['rotations'].items()}
    tracks['rot_pelvis'] = Rotation.from_euler('y',np.linspace(0,90,31)[:, None],degrees=True).as_quat()
    np.savez(rotation_path, num_frames=31, fps=30., **tracks)
    monkeypatch.setattr(pipeline,'do_extract',lambda *args:raw)
    monkeypatch.setattr(pipeline,'do_smooth',lambda *args:filtered)
    monkeypatch.setattr(pipeline,'do_solve',lambda *args:rotation_path)
    output = tmp_path/'result.bvh'
    pipeline.run(tmp_path/'video.mp4',output,fps=15,format='bvh')
    content=output.read_text()
    assert 'Frames: 16' in content
    assert 'Frame Time: 0.066667' in content
    values = np.array([float(v) for v in content.splitlines()[-1].split()])
    actual = Rotation.from_euler('XYZ',values[3:6],degrees=True)
    expected = Rotation.from_euler('y',90,degrees=True)
    assert (actual.inv()*expected).magnitude() < 1e-7


def test_trim_and_smoothing_keep_video_offset(tmp_path, monkeypatch):
    from mimic.extraction import video_to_landmarks
    from mimic.processing.smooth_landmarks import smooth
    from mimic.processing.landmarks_to_rotations import solve
    monkeypatch.setattr(video_to_landmarks,'get_video_info',lambda p:dict(width=10,height=10,fps=30.,total_frames=5,duration=5/30))
    monkeypatch.setattr(video_to_landmarks,'read_frames',lambda p:([None]*5,30.))
    vis=np.ones((5,33));vis[0]=vis[-1]=0
    monkeypatch.setattr(video_to_landmarks,'extract_landmarks',lambda frames,fps:dict(
        world_landmarks=np.stack([pose()]*5),visibility=vis,
        landmarks_2d=np.zeros((5,33,2)),landmark_names=[str(i) for i in range(33)]))
    raw=video_to_landmarks.extract(tmp_path/'video.mp4',tmp_path/'raw.npz')
    filtered=smooth(raw,tmp_path/'smooth.npz')
    solved=solve(filtered,tmp_path/'solved.npz')
    for path in [raw,filtered,solved]:
        with np.load(path) as data:
            assert int(data['first_frame'])==1
    with np.load(solved) as data:
        assert int(data['num_frames'])==3


def test_empty_detection_is_reported(tmp_path, monkeypatch):
    import pytest
    from mimic.extraction import video_to_landmarks
    monkeypatch.setattr(video_to_landmarks,'get_video_info',lambda p:dict(width=10,height=10,fps=30.,total_frames=3,duration=.1))
    monkeypatch.setattr(video_to_landmarks,'read_frames',lambda p:([None]*3,30.))
    monkeypatch.setattr(video_to_landmarks,'extract_landmarks',lambda frames,fps:dict(
        world_landmarks=np.zeros((3,33,3)),visibility=np.zeros((3,33)),
        landmarks_2d=np.zeros((3,33,2)),landmark_names=[]))
    with pytest.raises(ValueError,match='No person detected'):
        video_to_landmarks.extract(tmp_path/'video.mp4',tmp_path/'raw.npz')
    assert not (tmp_path/'raw.npz').exists()
