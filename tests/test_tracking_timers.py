"""Per-bone missing-frame budgets and quaternion gap reconstruction."""
import numpy as np
from scipy.spatial.transform import Rotation
from mimic.processing.rotation_solver import solve_rotations
from mimic.processing.tracking import tracking_window
from test_animation_regressions import pose


def sequence(n):
    return np.stack([pose()]*n), np.ones((n,33))


def test_twenty_missing_frames_allowed_twenty_first_ends_clip():
    lm,vis=sequence(40);vis[5:,19]=0
    result=solve_rotations(lm,vis)
    assert result['num_frames']==25
    assert result['stopped_bones']==['left_hand']
    assert all(len(q)==25 for q in result['rotations'].values())


def test_observation_resets_each_timer():
    lm,vis=sequence(44);vis[1:21,19]=0;vis[22:42,19]=0
    result=solve_rotations(lm,vis)
    assert result['num_frames']==44
    assert result['stopped_bones']==[]


def test_short_gap_interpolates_motion_instead_of_freezing():
    lm,vis=sequence(3)
    lm[2,19]=lm[2,15]+[0,.1,0]
    vis[1,19]=0
    q=solve_rotations(lm,vis)['rotations']['left_hand']
    halfway=Rotation.from_quat(q[1]).apply([1,0,0])
    np.testing.assert_allclose(halfway,[2**-.5,2**-.5,0],atol=1e-6)


def test_first_bone_to_expire_truncates_every_track():
    lm,vis=sequence(50);vis[5:,19]=0;vis[10:,20]=0
    result=solve_rotations(lm,vis,bone_limits={'left_hand':30,'right_hand':3})
    assert result['num_frames']==13
    assert result['stopped_bones']==['right_hand']


def test_observation_beyond_cutoff_is_not_used():
    lm,vis=sequence(30);vis[1:25,19]=0
    lm[25:,19]=lm[25:,15]+[0,.1,0]
    result=solve_rotations(lm,vis)
    assert result['num_frames']==21
    np.testing.assert_allclose(result['rotations']['left_hand'],np.tile([0,0,0,1],(21,1)))


def test_degenerate_geometry_does_not_reset_timer():
    lm,vis=sequence(30);lm[2:,19]=lm[2:,15]
    assert tracking_window(lm,vis)['num_frames']==22


def test_smoothing_cuts_before_interpolating_long_gap(tmp_path):
    from mimic.processing.smooth_landmarks import smooth
    from mimic.processing.landmarks_to_rotations import solve
    lm,vis=sequence(40);vis[5:,19]=0
    source=tmp_path/'raw.npz'
    np.savez(source,world_landmarks=lm,visibility=vis,landmarks_2d=np.zeros((40,33,2)),
             landmark_names=[str(i) for i in range(33)],fps=30.,first_frame=10)
    filtered=smooth(source,tmp_path/'filtered.npz')
    solved=solve(filtered,tmp_path/'rotations.npz')
    with np.load(solved) as data:
        assert data['num_frames']==25
        assert data['input_frames']==40
        assert data['first_frame']==10
        assert list(data['stopped_bones'])==['left_hand']
