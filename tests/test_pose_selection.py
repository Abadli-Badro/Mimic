"""Distinguish duplicate proposals from persistent second people."""
from types import SimpleNamespace
import pytest
from mimic.errors import MimicError
from mimic.extraction.pose_selection import PoseSelector


def person(x=.4, offset=0, confidence=1):
    points=[SimpleNamespace(x=x+offset,y=.4+offset,visibility=confidence,presence=confidence) for _ in range(33)]
    points[11].x-=.05;points[12].x+=.05
    points[23].x-=.04;points[24].x+=.04
    points[23].y=points[24].y=.7+offset
    return points


def test_duplicate_high_confidence_detections_are_one_person():
    selector=PoseSelector(25)
    for frame in range(30):
        assert selector.select([person(),person(offset=.01)],frame,frame*40) in (0,1)
    assert selector.multiple_frames==0


def test_weak_second_detection_ignored():
    selector=PoseSelector(25)
    for frame in range(30):
        assert selector.select([person(),person(.8,confidence=.2)],frame,frame*40)==0


def test_transient_second_pose_does_not_abort():
    selector=PoseSelector(25)
    for frame in range(7):selector.select([person(),person(.8)],frame,frame*40)
    selector.select([person()],7,280)
    assert selector.multiple_frames==0


def test_persistent_distinct_people_abort():
    selector=PoseSelector(25)
    for frame in range(7):selector.select([person(),person(.8)],frame,frame*40)
    with pytest.raises(MimicError,match='multiple_people'):
        selector.select([person(),person(.8)],7,280)


def test_primary_identity_survives_proposal_order_change():
    selector=PoseSelector(25)
    assert selector.select([person(.3)],0,0)==0
    assert selector.select([person(.8),person(.3)],1,40)==1
