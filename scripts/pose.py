from enum import Enum

class Pose(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    NONE = -1

HANDPOSE_JUDGE = {
    (False, False, False, False, False): Pose.ZERO,
    (False, True, False, False, False): Pose.ONE,
    (False, True, True, False, False): Pose.TWO,
    (False, True, True, True, False): Pose.THREE,
    (False, True, True, True, True): Pose.FOUR,
    (True, False, False, False, False): Pose.FIVE,
    (True, True, False, False, False): Pose.SIX,
    (True, True, True, False, False): Pose.SEVEN,
    (True, True, True, True, False): Pose.EIGHT,
    (True, True, True, True, True): Pose.NINE,
}

def judge_handpose(fingerIsOpenTuple):
    if not fingerIsOpenTuple in HANDPOSE_JUDGE:
        return Pose.NONE
    else:
        return HANDPOSE_JUDGE[fingerIsOpenTuple]