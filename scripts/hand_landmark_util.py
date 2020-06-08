import math
import json
import pandas as pd

# return p0 - p1
def subtract(p0, p1):
    return {
        k: p0[k] - p1[k]
        for k in p1
    }


def inner_product(v0, v1):
    val = sum([v0[k] * v1[k] for k in v0])
    return val


def distance(p0, p1):
    v = subtract(p0, p1)
    distance_square = inner_product(v, v)
    distance = math.sqrt(distance_square)
    return distance


def cos(v0, v1):
    val0 = inner_product(v0, v1)

    origin = {k: 0 for k in v0}  # {"x": 0 , "y":0, "z": 0}
    val1 = distance(v0, origin) * distance(v1, origin)

    return val0 / val1


def angle(v0, v1, rad=False):
    cos_val = cos(v0, v1)
    theta = math.acos(cos_val)

    if rad:
        return theta
    else:
        ang = theta / math.pi * 180.0 
        return ang


def curveness(finger_points):
    distances_between_joints = [
        distance(p0, p1)
        for p0, p1 in zip(finger_points, finger_points[1:])
    ]
    distance_to_tip = distance(finger_points[0], finger_points[-1])

    curveness = distance_to_tip / sum(distances_between_joints)
    return curveness


def bendness(wrist_point, finger_points):
    distance_to_tip = distance(wrist_point, finger_points[3])
    distances_between_joints = [
        distance(wrist_point, finger_points[0]),
        distance(finger_points[0], finger_points[3])
    ]

    bendness = distance_to_tip / sum(distances_between_joints)
    return bendness


FINGER_BONE_INDEXES = [
    (
        (0, i*4 + 1),
        (i*4 + 1, i*4 + 2),
        (i*4 + 2, i*4 + 3),
        (i*4 + 3, i*4 + 4),
    )
    for i in range(5)  # finger
]


def hand_info(landmark):
    angles = [
        [
            angle(
                subtract(landmark[bone0[0]], landmark[bone0[1]]),
                subtract(landmark[bone1[1]], landmark[bone1[0]])
            )
            for bone0, bone1 in zip(finger_bones, finger_bones[1:])
        ]
        for finger_bones in FINGER_BONE_INDEXES
    ]

    vectors = [
        [
            subtract(landmark[bone[0]], landmark[bone[1]])
            for bone in finger_bones
        ]
        for finger_bones in FINGER_BONE_INDEXES
    ]

    # 指先同士の距離
    tip_distances = [
        distance(
            # thumb, {first, second, third, forth}
            landmark[4], landmark[i*4 + 8]
        )
        for i in range(4)
    ]

    # 親指先と各指の根本との距離
    thumb_distances = [
        distance(
            # thumb, {first, second, third, forth}
            landmark[4], landmark[i*4 + 5]
        )
        for i in range(4)
    ]

    # 親指先と各指の根本との距離
    thumb_angle = [
        angle(
            # thumb, {first, second, third, forth}
            subtract(landmark[4], landmark[5]),
            subtract(landmark[4], landmark[17])
        )
    ]

    finger_curvenesses = [
        curveness(
            landmark[i*4+1: (i+1)*4+1]
        )
        for i in range(5)
    ]

    finger_bendnesses = [
        bendness(
            landmark[0],
            landmark[i*4+1: (i+1)*4+1]
        )
        for i in range(5)
    ]

    finger_lengths = [
        sum([
            distance(landmark[i*4+1+j], landmark[i*4+1+j+1])
            for j in range(3)
        ]) if not i == 0 else
        sum([
            distance(landmark[i*4+2+j], landmark[i*4+2+j+1])
            for j in range(2)
        ])
        for i in range(5)
    ]

    val_dict = {
        "angles": angles,
        "vectors": vectors,
        "thumb_angle": thumb_angle,
        "thumb_distances": thumb_distances,
        "tip_distances": tip_distances,
        "finger_curvenesses": finger_curvenesses,
        "finger_bendnesses": finger_bendnesses,
        "finger_lengths": finger_lengths,
    }

    keys = val_dict.keys()
    keys = sorted(keys, key=lambda x: x)
    vals = [val_dict[key] for key in keys]

    val_flatten = sum([sum(vals[0], [])] + vals[1:7] + [sum(vals[7], [])], [])

    series = pd.Series(val_flatten,
                      index=sum([
                          [f"angle_{i}" for i in range(15)],
                          [f"finger_bendnesses_{i}" for i in range(5)],
                          [f"finger_curvenesses_{i}" for i in range(5)],
                          [f"finger_lengths_{i}" for i in range(5)],
                          [f"thumb_angle" for i in range(1)],
                          [f"thumb_distances_{i}" for i in range(4)],
                          [f"tip_distances_{i}" for i in range(4)],
                          [f"vectors{i}" for i in range(20)],
                      ], []),
                      )

    return val_dict, val_flatten, series

def readLandmarkJsonFile(filepath):
    with open(filepath, "r", encoding="utf8") as f:
        # print(filepath)
        content = f.read()
    
    landmark_json = json.loads(content)
    landmark = landmark_json["landmark"]

    return landmark

