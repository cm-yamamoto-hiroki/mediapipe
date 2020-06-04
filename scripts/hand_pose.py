import os
import sys
import glob 
import re
import shutil
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint

import hand_landmark_util


def flattenHandState(hand_info):
    keys = hand_info.keys()
    keys = sorted(keys, key=lambda x: x)
    vals = [hand_info[key] for key in keys]

    vals_flatten = sum([sum(vals[0], [])] + vals[1:], [])
    
    # print(vals_flatten)
    return vals_flatten


def recognizeHandState(landmark):
    hand_info = hand_landmark_util.hand_info(landmark)

    return hand_info


def recognizeHandPoseFile(filepath):
    landmark = hand_landmark_util.readLandmarkJsonFile(filepath)
    state = recognizeHandState(landmark)
    # pprint(state)

    return state


def recognizeHandPoseDir(dirpath):
    landmark_files = glob.glob(f"{dirpath}/result/*landmarkRaw*.json")

    landmark_files = list(sorted(landmark_files, 
        key=lambda x : int(re.findall("iLoop=(\\d+)", x)[0])
    ))
    states = [
        recognizeHandPoseFile(landmark_file)
        for landmark_file in landmark_files
    ]
    pprint(states[0])
    flatten_states = [
        flattenHandState(state)
        for state in states
    ]
    
    df = pd.DataFrame(flatten_states, 
        columns=sum([
            [f"angle_{i}" for i in range(15)],
            [f"finger_bendnesses{i}" for i in range(5)],
            [f"finger_curvenesses{i}" for i in range(5)],
            [f"finger_lengths{i}" for i in range(5)],
            [f"tip_distances_{i}" for i in range(4)],
        ], []),
    )
    print(df)
    df.to_csv(f"{dirpath}/hand_states.csv")

    for i in range(5):
        df[f"angle_23_{i}"] = df[f"angle_{i*3+1}"] + df[f"angle_{i*3+2}"]

    cargs = {'kind':'line', 'use_index':True, 'ylim': (0, None), 'rot':45}
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(30, 15), sharex=True)

    # graph settings
    args_list = [
        # {'title': 'angles', 'column': [f"angle_{i}" for i in range(15)]},
        {'title': 'angles1', 'column': [f"angle_{3*i+0}" for i in range(5)]},
        {'title': 'angles23', 'column': [f"angle_23_{i}" for i in range(5)]},
        {'title': 'tip_dist', 'column': [f"tip_distances_{i}" for i in range(4)]},
        {'title': 'bend', 'column': [f"finger_bendnesses{i}" for i in range(5)]},
        {'title': 'curve', 'column': [f"finger_curvenesses{i}" for i in range(5)]},
    ]

    for ax, args in zip(axes.ravel(), args_list):
        df[args['column']].plot(ax=ax,  **cargs, title=args['title'])

    plt.savefig(f"{dirpath}/hand_states.png")

    p = pathlib.Path(f'{dirpath}')
    dirname = dirpath.split("/")[-1].split("\\")[-1]
    shutil.copy2(f"{dirpath}/hand_states.png", f"{p.parent}/handtracked/{dirname}.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print("usage: python hand_pose.py PATH_TO_LANDMARK_FILE or PATH_TO_LANDMARK_FILES_DIR")
        exit(1)

    print(path)

    if os.path.isfile(path):
        recognizeHandPoseFile(path)
    elif os.path.isdir(path):
        recognizeHandPoseDir(path)
    else:
        print("usage: python hand_pose.py PATH_TO_LANDMARK_FILE or PATH_TO_LANDMARK_FILES_DIR")


