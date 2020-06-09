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
import pose
import pose2

import cv2


def getLandmarkRawFiles(dirpath):
    filepaths = glob.glob(f"{dirpath}/result/*landmarkRaw*.json")

    filepaths = sorted(filepaths,
                       key=lambda x: int(
                           re.findall("iLoop=(\\d+)", x)[0])
                       )

    vals = [
        hand_landmark_util.hand_info(
            hand_landmark_util.readLandmarkJsonFile(filepath))
        for filepath in filepaths
    ]

    return vals


def getOutputImagesFile(dirpath):
    filepaths = glob.glob(f"{dirpath}/result/*outputFrame*.jpg")

    filepaths = sorted(filepaths,
                       key=lambda x: int(
                           re.findall("iLoop=(\\d+)", x)[0])
                       )

    frames = [
        cv2.imread(filepath)
        for filepath in filepaths
    ]

    return frames


def calcFeatures(vals):
    # states = [val[0] for val in vals]
    # flatten_states = [val[1] for val in vals]
    serieses = [val[2] for val in vals]

    df = pd.DataFrame(serieses)

    for i in range(5):
        df[f"angle_0_{i}"] = 180 - df[f"angle_{i*3+0}"]
        df[f"angle_1_{i}"] = 180 - df[f"angle_{i*3+1}"]
        df[f"angle_2_{i}"] = 180 - df[f"angle_{i*3+2}"]
        df[f"angle_12_{i}"] = 360 - \
            (df[f"angle_{i*3+1}"] + df[f"angle_{i*3+2}"])
        df[f"angle_012_{i}"] = 540 - (df[f"angle_{i*3}"] +
                                      df[f"angle_{i*3+1}"] + df[f"angle_{i*3+2}"])
    df[f"thumb_dist_sum"] = sum([df[f"thumb_distances_{i}"] for i in range(4)])
    for i in range(3):
        df[f"thumb_angle_{i}"] = 180 - df[f"angle_{i}"]
    df[f"thumb_angle_{3}"] = (180 - df[f"angle_{0}"]) + \
        (180 - df[f"angle_{1}"])
    df[f"thumb_angle_{4}"] = (180 - df[f"angle_{1}"]) + \
        (180 - df[f"angle_{2}"])

    for i in range(5):
        if i == 0:
            threshold = 70
        else:
            threshold = 100

        df[f"finger_isOpen_{i}"] = df[f"angle_012_{i}"] < threshold

    isOpen = [
        tuple(row[f"finger_isOpen_{i}"] for i in range(5))
        for index, row in df.iterrows()
    ]
    pprint(isOpen)

    hand_pose = [
        pose.judge_handpose(isOpenStates)
        for isOpenStates in isOpen
    ]
    pprint(hand_pose)
    df["hand_pose"] = hand_pose

    return df


def plotGraph(df, dirpath):
    cargs = {'kind': 'line', 'use_index': True, 'ylim': (0, None), 'rot': 45}

    # graph settings
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(20, 10), sharex=True)
    args_list = [
        # {'title': 'angles', 'column': [f"angle_{i}" for i in range(15)]},
        {'title': 'angles1', 'column': [
            f"angle_1_{i}" if not i == 0 else f"angle_2_{i}" for i in range(5)]},
        {'title': 'angles2', 'column': [
            f"angle_2_{i}" if not i == 0 else f"angle_2_{i}" for i in range(5)]},
        {'title': 'angles12', 'column': [
            f"angle_12_{i}" if not i == 0 else f"angle_2_{i}" for i in range(5)]},
        {'title': 'angles0', 'column': [
            f"angle_0_{i}" if not i == 0 else f"angle_1_{i}" for i in range(5)]},
        # {'title': 'angles123', 'column': [f"angle_123_{i}" for i in range(5)]},
        # {'title': 'thumb_dist', 'column': [
        # f"thumb_distances_{i}" for i in range(5)]},
        # {'title': 'thumb_dist_sum', 'column': ["thumb_dist_sum"]},
        # {'title': 'thumb_angle', 'column': ["thumb_angle"]},
        # {'title': 'bend', 'column': [
        #     f"finger_bendnesses_{i}" for i in range(5)]},
        # {'title': 'curve', 'column': [
        #     f"finger_curvenesses_{i}" for i in range(5)]},
        {'title': 'tip_dist', 'column': [
            f"tip_distances_{i}" for i in range(5)]},
        {'title': 'thumb_angle', 'column': [
            f"thumb_angle_{i}" for i in range(5)]},
    ]

    for ax, args in zip(axes.ravel(), args_list):
        df[args['column']].plot(ax=ax,  **cargs, title=args['title'])

    plt.savefig(f"{dirpath}/hand_states.png")

    p = pathlib.Path(f'{dirpath}')
    dirname = dirpath.split("/")[-1].split("\\")[-1]
    shutil.copy2(f"{dirpath}/hand_states.png",
                 f"{p.parent}/handtracked/{dirname}.png")


def writeVideoWithHandPose(df, dirpath):
    frames = getOutputImagesFile(dirpath)

    output_filepath = f"{dirpath}/hand_pose.wmv".replace("\\", "/")
    fourcc = cv2.VideoWriter_fourcc('W', 'M', 'V', '2')
    writer = cv2.VideoWriter(output_filepath, fourcc,
                             30.0, (frames[0].shape[1], frames[0].shape[0]))

    for i, (frame, pose) in enumerate(zip(frames, df["hand_pose"])):
        cv2.putText(frame, str(pose), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        writer.write(frame)

    writer.release()

    p = pathlib.Path(f'{dirpath}')
    dirname = dirpath.split("/")[-1].split("\\")[-1]
    shutil.copy2(f"{output_filepath}",
                 f"{p.parent}/handtracked/{dirname}_hand_pose.wmv")


def writeVideoWithHandPose2(df, dirpath):
    frames = getOutputImagesFile(dirpath)

    output_filepath = f"{dirpath}/hand_pose_2.wmv".replace("\\", "/")
    fourcc = cv2.VideoWriter_fourcc('W', 'M', 'V', '2')
    writer = cv2.VideoWriter(output_filepath, fourcc,
                             30.0, (frames[0].shape[1], frames[0].shape[0]))

    for i, (frame, pose) in enumerate(zip(frames, df["hand_pose_2"])):
        cv2.putText(frame, str(pose), (100, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        writer.write(frame)

    writer.release()

    p = pathlib.Path(f'{dirpath}')
    dirname = dirpath.split("/")[-1].split("\\")[-1]
    shutil.copy2(f"{output_filepath}",
                 f"{p.parent}/handtracked/{dirname}_hand_pose_2.wmv")


def recognizeHandPoseDir(dirpath):
    vals = getLandmarkRawFiles(dirpath)
    df = calcFeatures(vals)
    df.to_csv(f"{dirpath}/hand_states.csv")
    plotGraph(df, dirpath)
    writeVideoWithHandPose(df, dirpath)
    pose2.addHandPose2(df)
    writeVideoWithHandPose2(df, dirpath)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print(
            "usage: python hand_pose.py PATH_TO_LANDMARK_FILE or PATH_TO_LANDMARK_FILES_DIR")
        exit(1)

    print(path)

    if os.path.isfile(path):
        recognizeHandPoseFile(path)
    elif os.path.isdir(path):
        recognizeHandPoseDir(path)
    else:
        print(
            "usage: python hand_pose.py PATH_TO_LANDMARK_FILE or PATH_TO_LANDMARK_FILES_DIR")
