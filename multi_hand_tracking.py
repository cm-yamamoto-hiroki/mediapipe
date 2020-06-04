import sys
from os import path
import os
import subprocess

import sys
import shutil

from pprint import pprint


def doMultiHandTracking(inputVideoPath):
    dirname = path.dirname(inputVideoPath)
    basename = path.basename(inputVideoPath)
    outputDir = inputVideoPath.replace(".mp4", "-mp4")

    if not path.exists(outputDir):
        os.makedirs(outputDir)

    if os.path.exists(f"{outputDir}/result"):
        shutil.rmtree(f"{outputDir}/result")

    command_options = [
        f'./bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu',
        f'  --calculator_graph_config_file="./mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt"',
        f'  --input_video_path="{inputVideoPath}"',
        f'  --output_video_path="{outputDir}/{basename}"',
        f'> "{outputDir}/result.txt"',
    ]
    pprint(command_options)
    command = " ".join(command_options)
    res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)

    command2 = " ".join([
        f'python3 convertProtobufToJson.py "{outputDir}/result/"',
    ])
    print(command2)
    res = subprocess.run(command2, stderr=subprocess.STDOUT, shell=True)

    shutil.copy2(f"{outputDir}/{basename}", f"{outputDir}/video.mp4")

    os.makedirs(f"{dirname}/handtracked/", exist_ok=True)
    shutil.copy2(f"{outputDir}/{basename}", f"{dirname}/handtracked/{basename}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        inputVideoPaths = sys.argv[1]
    else:
        print("usage: python multi_hand_tracking.py PATH_TO_VIDEOFILE")
        sys.exit(1)

    print(inputVideoPath)

    # command = " ".join([
    #     f'bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu',
    # ])
    # res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)

    doMultiHandTracking(inputVideoPath)

