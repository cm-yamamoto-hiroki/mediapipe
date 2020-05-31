import sys
from os import path
import os
import subprocess

import sys
import shutil

# INPUTVIDEOPATH = "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200515 Multi Hand Tracking/cafe_test_bone/converted/iPhoneXR_overshelf.mp4"
# INPUTVIDEOPATH = "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200515 Multi Hand Tracking/cafe_test_handtracking/WIN_20200515_16_11_51_Pro.mp4"
# INPUTVIDEOPATH = "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200515 Multi Hand Tracking/cafe_test_handtracking/WIN_20200515_16_13_33_Pro.mp4"
INPUTVIDEOPATHs = [
    # "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200527 two people/converted/iPhoneXR_overshelf_scene1.mp4",
    # "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200527 two people/converted/iPhoneXR_overshelf_ss_49.mp4",
    # "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200527 two people/converted/iPhoneXR_overshelf_ss_53.mp4",
    # "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200527 two people/converted/iPhoneXR_overshelf_ss_55.mp4",
    # "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200527 two people/converted/iPhoneXR_overshelf_ss_87.mp4",
    # "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200527 two people/converted/WIN_20200515_16_11_51_Pro_ss_19.mp4",
    # "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200527 two people/converted/WIN_20200515_16_13_33_Pro_ss_24.mp4",
    "/mnt/c/Users/yamamoto.hiroki/Desktop/log/20200515 Multi Hand Tracking/cafe_test_bone/converted/iPhoneXR_overshelf_scene1_short.mp4",
]



def doMultiHandTracking(inputVideoPath):
    dirname = path.dirname(inputVideoPath)
    basename = path.basename(inputVideoPath)
    outputDir = inputVideoPath.replace(".mp4", "-mp4")

    if not path.exists(outputDir):
        os.makedirs(outputDir)

    command = " ".join([
        # f'GLOG_logtostderr=1 \\',
        f'./bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu',
        f'--calculator_graph_config_file="./mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt"',
        f'--input_video_path="{inputVideoPath}"',
        f'--output_video_path="{outputDir}/{basename}"',
        f'> "{outputDir}/result.txt"',
    ])
    res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)

    command2 = " ".join([
        f'python3 convertProtobufToJson.py ./result/{basename}',
    ])
    res = subprocess.run(command2, stderr=subprocess.STDOUT, shell=True)

    shutil.copy2(f"{outputDir}/{basename}", f"{outputDir}/video.mp4")
    if os.path.exists(f"{outputDir}/result"):
        shutil.rmtree(f"{outputDir}/result")
    shutil.copytree(f"./result/{basename}", f"{outputDir}/result")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        inputVideoPaths = [sys.argv[1]]
    else:
        inputVideoPaths = INPUTVIDEOPATHs

    command = " ".join([
        f'bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu',
    ])
    res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)

    for inputVideoPath in inputVideoPaths:
        print(inputVideoPath)
        doMultiHandTracking(inputVideoPath)

