import os
import subprocess
import sys
import shutil

from pprint import pprint
from scripts import convertProtobufToJson


def multi_hand_track(input_video_path, input_basename,  output_dir):
    command = " ".join([
        f'/usr/local/bazel/2.0.0/lib/bazel/bin/bazel build -c opt',
        f'  --define MEDIAPIPE_DISABLE_GPU=1',
        f'  mediapipe/examples/desktop/multi_hand_tracking:multi_hand_tracking_cpu',
    ])
    res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)

    if os.path.exists(f"{output_dir}/result"):
        shutil.rmtree(f"{output_dir}/result"    )

    command_options = [
        f'./bazel-bin/mediapipe/examples/desktop/multi_hand_tracking/multi_hand_tracking_cpu',
        f'  --calculator_graph_config_file="./mediapipe/graphs/hand_tracking/multi_hand_tracking_desktop_live.pbtxt"',
        f'  --input_video_path="{input_video_path}"',
        f'  --output_video_path="{output_dir}/{input_basename}"',
        f'> "{output_dir}/result.txt"',
    ]
    pprint(command_options)

    command = " ".join(command_options)
    res = subprocess.run(command, stderr=subprocess.STDOUT, shell=True)


def doMultiHandTracking(input_video_path):
    input_dir = os.path.dirname(input_video_path)
    input_basename = os.path.basename(input_video_path)
    output_dir = input_video_path[:input_video_path.rfind(
        ".")] + "-" + input_video_path[input_video_path.rfind(".")+1:]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    multi_hand_track(input_video_path, input_basename, output_dir)

    print("convert protobuf files")
    convertProtobufToJson.convertFilesInDir(f"{output_dir}/result/")

    if os.path.exists(f"{output_dir}/{input_basename}"):
        shutil.copy2(f"{output_dir}/{input_basename}", f"{output_dir}/video.mp4")
        os.makedirs(f"{input_dir}/handtracked/", exist_ok=True)
        shutil.copy2(f"{output_dir}/{input_basename}",
                    f"{input_dir}/handtracked/{input_basename}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_video_path = sys.argv[1]
    else:
        print("usage: python multi_hand_tracking.py PATH_TO_VIDEOFILE")
        sys.exit(1)

    print(input_video_path)

    doMultiHandTracking(input_video_path)
