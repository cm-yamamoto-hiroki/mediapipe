from mediapipe.framework.formats.detection_pb2 import Detection
from mediapipe.framework.formats.landmark_pb2 import LandmarkList
from mediapipe.framework.formats.rect_pb2 import NormalizedRect
from google.protobuf.json_format import MessageToJson

import glob
import re
from pprint import pprint
import json
import sys


TYPE_INFOS = [
    {
        "filepath_frag": "detection",
        "data_class": Detection,
    },
    {
        "filepath_frag": "landmark",
        "data_class": LandmarkList,
    },
    {
        "filepath_frag": "handRect",
        "data_class": NormalizedRect,
    },
    {
        "filepath_frag": "palmRect",
        "data_class": NormalizedRect,
    },
    {
        "filepath_frag": "handRectFromLandmarks",
        "data_class": NormalizedRect,
    },
    {
        "filepath_frag": "landmarkRaw",
        "data_class": LandmarkList,
    },
]


def convertFilesOfType(target_dir, type_info):
    filepath_frag = type_info["filepath_frag"]
    DataClass = type_info["data_class"]

    output_files = glob.glob(target_dir + "/" + "*.txt")

    target_files = [(re.findall(
        f"iLoop=(\\d+)_{filepath_frag}_j=(\\d+).txt", output_file), output_file) for output_file in output_files]
    target_files_filtered = [
        (target_file[0], target_file[1].replace("\\", "/")) for target_file in target_files if target_file[0]]

    for target_file in target_files_filtered:
        with open(target_file[1], "rb") as f:
            content = f.read()

        data = DataClass()
        data.ParseFromString(content)
        json_obj = MessageToJson(data)

        target_file_output = target_file[1].replace("txt", "json")
        with open(target_file_output, "w") as f:
            f.write(json_obj)     


def convertFilesInDir(target_dir):
    for type_info in TYPE_INFOS:
        convertFilesOfType(target_dir, type_info)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        print("usage: python convertProtobufToJson.py PATH_TO_FOLDER")

    convertFilesInDir(target_dir)
