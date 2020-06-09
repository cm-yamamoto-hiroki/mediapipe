from enum import Enum


class Pose(Enum):
    ZERO = 0
    PINCH = 1
    MONEY = 2
    OK = 3
    NONE = -1


class IpState(Enum):
    CLOSED = 0
    MIDDLE = 1
    OPEN = 2


class McpState(Enum):
    CLOSED = 0
    MIDDLE = 1
    OPEN = 2


class TipState(Enum):
    TOUCHED = 0
    CLOSE = 1
    OPEN = 2


IP_THRESHOD = [50, 120]
MCP_THRESHOD = [30, 50]
TIP_THRESHOLD = [25, 50]

def convertState(row):
    finger_states = []
    for i in range(5):
        if row[f"angle_012_{i}"] < IP_THRESHOD[0]:
            ipState = IpState.OPEN
        elif row[f"angle_012_{i}"] < IP_THRESHOD[1]:
            ipState = IpState.MIDDLE
        else:
            ipState = IpState.CLOSED

        if row[f"angle_0_{i}"] < MCP_THRESHOD[0]:
            mcpState = McpState.OPEN
        elif row[f"angle_0_{i}"] < MCP_THRESHOD[1]:
            mcpState = McpState.MIDDLE
        else:
            mcpState = McpState.CLOSED

        if row[f"tip_distances_{i}"] < TIP_THRESHOLD[0]:
            tipState = TipState.TOUCHED
        elif row[f"tip_distances_{i}"] < TIP_THRESHOLD[1]:
            tipState = TipState.CLOSE
        else:
            tipState = TipState.OPEN

        finger_states.append((ipState, mcpState, tipState))
    
    return finger_states


def judge_handpose(finger_states):
    print(finger_states)
    if all([state[0] == IpState.CLOSED for state in finger_states]):
        return Pose.ZERO
    elif finger_states[1][2] == TipState.TOUCHED:
        if all([finger_states[i][1] == McpState.CLOSED for i in range(2,5)]):
            return Pose.PINCH
        elif all([finger_states[i][1] == McpState.MIDDLE for i in range(2,5)]):
            return Pose.MONEY            
        elif all([finger_states[i][1] == McpState.OPEN for i in range(2,5)]):
            return Pose.OK
        else:
            return Pose.NONE
    else:
        return Pose.NONE


def addHandPose2(df):
    finger_states_list = [
        convertState(row)
        for index, row in df.iterrows()
    ]

    hand_pose = [
        judge_handpose(finger_states)
        for finger_states in finger_states_list
    ]

    df["hand_pose_2"] = hand_pose

    return df
