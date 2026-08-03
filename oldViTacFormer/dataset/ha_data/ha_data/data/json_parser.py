import numpy as np
import copy
class JointParser:
  @staticmethod
  def get_hand_torque(closest_hand_state):
    ht = []
    for i in range(21):
        ht.append(closest_hand_state["Payload"]["Joints"][f'Joint_{i}']['Torque'])
    return ht

  @staticmethod
  def get_hand_angle(closest_hand_state):
    ht = []
    for i in range(21):
        ht.append(closest_hand_state["Payload"]["Joints"][f'Joint_{i}']['Angle'])
    return ht

  @staticmethod
  def gen_angles(hand_actions):
      alpha = 0.05
      last_data = JointParser.get_hand_angle(hand_actions[0])
      raw_hand_action_joint_angles=[last_data]
      for i in range(1, len(hand_actions)):
          data = copy.deepcopy(JointParser.get_hand_angle(hand_actions[i]))
          res = (np.array(data) - (1-alpha) * np.array(last_data))/ alpha
          raw_hand_action_joint_angles.append(res.tolist())
          last_data = data
      return raw_hand_action_joint_angles,[a['ReceiveTime'] / 1e6 for a in hand_actions]

  @staticmethod
  def is_peak(raw, i):
      if i == 0 or i > len(raw) - 2:
          return False
      data = raw[i - 1:i + 2]
      for j in range(21):
          diff01 = abs(data[0][j] - data[1][j])
          diff12 = abs(data[1][j] - data[2][j])
          diff02 = abs(data[0][j] - data[2][j])
          if diff01 > 0.1 and diff01 > diff02 * 10 and diff12 > diff02 * 10:
              return True
      return False

  @staticmethod
  def gen_raw_hand_action_joint_angles(hand_actions, stamp):
      raw, ts = JointParser.gen_angles(hand_actions)
      assert len(raw) == len(hand_actions)
      stamp_idx=0
      valid_data = []
      pick=[]
      while ts[0] > stamp[stamp_idx] and stamp_idx < len(stamp):
          valid_data.append(raw[0])
          pick.append(0)
          stamp_idx+=1
      i = 1
      while i < len(hand_actions):
          if ts[i-1] <= stamp[stamp_idx] and ts[i] >= stamp[stamp_idx]:
              if i < len(raw) -2 and JointParser.is_peak(raw, i):
                  i += 1
              valid_data.append(raw[i])
              pick.append(i)
              stamp_idx += 1
              if len(stamp) <= stamp_idx:
                  break
          i += 1
      while len(stamp) > len(valid_data):
          valid_data.append(raw[-1])
      assert len(stamp) == len(valid_data), f'{len(stamp)}, {len(valid_data)}'
      return valid_data
