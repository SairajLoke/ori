RED   = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"

def redstr(s):
    return RED + s + RESET
def greenstr(s):
    return GREEN + s + RESET


class NamingRule2410:
  def __init__(self):
      self.VERSION='0.1.1'
      self.name = 'NamingRule2410'
      self._gen_path()
      self._gen_h5_naming()
      self._gen_version()

  def _gen_h5_naming(self):
      self.IMAGE_BASELINE = "/observe/vision/head/rgbd/rgb"
      self.IMAGE_PREFIX = "/observe/vision/"
      self.CTRL_PREFIX = "/action/"
      self.ROBOT_PREFIX = "/state/"
      self.TIME_SUFFIX = "/time"
      self.ANNO_PREFIX = "/anno/"

  def _gen_path(self):
      self.ANNO_FOLDER = "anno/"
      self.DATA_FOLDER = "data/"

  def _gen_version(self):
      self.LUBAN_VERSION = "Luban2410"
      self.ANNO_VERSION = f"harobotDE_2_0_0"
      self.DATA_VERSION = f"Train_0_2_0"
      self.RAW_VERSION= f"RAW_0_2_0"


class NamingRule2411(NamingRule2410):
    def __init__(self):
        self.VERSION='0.2.0'
        self.name = 'NamingRule2411'
        self._gen_path()
        self._gen_h5_naming()
        self.ALIGN_SUFFIX= "aligned_index"
        self._gen_version()

class NamingRule2412(NamingRule2411):
    def __init__(self):
        self.VERSION='0.2.2'
        self._gen_path()
        self._gen_h5_naming()
        self.TACTILE_PREFIX = "/observe/tactile/"
        self.name = 'NamingRule2412'
        self.TACTILE_MAP = {
            '05':'right_little',
            '06':'right_ring',
            '07':'right_middle',
            '08':'right_index',
            '09':'right_thumb',
            '15':'left_little',
            '16':'left_ring',
            '17':'left_middle',
            '18':'left_index',
            '19':'left_thumb'
        }
        self.ALIGN_SUFFIX= "aligned_index"
        self.VALID_SUFFIX= '/valid'
        self.IMAGE_BASELINE = "/observe/vision/chest/rgbd/rgb"
        self._gen_version()

    def _gen_version(self):
        self.LUBAN_VERSION = "Luban2412"
        self.ANNO_VERSION = f"anno_{self.VERSION}"
        self.DATA_VERSION = f"train_{self.VERSION}"
        self.RAW_VERSION= f"raw_{self.VERSION}"

    def tactile_force_name(self, name):
        return f'p_100{name}_force6.json'

class NamingRule2503(NamingRule2412):
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        self.VERSION = '0.3.0'
        self.name = 'NamingRule2503'
        self._gen_version()

    def _gen_version(self):
        self.LUBAN_VERSION = "Luban2503"
        self.ANNO_VERSION = f"anno_{self.VERSION}"
        self.DATA_VERSION = f"train_{self.VERSION}"
        self.RAW_VERSION = f"raw_{self.VERSION}"

def deduce_naming(tag):
    namings = [NamingRule2410(), NamingRule2411(), NamingRule2412(), NamingRule2503()]
    if not isinstance(tag, str) and isinstance(tag, object):
        return tag
    def guess_simple(tag):
        if '.' in tag:
            if '_' in tag:
                tag = tag.split('_')[-1]
            for item in namings:
                if tag == item.VERSION:
                    return item
        return None
    def guess_version_name(tag_ori):
        tag = str(tag_ori)
        if '@' in tag:
          tag = tag.split('@')[0]
        first_letter = tag.split('_')[0]
        if len(first_letter) > 1:
            tag = tag[len(first_letter)+1:]
        for item in namings:
            if tag.replace('_', '.') == item.VERSION:
                return item
        return None
    naming = guess_simple(tag)
    if naming is None:
      naming = guess_version_name(tag)
    assert naming is not None, tag
    print(f'deduce naming from {redstr(tag)}, get version {greenstr(naming.VERSION)}')
    return naming


def replace_last(sentence, word, sep='/'):
  return sep.join(sentence.rsplit("/", 1)[:-1]) + word

# initial raw format before 2410

COLLECTION_NAME_CONVERTER = {
    # Image
    "Front_OrbbecCamera_Depth": "chest/rgbd/depth",
    "Front_OrbbecCamera_Color": "chest/rgbd/rgb",

    "Head_OrbbecCamera_Color": "head/rgbd/rgb",
    "Head_OrbbecCamera_Depth": "head/rgbd/depth",

    "Left_Stereo_Color": "head/stereo/lefteye/rgb",
    "Right_Stereo_Color": "head/stereo/righteye/rgb",

    "Right_Realsense_Color": "right_wrist/rgbd/rgb",
    "Right_Realsense_Depth": "right_wrist/rgbd/depth",
    "Right_Fish_Color": "right_wrist/fisheye/rgb",
    "Left_Fish_Color": "left_wrist/fisheye/rgb",

    # Tactile
    # left hand
    "Right_10005_Color": "right_little/stereo/righteye/rgb",
    "Left_10005_Color": "right_little/stereo/lefteye/rgb",
    "Deform_10005_Color": "right_little/deform",
    "Deform_10005_PointCloud": "right_little/ply",
    "Right_10006_Color": "right_ring/stereo/righteye/rgb",
    "Left_10006_Color": "right_ring/stereo/lefteye/rgb",
    "Deform_10006_Color": "right_ring/deform",
    "Deform_10006_PointCloud": "right_ring/ply",
    "Right_10007_Color": "right_middle/stereo/righteye/rgb",
    "Left_10007_Color": "right_middle/stereo/lefteye/rgb",
    "Deform_10007_Color": "right_middle/deform",
    "Deform_10007_PointCloud": "right_middle/ply",
    "Right_10008_Color": "right_index/stereo/righteye/rgb",
    "Left_10008_Color": "right_index/stereo/lefteye/rgb",
    "Deform_10008_Color": "right_index/deform",
    "Deform_10008_PointCloud": "right_index/ply",
    "Right_10009_Color": "right_thumb/stereo/righteye/rgb",
    "Left_10009_Color": "right_thumb/stereo/lefteye/rgb",
    "Deform_10009_Color": "right_thumb/deform",
    "Deform_10009_PointCloud": "right_thumb/ply",
    # right hand
    "Right_10015_Color": "left_little/stereo/righteye/rgb",
    "Left_10015_Color": "left_little/stereo/lefteye/rgb",
    "Deform_10015_Color": "left_little/deform",
    "Deform_10015_PointCloud": "left_little/ply",
    "Right_10016_Color": "left_ring/stereo/righteye/rgb",
    "Left_10016_Color": "left_ring/stereo/lefteye/rgb",
    "Deform_10016_Color": "left_ring/deform",
    "Deform_10016_PointCloud": "left_ring/ply",
    "Right_10017_Color": "left_middle/stereo/righteye/rgb",
    "Left_10017_Color": "left_middle/stereo/lefteye/rgb",
    "Deform_10017_Color": "left_middle/deform",
    "Deform_10017_PointCloud": "left_middle/ply",
    "Right_10018_Color": "left_index/stereo/righteye/rgb",
    "Left_10018_Color": "left_index/stereo/lefteye/rgb",
    "Deform_10018_Color": "left_index/deform",
    "Deform_10018_PointCloud": "left_index/ply",
    "Right_10019_Color": "left_thumb/stereo/righteye/rgb",
    "Left_10019_Color": "left_thumb/stereo/lefteye/rgb",
    "Deform_10019_Color": "left_thumb/deform",
    "Deform_10019_PointCloud": "left_thumb/ply",
}

# to be deleted
RAW10_2_RAW11_meta_json = {
  "hand_action":"/action/right_hand/timestamp",
  "robot_action":"/action/right_arm/timestamp",

  "/observation/right_hand/joint_angle":"/observation/right_hand/timestamp",
  "/observation/robot":"/observation/right_arm/timestamp",

  "/observation/images/Head_OrbbecCamera_Color": "/observation/images/head_rgbd_timestamp",
  "/observation/images/Right_Realsense_Color":"/observation/images/right_wrist_timestamp",

  "/observation/images/Left_Stereo_Color":"/observation/images/head_stereo_left_timestamp",
  "/observation/images/Right_Stereo_Color":"/observation/images/head_stereo_right_timestamp",
}

def test_reduce_naming():
    deduce_naming('TRAIN_0_2_0')
    deduce_naming('TRAIN_0_1_1')
    deduce_naming('ANNO_0_2_0@2024_03_43')
    deduce_naming('0.2.0')
    deduce_naming('0.1.1')

if __name__ == '__main__':
    test_reduce_naming()
