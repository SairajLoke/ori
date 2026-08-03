from curses import meta
from pathlib import Path
import pdb
import json
import os
import time

from .naming_rule import deduce_naming
from ha_data.data import naming_rule

anno_template = {
    "basic_tag": "harobotDE_0_0.0",
    "date": "2024_11_05_17_45_50",
    "depend": "data",
    "url": []
}

meta_template = {
  "content": {
    "anno": {},
    "data": {},
  },
  "version": "V0"
}

def timeformat_h(stamp):
   return time.strftime("%Y-%m-%d %H:%M:%S", stamp)

def timeformat_name(stamp):
   return time.strftime("%Y_%m_%d_%H_%M_%S", stamp)


class MetaUnit:
    def __init__(self, context):
        self.context = context
        self.key = ''
        self.group = ''


    def _get_group(self):
        return ''
    def _get_key(self):
        return ''
    def _get_version_prefix(self, version):
        res = version.split('_')
        return f'{res[0]}_{res[1]}_{res[2]}'


    def _set_version(self, data):
        return {}

    def init(self, depend, stamp='', meta_content=None):
        if stamp == '':
            stamp  = timeformat_name(time.localtime())
        self.group = self._get_group()
        self.key = self._get_key()
        self.context.meta['content'][self.group][self.key] = {}
        self.context.meta['content'][self.group][self.key]["depend"] =str(depend)
        self.context.meta['content'][self.group][self.key]["date"] = stamp
        if meta_content is not None:
            self.context.meta['content'].update(meta_content)

    def add_key(self, key, value):
        self.context.meta['content'][self.group][self.key][key] = value

    def add(self, set_default = True):
        data =self._set_version()
        # data["content_meta"]["sensor"] = sensor_list
        data["url"] = []
        group = self._get_group()
        key = self._get_key()
        for episode in os.listdir(self.context.root):
            path = os.path.join(self.context.root, episode)
            if not os.path.isdir(path): continue
            hdf5_file = self.get_latest_hdf5(path)
            if hdf5_file is None:
                print('failed to find hdf5 file', path)
                continue
            data["url"].append(str(os.path.join(episode, hdf5_file)))
        print(f'train_meta find {len(data["url"])} {key} hdf5 files')
        self.context.meta['content'][group][key].update(data)
        if set_default:
            self.context.meta['content'][group]['default'] = key

    # major, minor, patch
    def get_from_simple_version(self, version):
        # print("deprecated interface, using get for new version ('0.2.2')")
        key = f'{version[0]}.{version[1]}.{version[2]}'
        print(f'simple version major:{version[0]}, minor:{version[1]}, patch:{version[2]}, makeup for {key}')
        return self.get(key)

    def get(self, version = ""):
        try:
            prefix = self._get_group()
            if version == "":
                version = self.context.meta["content"][prefix]["default"]
                return self.context.meta["content"][prefix][version], version
            elif self.context.meta["content"][prefix]['default']== version:
                return self.context.meta["content"][prefix][version], version
            elif self.context.meta["content"][prefix]['default'].split('_')[-1] == version:
                new_version = self.context.meta["content"][prefix]['default']
                return self.context.meta["content"][prefix][new_version], new_version
            elif '@' in version: # 0.1.0, 0.2.0
                return self.context.meta["content"][prefix][version], version
            else: # 0.1.0, 0.2.0
                vp_len = len(version)
                key_candidate = []
                for key, value in self.context.meta["content"][prefix].items():
                    if key == 'default':
                        continue
                    tag_version = value['basic_tag'].split('_', 1)[-1].replace('_', '.')
                    if (tag_version[:vp_len] == version or tag_version[:vp_len]=='2.0.0') and len(value['url']) > 0:
                        key_candidate.append(key)
                assert len(key_candidate) > 0
                key = sorted(key_candidate)[-1]
                return self.context.meta["content"][prefix][key], key
        except Exception as e:
            print('error', e, prefix)
            exit()


class DataUnit(MetaUnit):
    def get_h5_name(self, now):
        if now == '':
            now = timeformat_name(time.localtime())
        return f'{self.context.naming.DATA_VERSION}@{now}.hdf5'

    def _get_group(self):
        return 'data'

    def _get_key(self):
        return self.context.naming.DATA_VERSION

    def get_latest_hdf5(self, path):
        try:
            hdf5s = Path(path).glob(f"{self._get_key()}*.hdf5")
            files = sorted(list(hdf5s))
            return files[-1].name
        except:
            return None

    def _set_version(self):
        data = {}
        data["basic_tag"] = self.context.naming.DATA_VERSION
        data["sdk_version"] = self.context.naming.LUBAN_VERSION
        return data


class AnnoUnit(MetaUnit):
    def get_h5_name(self, now):
        if now == '':
            now = timeformat_name(time.localtime())
        return f'anno/{self.context.naming.ANNO_VERSION}@{now}.hdf5'

    def _get_group(self):
        return 'anno'

    def get_latest_hdf5(self, path):
        try:
            hdf5s = Path(path).glob(f"anno/{self._get_key()}*.hdf5")
            files = sorted(list(hdf5s))
            return f'anno/{files[-1].name}'
        except:
            return None

    def _get_key(self):
        return self.context.naming.ANNO_VERSION

    def _set_version(self):
        data = {}
        data["basic_tag"] = self.context.naming.ANNO_VERSION
        return data

class Context:
    def __init__(self, root, naming=''):
        self.root = root
        self.meta_path = str(root) + '/train.json'
        try:
            with open(self.meta_path, 'r') as f:
                self.meta = json.load(f)
        except Exception as e:
            self.meta = meta_template
        try:
            if naming == '':
                naming = str(self.meta['content']['data']['default'])
            self.naming = deduce_naming(naming)
        except Exception as e:
            pdb.set_trace()
            print('failed to decuce naming convetion!!!', e, str(self.meta['content']['data']['default']))
            exit()

    def dump(self, dump_path = ''):
        if dump_path == '':
            dump_path = self.meta_path
        with open(dump_path, 'w') as f:
            # print(self.meta)
            json.dump(self.meta, f)

    def get_episode_path(self, robotSN, stamp):
        # todo use now??
        return f'{robotSN}_{stamp}_train/'

class TrainMeta:
    def __init__(self, root, naming='0.3.0'):
        self.context = Context(root, naming)
        self.data = DataUnit(self.context)
        self.anno = AnnoUnit(self.context)

    def redump(self, path='', set_stamp = ''):
        if set_stamp == '':
            set_stamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        if path=='':
            path = self.context.root
        data_version = self.data._get_key()
        try:
            jj = json.load(open(self.context.root+'/train.json', 'r'))
            data_depend = jj['content']['data'][data_version]['depend']
            print(f'inherit data depend={data_depend}')
        except:
            data_depend = ''
        self.data.init(data_depend, set_stamp)
        self.data.add()
        self.anno.init('data/'+ data_version, set_stamp)
        self.anno.add()
        self.context.dump(path)


    def get_data_h5_path(self, robotSN, stamp, gen_stamp=''):
        if gen_stamp == '':
            gen_stamp = stamp
        return self.context.get_episode_path(robotSN, stamp) \
            + self.data.get_h5_name(gen_stamp)

    def get_anno_h5_path(self, data_h5, gen_stamp=''):
        robotSN = data_h5.split('_', 1)[0]
        stamp = data_h5.split('@')[-1][:-5]
        if gen_stamp == '':
            gen_stamp = stamp
        return self.context.get_episode_path(robotSN, stamp) \
            + self.anno.get_h5_name(gen_stamp)
