#!/bin/bash



# during every new terminal----------------------------------
export LD_LIBRARY_PATH=\
$HOME/other/shim312_new:\
$VIRTUAL_ENV/lib/python3.12/site-packages/av.libs:\
/home/sr5/sairaj.loke/other/uvexpt/python/lib:\
$LD_LIBRARY_PATH

source $HOME/other/venv312_new/bin/activate
# -------------------------------------------------------------






########### to setup again ################ on a new system (this has been testd on a100 )



# # 1. Create venv from the standalone shared-libpython 3.12
# /home/sr5/sairaj.loke/other/uvexpt/python/bin/python3 -m venv $HOME/other/venv312_new
# source $HOME/other/venv312_new/bin/activate


# /home/sr5/sairaj.loke/other/uvexpt/python/bin/python3 -m venv $HOME/other/venv312_new
# source $HOME/other/venv312_new/bin/activate

# pip install uv

# export UV_INDEX_URL=http://202.20.187.241:9001/artifactory/api/pypi/pypi-remote/simple/
# export UV_INSECURE_HOST=202.20.187.241

# uv pip install -r /home/sr5/sairaj.loke/other/ori/ori/ViTacFormer/req_uv.txt


######### #NOTE: ############## only when getting the so files for ffmpeg 
VENV=$HOME/other/venv312_new
AVLIBS=$VENV/lib/python3.12/site-packages/av.libs
SHIM=$HOME/other/shim312_new


set -euo pipefail

mkdir -p "$SHIM" && cd "$SHIM"

for n in libavutil libavcodec libavformat libavfilter libavdevice libswscale libswresample; do
  src=$(ls "$AVLIBS"/${n}-*.so.* 2>/dev/null | head -1)
  if [ -z "$src" ]; then echo "MISSING: $n" >&2; exit 1; fi
  major=$(basename "$src" | sed -E 's/.*\.so\.([0-9]+).*/\1/')
  ln -sf "$src" "${n}.so.${major}"
  echo "linked ${n}.so.${major}"
done

ls -l
######################################################################


# during every new terminal----------------------------------
export LD_LIBRARY_PATH=\
$HOME/other/shim312_new:\
$VIRTUAL_ENV/lib/python3.12/site-packages/av.libs:\
/home/sr5/sairaj.loke/other/uvexpt/python/lib:\
$LD_LIBRARY_PATH

source $HOME/other/venv312_new/bin/activate
# -------------------------------------------------------------

# python -c "
# from torchcodec.decoders import VideoDecoder
# d = VideoDecoder('/home/sr5/sairaj.loke/other/new_data/larger_data/videos/observation.images.head_left/chunk-000/file-000.mp4')
# print(d.metadata)
# f = d[0]; print(f.shape, f.dtype, f.float().mean())
# "

# python -c "
# # from torchcodec.decoders import VideoDecoder
# # d = VideoDecoder('PATH/TO/episode_000000.mp4')
# # print(d.metadata)
# # f = d[0]; print(f.shape, f.dtype, f.float().mean())
# # "


# source $HOME/other/venv310/bin/activate remem this 

# python --version
# python -c "import sysconfig; print(sysconfig.get_config_var('Py_ENABLE_SHARED'))"   # must be 1


