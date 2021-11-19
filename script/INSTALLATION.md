# Installation of optional faster packages (Ubuntu 20.04) for ActEV'21

# System prep
sudo apt-get install ffmpeg -y
sudo apt-get install zlib1g-dev libjpeg-dev libjpeg62 -y
sudo apt-get install python3.9 python3.9-dev python3.9-venv python3-pip  -y

# Create virtualenv
python3.9 -m venv ~/virtualenv/heyvi
source ~/virtualenv/heyvi/bin/activate

# Virtualenv installation
python3 -m pip install pip setuptools twine wheel --upgrade
python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install vipy pycollector
python3 -m pip uninstall -y pillow
CC="cc -mavx2" python3 -m pip install --global-option="build_ext"  -U --force-reinstall pillow-simd --global-option="--enable-jpeg" --global-option="--enable-zlib"
python3 -m pip install icc_rt numba ujson ipython

