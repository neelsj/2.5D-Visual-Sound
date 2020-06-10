FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

USER root

RUN apt-get -y update --allow-unauthenticated --allow-insecure-repositories

RUN apt-get --purge -y remove "*cublas*" "cuda*"
RUN apt-get -y install cuda-9.2

RUN apt-get -y install build-essential cmake
RUN apt-get -y install libgtk-3-dev
RUN apt-get -y install libsndfile1

RUN pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html --ignore-installed

RUN pip install matplotlib
RUN pip install scikit-image
RUN pip install scikit-learn
RUN pip install opencv-python
RUN pip install tensorboard
RUN pip install librosa==0.6.3

RUN pip uninstall -y numpy
RUN pip uninstall -y numpy
RUN pip install numpy

RUN pip install tqdm
RUN pip install joblib

RUN pip install h5py