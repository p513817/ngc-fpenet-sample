FROM nvcr.io/nvidia/tensorrt:21.03-py3
USER root

# Set Environment Variable
ENV TEMP_WS=/opt/fpenet_sample \
    WS=/workspace \
    DEBIAN_FRONTEND=noninteractive \
    TIMEZONE=Asia/Taipeia 

# Setting TimeZone
RUN apt-get -yq update && \
    apt-get -yq install tzdata && \
    ln -fs /usr/share/zoneinfo/${TIMEZONE} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Install Basic Package
RUN apt-get install -yq cmake git libgtk2.0-dev pkg-config yasm libcanberra-gtk-module \
    bsdmainutils zip jq wget \
    openssl libssl-dev openssh-server \
    ubuntu-restricted-extras \
    python3-dev python3-numpy \
    figlet boxes tree usbutils lsof && \
    apt-get -o Dpkg::Options::="--force-confmiss" install -y --reinstall netbase

# For TLT Converter
RUN apt-get install -qqy libssl-dev #> /dev/null 2>&1 && \
    echo 'export TRT_LIB_PATH=/usr/lib/x86_64-linux-gnu' >> ~/.bashrc  && \
    echo 'TRT_INC_PATH=/usr/include/x86_64-linux-gnu' >> ~/.bashrc 

# Maybe need ONNX
RUN pip3 install onnx==1.9.0

# Maybe need Msicellaneous packages
RUN pip3 install \
    wget \
    colorlog \
    tqdm==4.64.0 \
    cython==0.29.32 \
    gdown==4.5.1 \
    setuptools==52.0.0 \
    packaging==21.3 \
    GPUtil==1.4.0 \
    Cython==0.29.32 \
    gdown==4.5.1 \
    psutil==5.9.1 \
    flasgger==0.9.5 \
    Flask==2.0.3 \
    Flask-Cors==3.0.10 \
    flask-sock==0.5.2 \
    python-dateutil==2.8.2 \
    Flask-Cors==3.0.10 \
    Flask-SocketIO==5.1.2 \
    flask-socketio==5.1.2 \
    gunicorn==20.1.0 \
    eventlet==0.30.2 \
    python-dateutil==2.8.2 \
    python-engineio==4.3.2 \
    python-socketio==5.6.0 \
    flask_mqtt==1.1.1 \
    paho-mqtt==1.6.1

# For OpenCV
RUN apt-get install -qqy libxrender1 libsm6 libxext6 && \
    pip3 install --disable-pip-version-check opencv-python
