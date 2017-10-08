FROM bvlc/caffe:cpu

COPY . /root/app
WORKDIR /root/app

RUN pip install -r requirements.txt