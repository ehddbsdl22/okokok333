#FROM nvidia/cuda:10.0-base-ubuntu16.04
#FROM centos:7
#FROM centos/python-36-centos7
#USER root
#MAINTAINER  jeon

FROM python:3.6

RUN yum -y install epel-release; yum clean all
RUN yum -y install python-pip; yum clean all

RUN mkdir /src
WORKDIR /src

#COPY libm.so.6 /lib/libm.so.6

COPY requirements.txt /src/requirements.txt
RUN pip install -r requirements.txt

COPY . /src
RUN pip install .



ENV LD_LIBRARY_PATH ./:$LD_LIBRARY_PATH
RUN export LD_LIBRARY_PATH PATH

#RUN LD_LIBRARY_PATH="/opt/root-app/src:$LD_LIBRARY_PATH"
#RUN export LD_LIBRARY_PATH PATH


#RUN sed -i 's/archive.ubuntu.com/mirror.us.leaseweb.net/' /etc/apt/sources.list \
#    && sed -i 's/deb-src/#deb-src/' /etc/apt/sources.list \
#    && apt-get update \
#    && apt-get upgrade -y \
#    && apt-get install -y \
#    build-essential \
#    ca-certificates \
#    gcc \
#    git \
#    libpq-dev \
#    make \
#    pkg-config \
#    python3 \
#    python3-dev \
#    python3-pip \
#    aria2 \
#    && apt-get autoremove -y \
#    && apt-get clean

#RUN mkdir -p /root/detection
#WORKDIR /root/detection
#WORKDIR /opt/app-root/src
#ADD . /root/detection/
#ADD . /opt/app-root/src/


#RUN apt-get update
#RUN apt-get install wget

#RUN pip3 install --upgrade pip
#RUN pip3 install -U virtualenv
#RUN pip3 install zipp==1.0.0
#RUN yum -y install libXrender
#RUN yum -y install libXrender1
#RUN yum -y install libXrender-dev

#RUN pip3 install -r requirements.txt
#RUN apt-get update
#RUN apt-get install -y libsm6 libxext6 libxrender-dev
#RUN pip3 install opencv-python

#RUN apt-get install -y python3.5-tk
#RUN pip3 install pillow
#RUN pip3 install flask

 #echo "TEST----1"
#RUN pip3 install tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl
#RUN pip3 list
# echo "TEST----2"

#RUN unlink /usr/lib64/libm.so.6
#COPY /opt/app-root/src/libm.so.6 /lib64/


#CMD []
#ENTRYPOINT ["/bin/bash" , ]
#CMD ["pip", "install tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl"]

#RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./ 
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./
#CMD ["LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/root-app/src python3", "app.py"]
#ENTRYPOINT ["./export.sh"]

#ENV PYTHONPATH=/deploy
#ENV PATH ./:$PATH
#ENV LD_LIBRARY_PATH ./:$LD_LIBRARY_PATH


EXPOSE 8080

CMD ["python3", "app.py"]

#RUN 
#RUN sed -i 's/exec sudo -E -H -u $NB_USER/exec sudo -E -H -u $NB_USER LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH/' ./export.sh

#ENTRYPOINT ["/bin/bash", "$LD_LIBRARY_PATH", python3", "app.py"]
#ENTRYPOINT ["./export.sh"]
#CMD ["export.sh"]


