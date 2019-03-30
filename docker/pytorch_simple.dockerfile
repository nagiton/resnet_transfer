FROM nvidia/cuda:8.0-devel-ubuntu16.04


RUN apt update
RUN apt install sudo

COPY ./install_dependencies.sh /tmp/install_dependencies.sh
RUN yes "Y" | /tmp/install_dependencies.sh

COPY ./install_pytorch.sh /tmp/install_pytorch.sh
RUN yes "Y" | /tmp/install_pytorch.sh

COPY ./install_more.sh /tmp/install_more.sh
RUN yes "Y" | /tmp/install_more.sh

COPY ./install_atom.sh /tmp/install_atom.sh
RUN yes "Y" | /tmp/install_atom.sh
