#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import socket
import getpass
import yaml

if __name__=="__main__":
    user_name = getpass.getuser()
    image_name = "pytorch_adabound"
    default_image_name = user_name + '-' + image_name
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str,
        help="(required) name of the image that this container is derived from", default=default_image_name)

    parser.add_argument("-c", "--container", type=str, default=image_name, help="(optional) name of the container")


    args = parser.parse_args()
    print("running docker container derived from image %s" %args.image)
    source_dir=os.path.join(os.getcwd(),"../")

    print(source_dir)

    home_directory = '/home/' + user_name
    dense_correspondence_source_dir = os.path.join(home_directory, 'code')

    cmd = "xhost +local:root \n"
    cmd += "nvidia-docker run "
    if args.container:
        cmd += " --name %(container_name)s " % {'container_name': args.container}

    cmd += " -v %(source_dir)s:%(home_directory)s" \
        % {'source_dir': source_dir, 'home_directory': home_directory}

    cmd += " -w %(home_directory)s"\
        % {'home_directory': home_directory} #default directry in the container

    cmd += " -it "
    cmd += " --rm " # remove the image when you exit

    cmd += args.image

    cmd_endxhost = "xhost -local:root"



    print("command = \n \n", cmd, "\n", cmd_endxhost)
    print("")

    # build the docker image

    print("executing shell command")
    code = os.system(cmd)
    print("Executed with code ", code)
    os.system(cmd_endxhost)
        # Squash return code to 0/1, as
        # Docker's very large return codes
        # were tricking Jenkins' failure
        # detection
    exit(code != 0)
