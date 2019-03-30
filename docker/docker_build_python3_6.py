#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import getpass

if __name__=="__main__":

	print("building docker container . . . ")
	user_name = getpass.getuser()
	image_name = "pytorch_adabound"
	default_image_name = user_name + "-" + image_name

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--image", type=str,
		help="name for the newly created docker image", default=default_image_name)

	args = parser.parse_args()
	print("building docker image named ", args.image)
	cmd = "docker build"
	cmd += " -t %s -f" % args.image
	cmd +=  image_name + ".dockerfile ."


	print("command = \n \n", cmd)
	print("")

	# build the docker image
	print("executing shell command")
	os.system(cmd)
