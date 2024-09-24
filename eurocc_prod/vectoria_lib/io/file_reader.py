#
# CHECK
#
# @authors : Eric Pascolo
#

import sys
import json
import os
from vectoria_lib.common import utils

"""
Python module that contain read file methods
"""

####--------------------------------------------------------------------------------------------------------------

def json_reader(filepath):
    """
    Checking existence of json file and read it
    """
    filecontained = "-999"
    try:
        if os.path.isfile(filepath) :
            filecontained = json.loads(open(filepath).read())
        else :
            sys.exit()

    except:
        sys.exit("ERROR JSON file not readable "+filepath)

    return filecontained


####--------------------------------------------------------------------------------------------------------------

def filetostring(filepath):
    
    onestring=""
    
    try:
        multistringfile = generic_file_reader(filepath)
        for r in multistringfile:
            onestring = onestring+r
    except:
        print("ERROR file dosn't exist:"+filepath)
    
    return onestring


####--------------------------------------------------------------------------------------------------------------

def generic_file_reader(filename):
    """
        Read a generic ascii file and return an array of string (1 per row)
    """
    rows_from_file = []
    if os.path.isfile(filename):
        file_toread = open(filename, 'r')
        rows_from_file = file_toread.readlines()
        file_toread.close()
    else:
        print("ERROR file dosn't exist:"+filename)

    return rows_from_file

####--------------------------------------------------------------------------------------------------------------