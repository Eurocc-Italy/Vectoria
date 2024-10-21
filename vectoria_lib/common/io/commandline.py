#
# CHECK
#
# @authors : Eric Pascolo
#

import logging
import argparse
import sys
from vectoria_lib.common import utils
from vectoria_lib.io import file_reader
from pathlib import Path

####--------------------------------------------------------------------------------------------------------------

def create_arg_dict_from_string(a):
    
    """ Transform string in args dictionary in python callable object. 
        Not use eval() to avoid code error or hacking """
    
    logger = logging.getLogger("io")
    
    for k,v in utils.get_iter_object_from_dictionary(a):
        
        if k == "required": 
          
            if v == "True": 
                a[k] = True
            else:
                a[k] = False


        if k == "type": 
            
            if v == "str":
                a[k] = str
            if v == "int":
                a[k] = int
            if v == "float":
                a[k] = float
            if v == "open":
                a[k] = open
               


####--------------------------------------------------------------------------------------------------------------

def create_cl_parser_from_json(parser, json_args: str | Path) -> argparse.ArgumentParser:
    """
    Parsing command line pars with argparse module, the cl argoument is dynamically loaded from configuration file,
    and after parser by a fuction that convert string in py callable object.
    """
    check_setting_path = utils.get_setting_file_path(json_args)
        
    # extract setting information from json file
    cl_setting = file_reader.json_reader(check_setting_path[0])
    
    # add parser argument from dictionary obtained to json file
    for arg in cl_setting["flag"]:
        # trasform string in object
        create_arg_dict_from_string(arg["param"])
        # add arg name and arg parameter dictionary
        parser.add_argument(arg["name"], **arg["param"])

    return parser

    
####--------------------------------------------------------------------------------------------------------------

def cl_convert_to_dict(args):
    """
    Convert args object in dictionary
    """
    
    convdict = utils.get_iter_object_from_dictionary(vars(args))
    newdict =   dict([(vkey, vdata) for vkey, vdata in convdict if(vdata) ])
    
    return newdict

####--------------------------------------------------------------------------------------------------------------