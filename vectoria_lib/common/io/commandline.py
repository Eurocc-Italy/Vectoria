#
# 
#
# @authors : Eric Pascolo
#
import os, sys
import logging
import argparse
from vectoria_lib.common import utils
from vectoria_lib.common.io import file_reader
from pathlib import Path
from vectoria_lib.common.paths import ETC_DIR

####--------------------------------------------------------------------------------------------------------------

def create_arg_dict_from_string(a):
    
    """ Transform string in args dictionary in python callable object. 
        Not use eval() to avoid code error or hacking """
    
    logger = logging.getLogger("io")
    
    for k,v in get_iter_object_from_dictionary(a):
        
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
    check_setting_path = get_setting_file_path(json_args)
        
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

def get_setting_file_path(filename) -> list:

    """ Return a list of setting file given its name, the search path is in oreder etc and etc/default """

    check_setting_path = []
    
    check_setting_path_standard = ETC_DIR / "custom" / "cli" / filename
    check_setting_path_default  = ETC_DIR / "default" / "cli" / filename
    
    ## check if setting file is in default location
    if os.path.exists(check_setting_path_default):
        check_setting_path.append(check_setting_path_default)
        check_setting_find = False
   
    ## check if you create a personal setting file
    if os.path.exists(check_setting_path_standard):
        check_setting_path.append(check_setting_path_standard)
        check_setting_find = False
    
    ## setting file not found
    if check_setting_find:
        sys.exit("ERROR CHECK setting file not found:"+filename)

    return check_setting_path
    
####--------------------------------------------------------------------------------------------------------------

def cl_convert_to_dict(args):
    """
    Convert args object in dictionary
    """
    
    convdict = get_iter_object_from_dictionary(vars(args))
    newdict =   dict([(vkey, vdata) for vkey, vdata in convdict if(vdata) ])
    
    return newdict

####--------------------------------------------------------------------------------------------------------------


def get_iter_object_from_dictionary(d):
    """ Return different iter object of dictionary, the objects depends to python version"""

    if sys.version_info[:2] < (3,0):
        return d.iteritems()
    else:
        return d.items()
