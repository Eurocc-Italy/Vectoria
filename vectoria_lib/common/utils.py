#
# CHECK
#
# @authors : Eric Pascolo, Roberto Da Via
#


import os
import re
import sys
import json
import platform
import datetime
from vectoria_lib.common.archive import *
from vectoria_lib.common.paths import ETC_DIR

####--------------------------------------------------------------------------------------------------------------

def resolve_env_path(dictionary):
    '''Given dictionary substitute into path env var'''
    for key, value in get_iter_object_from_dictionary(dictionary):

        try:
            path = os.path.expandvars(value)
            if is_valid(path):
                dictionary[key] = path
        except:
            pass

####--------------------------------------------------------------------------------------------------------------

def is_valid(path):
    '''Check if path is linux path'''

    #re to check path
    prog = re.compile(re_path)#(r"^(/)([^/\0]+(/)?)+$")
    valid = prog.match(path)
    if valid is None:
        return False
    else:
        return True

####--------------------------------------------------------------------------------------------------------------

def split_name_version(software_string):
    '''
    Given the software split name version and architecture
    Use @ symbol to architecture and # symbol for the version   
    '''
    prog = re.compile(re_splitsoftware)
    software_list = re.split(re_splitsoftware,software_string)

    software_hardware = "__all__"
    software_version  = ""
    software_name = software_list[0]
    num_parameter = len(software_list)
    
    if num_parameter >= 2:
        software_hardware = software_list[1]
    if num_parameter == 3:
        software_version  = software_list[2]
    
    return software_name,software_hardware,software_version,num_parameter

####--------------------------------------------------------------------------------------------------------------

def split_hostline(line):
    '''
    Split hostline with this two syntax:

        - arch#recipes:nodelist/
        - arch:nodelist/
    
    The pattern can be repeated with / as separator

    '''

    # find the pattern that mathc with re
    
    reg_compiled = re.compile(re_parser_hostlist)
    result = reg_compiled.findall(line)
    
    architecture = []

    for r in result:
        #check if node is env var and substitute it
        if "$" in r[4]:
            r[4] = os.path.expandvars(r[4])
        
        # see archive.py for details about the regex groups
        nodes=r[4]
        arch=r[1] if r[1] != "" else r[2]
        settings=r[3] if r[3] != "" else "default"

        #split nodelist 
        nodes_splitted = "".join(nodes).split(",")
        
        

        architecture.append({"arch":arch,"setting":settings,"nodes":nodes_splitted})
        
   
    return architecture
    
    

####--------------------------------------------------------------------------------------------------------------

def list_to_String(slist,separator):
    '''
    Return string from list symbol separeted and remove newline and substitute double quote
    '''

    return remove_newline_in(separator.join(slist).replace("\"","\'")) 

####--------------------------------------------------------------------------------------------------------------

def extract_elements_from_dict_by_keylist(keylist,dictionary):
    '''
    Given a key list and  dict{of list[]} return a single list
    extended with each list in a dictionary that have a key in keylist 
    '''
    dict_list = [] 
    
    for k in keylist: 
        
        if k in dictionary:
            dict_list.extend(dictionary[k])
        else:
            dict_list.append(str(k)) 
    
    return dict_list


####--------------------------------------------------------------------------------------------------------------

def remove_newline_in(stringline):
    '''
    remove newline at the end of the string
    '''
    if stringline.endswith("\r\n"): 
        return stringline[:-2]
    if stringline.endswith("\n"): 
        return stringline[:-1]
    else :
        return stringline


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

def get_iter_object_from_dictionary(d):
    """ Return different iter object of dictionary, the objects depends to python version"""

    if sys.version_info[:2] < (3,0):
        return d.iteritems()
    else:
        return d.items()

####--------------------------------------------------------------------------------------------------------------

def is_tool(tool_name):
    """Check if tool is on PATH."""

    from distutils.spawn import find_executable

    return find_executable(tool_name) is not None

####--------------------------------------------------------------------------------------------------------------

def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except ValueError as e:
    return False
  return True

####--------------------------------------------------------------------------------------------------------------

def convert_request_to_json(s0):
    
    s1 = s0.replace("<","{")
    s2 = s1.replace(">","}")
    s3 = s2.replace("=",":")
    s4 = s3.replace(";",",")
    if is_json(s4):
        return json.loads(s4)
        
    else:
        return "-999"

####--------------------------------------------------------------------------------------------------------------

def get_name_of_nodes(resources=None):
    ''' 
    Return a name of the node or a list of the node if it is in a job
    '''
    nodename = platform.node()
    if resources is not None:
        if not "$" in resources["NODENAME"]:
            nodename = resources["NODENAME"]
    
    return nodename

####--------------------------------------------------------------------------------------------------------------

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]