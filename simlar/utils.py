import sys, os, importlib, glob, yaml
import torch
import numpy as np
from functools import partial
import hashlib

def validate_config_keys(user_config, default_config, parent_key=""):
    """
    Recursively validate that the keys in the user-provided configuration match the default configuration.

    Parameters
    ----------
    user_config : dict
        The user-provided configuration dictionary.
    default_config : dict
        The default configuration dictionary.
    parent_key : str
        The parent key path for nested dictionaries (used for error messages).

    Raises
    ------
    ValueError
        If there are missing or extra keys in the user configuration.
    """
    # Ensure both inputs are dictionaries
    if not isinstance(user_config, dict) or not isinstance(default_config, dict):
        raise ValueError(f"Expected dictionaries at '{parent_key}', but got {type(user_config)} and {type(default_config)}")

    # Get the keys from both dictionaries
    user_keys = set(user_config.keys())
    default_keys = set(default_config.keys())

    # Find missing and extra keys
    missing_keys = default_keys - user_keys
    extra_keys = user_keys - default_keys

    # Raise errors if there are mismatches
    if missing_keys:
        raise ValueError(f"Missing keys in user configuration at '{parent_key}': {missing_keys}")
    if extra_keys:
        raise ValueError(f"Extra keys in user configuration at '{parent_key}': {extra_keys}")

    # Recursively validate nested dictionaries
    for key in default_keys:
        if isinstance(default_config[key], dict):
            # Construct the full key path for error messages
            full_key = f"{parent_key}.{key}" if parent_key else key
            validate_config_keys(user_config[key], default_config[key], parent_key=full_key)


def build_config(config):
    """
    Build the configuration dictionary by validating the keys and values.

    Parameters
    ----------
    config : dict
        The configuration dictionary to be validated and built.

    Returns
    -------
    dict
        The validated and built configuration dictionary.
    """
    # Load user configuration
    if os.path.isfile(config):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)
    elif config in list_config():
        config = load_config(config)
    else:
        raise ValueError(f"Configuration file or key '{config}' not found.")

    # Load the default configuration
    default_config = load_config('run_develop')

    # Validate the user-provided configuration against the default configuration
    for key in default_config.keys():
        if key == 'DEVICE':
            if not key in config:
                raise ValueError(f"Missing key '{key}' in user configuration")
            if not config[key] in list_available_devices():
                raise ValueError(f"Unavailable device '{config[key]}' requested")
        elif key == 'DEBUG':
            if not key in config:
                raise ValueError(f"Missing key '{key}' in user configuration")
        else:
            if not key in config:
                raise ValueError(f"Missing key '{key}' in user configuration")
            if isinstance(default_config[key], str):
                default_config[key] = load_config(default_config[key])
            #if isinstance(config[key], str) and key in default_config and isinstance(default_config[key], dict):
                #config[key] = load_config(config[key])
            if isinstance(config[key], str):
                config[key] = load_config(config[key])
            validate_config_keys(config[key],default_config[key])

            if type(config[key]) != type(default_config[key]):
                try:
                    config[key] = type(default_config[key])(config[key])
                except Exception as e:
                    raise ValueError(f"Type mismatch for key '{key}': expected {type(default_config[key])}, got {type(config[key])}. Error: {e}")

    return config


def get_file_checksum(file_path, hash_algorithm='md5'):
    """
    Calculate the checksum of a file using the specified hash algorithm.

    Parameters
    ----------
    file_path : str
        Path to the file.
    hash_algorithm : str
        Hash algorithm to use (e.g., 'md5', 'sha256').

    Returns
    -------
    str
        The checksum of the file.
    """
    hash_func = hashlib.new(hash_algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def list_available_devices():
    '''
    List available devices [cpu|cuda|cuda:i|mps]

    Returns
    -------
    list
        Available torch.device instances
    '''
    devs=dict(cpu=torch.device('cpu'))

    if torch.cuda.is_available():
        devs['cuda'] = torch.device('cuda:0')

        device_count = torch.cuda.device_count()
        for i in range(device_count):
            devs['cuda:%d' % i] = torch.device('cuda:%d' % i)
        
    if torch.backends.mps.is_available():
        devs['mps'] = torch.device('mps')

    return devs


def get_device(request):
    '''
    Check and return if the requested device is available

    Parameters
    ----------
    request : str
        Request type of a device

    Returns
    -------
    torch.device
        The requested device instance
    '''
    devs = list_available_devices()

    if not request in devs:
        print(request,'not supported')
        return None
    else:
        return devs[request]


def import_from(src):
    '''
    Import a python object from a period-separated module tree (e.g. configuration file argument)

    Parameters
    ----------
    src : str
        The module tree path (e.g. numpy.random.random becomes "from numpy.random import random")

    Returns
    -------
    object
        Python imported object
    '''
    if src.count('.') == 0:
        module = sys.modules['__main__']
        obj_name = src
    else:
        module_name, obj_name = os.path.splitext(src)
        module = importlib.import_module(module_name)

    return getattr(module, obj_name.lstrip('.'))



def get_config_dir():
    '''
    Function to return the module configuration directory.

    Returns
    -------
    str
        Path to the configuration directory
    '''

    return os.path.join(os.path.dirname(__file__),'config')


def list_config(full_path=False):
    '''
    List available "default" configurations that come with slar

    Parameters
    ----------
    full_path : bool
        If True, a list of configuration file (full) paths are returned. 
        If False, a configuration keywords will be returned (see get_config function).

    Returns
    -------
    list
        Either a list of full path to configuration files or configuration keywords
        (see get_config function).
    '''
    fs = glob.glob(os.path.join(get_config_dir(), '*.yaml'))

    if full_path:
        return fs

    return [os.path.basename(f)[:-5] for f in fs]


def get_config(name):
    '''
    Returns the full path to a configuration given the keyword

    Parameters
    ----------
    name : str
        A configuration keyword specific to each of prepared configurations.

    Returns
    -------
        A full path to the specified configuration file.
    '''
    options = list_config()
    results = list_config(True)

    if name in options:
        return results[options.index(name)]

    alt_name = name + '.yaml'
    if alt_name in options:
        return results[options.index(alt_name)]

    print('No data found for config name:',name)
    raise NotImplementedError


def load_config(name):
    '''
    Wrapper function for get_config and this return the configuration dictionary
    after interpreting the configuration file.

    Parameters
    ----------
    name : str
        A configuration keyword specific to each of prepared configurations.

    Returns
    -------
    dict
        A loaded (interpreted) configuration file contents.
    '''
    return yaml.safe_load(open(get_config(name),'r'))


class CSVLogger:
    '''
    Logger class to store training progress in a CSV file.
    '''

    def __init__(self,cfg):
        '''
        Constructor

        Parameters
        ----------
        cfg : dict
            A collection of configuration parameters. `dir_name` and `file_name` specify
            the output log file location. `analysis` specifies analysis function(s) to be
            created from the analysis module and run during the training.
        '''
        
        log_cfg = cfg.get('logger',dict())
        self._logdir  = self.make_logdir(log_cfg.get('dir_name','logs'))
        self._logfile = os.path.join(self._logdir, cfg.get('file_name','log.csv'))
        self._log_every_nsteps = log_cfg.get('log_every_nsteps',1)

        print('[CSVLogger] output log directory:',self._logdir)
        print(f'[CSVLogger] recording a log every {self._log_every_nsteps} steps')
        self._fout = None
        self._str  = None
        self._dict = {}
        
        self._analysis_dict={}
        
        for key, kwargs in log_cfg.get('analysis',dict()).items():
            print('[CSVLogger] adding analysis function:',key)
            self._analysis_dict[key] = partial(getattr(importlib.import_module('slar.analysis'),key), **kwargs)
        
    @property
    def logfile(self):
        return self._logfile
    
    @property
    def logdir(self):
        return self._logdir
        
    def make_logdir(self, dir_name):
        '''
        Create a log directory

        Parameters
        ----------
        dir_name : str
            The directory name for a log file. There will be a sub-directory named version-XX where XX is
            the lowest integer such that a subdirectory does not yet exist.

        Returns
        -------
        str
            The created log directory path.
        '''
        versions = [int(d.split('-')[-1]) for d in glob.glob(os.path.join(dir_name,'version-[0-9][0-9]'))]
        ver = 0
        if len(versions):
            ver = max(versions)+1
        logdir = os.path.join(dir_name,'version-%02d' % ver)
        os.makedirs(logdir)

        return logdir

    def record(self, keys : list, vals : list):
        '''
        Function to register key-value pair to be stored

        Parameters
        ----------
        keys : list
            A list of parameter names to be stored in a log file.

        vals : list
            A list of parameter values to be stored in a log file.
        '''
        for i, key in enumerate(keys):
            self._dict[key] = vals[i]
            
    def step(self, iteration, label=None, pred=None):
        '''
        Function to take a iteration step during training/inference. If this step is
        subject for logging, this function 1) runs analysis methods and 2) write the
        parameters registered through the record function to an output log file.

        Parameters
        ----------
        iteration : int
            The current iteration for the step. If it's not modulo the specified steps to
            record a log, the function does nothing.

        label : torch.Tensor
            The target values (labels) for the model run for training/inference.

        pred : torch.Tensor
            The predicted values from the model run for training/inference.


        '''
        if not iteration % self._log_every_nsteps == 0:
            return
        
        if not None in (label,pred):
            for key, f in self._analysis_dict.items():
                self.record([key],[f(label,pred)])
        self.write()

    def write(self):
        '''
        Function to write the key-value pairs provided through the record function
        to an output log file.
        '''
        if self._str is None:
            self._fout=open(self._logfile,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'
        self._fout.write(self._str.format(*(self._dict.values())))
        self.flush()
        
    def flush(self):
        '''
        Flush the output file stream.
        '''
        if self._fout: self._fout.flush()

    def close(self):
        '''
        Close the output file.
        '''
        if self._str is not None:
            self._fout.close()
