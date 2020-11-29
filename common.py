
import yaml
import sounddevice as sd



def load_config(filename='settings.yaml'):
    '''
    read a yaml file

    :param filename: filename
    :return: dict
    '''
    cfg = {}

    with open(filename, 'rt') as f:
        cfg = yaml.safe_load(f)

    return cfg


def find_device_id(dev_str):
    '''
    find audio device id by looking for a substring in the device name

    :param dev_str: substring to search in the device name
    :return: device id. returns None if no device is found
    '''

    device_list = sd.query_devices()

    for dev_ind, dev in enumerate(device_list):
        if dev_str in dev['name']:
            return dev_ind

    return None