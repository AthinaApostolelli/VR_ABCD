import numpy as np
import h5py
import pandas as pd

def add_time(session,framerate=45):
    max_time = len(session['position'])/framerate
    session['time'] = np.linspace(0,max_time,len(session['position']))

    return session

def add_lick_rate(session):
    # calculate lick rate in a sliding window
    window = 100
    lick_rate = np.zeros(len(session['time']))
    for i in range(len(session['time'])-window):
        time_diff = session['time'][i+window]-session['time'][i]
        lick_num = len(np.where((session['licks'] > i) & (session['licks'] < i+window))[0])
        lick_rate[i] = lick_num/time_diff
    
    session['lick_rate'] = lick_rate

    return session

def add_thlick_rate(session):
    # calculate lick rate in a sliding window
    window = 100
    lick_rate = np.zeros(len(session['time']))
    for i in range(len(session['time'])-window):
        time_diff = session['time'][i+window]-session['time'][i]
        lick_num = len(np.where((session['lick_threshold'] > i) & (session['lick_threshold'] < i+window))[0])
        lick_rate[i] = lick_num/time_diff
    
    session['thlick_rate'] = lick_rate

    return session

def add_lick_array(session):
    binary_array = np.zeros(len(session['position']), dtype=int)
    binary_array[session['licks']] = 1

    session['lick_array'] = binary_array.astype(bool)

    return session

def get_somerandombarcode():
    barcode_path = 'D:/ExampleMouse/TAA0000066/ses-018_date-20250403_protocol-t12/funcimg/20250403_140531__TAA0000066_00001.h5'
    barcode = read_h5_with_key(barcode_path, print_key=False)
    somerandombarcode = np.array(barcode) > 2.5

    return somerandombarcode

def get_somerandombarcode_df(start, length):
    somerandombarcode = get_somerandombarcode()
    length_converted = int(9000*length/45)
    somerandombarcode=somerandombarcode[start:start+length_converted]
    df = pd.DataFrame(somerandombarcode, index=np.linspace(0,len(somerandombarcode)/9000, len(somerandombarcode)))
    return df

def read_h5_with_key(signals_file, print_key=True, key='SyncTTL'):
    with h5py.File(signals_file, "r") as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        if print_key:
            print("Keys: %s" % f.keys())

        # If key is a dataset name, 
        # this gets the dataset values and returns as a list
        if key in f.keys():
            data = list(f[key])
        else:
            KeyError("This is not a valid key.")
    
    return data