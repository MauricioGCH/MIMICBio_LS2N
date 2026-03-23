##Main functions for processing and modeling
import os
import numpy as np
from matplotlib import pyplot as plt

def read_multichannel_bin_data(filepath, quantum=0.0050863, skip_s=0, length_s=0):

    

    # filepath : path to the .bin file
    # quantum  : acquisition quantum -- scaling factor that converts the raw binary values (signed 16-bit integers, >i2) into physical units (likely volts or microvolts).
        #The file contains 2-byte signed integers representing the raw ADC (Analog-to-Digital Converter) counts from the acquisition hardware.

    # skip_s   : Time (in seconds) to skip (ie. ignore) at the beginning of the file
    # length_s : Time (in seconds) of data to read (leave 0 to read whole file)
    
    Fs               = 10000 # Hz
    N_channels       = 64 # Channnels -- electrodes, each channel is an electrode in the MEA
    data_width_bytes = 2  # 2 bytes per sample (>i2 : signed 2-byte integers, big-endian)
    filesize = os.path.getsize(filepath)
    
    # At every sampling time (1/Fs), 64 samples (1 for every channel) are recorded
    # They are recorded in a 128B chunk (2B per channel)
    #   @0x0000 <Sample0/Ch0><Sample0/Ch1><Sample0/Ch2> ... <Sample0/Ch63> (@t=0 s)
    #   @0x0080 <Sample1/Ch0><Sample1/Ch1><Sample1/Ch2> ... <Sample1/Ch63> (@t=0.1 ms)
    #   @0x0100 <Sample2/Ch0><Sample2/Ch1><Sample2/Ch2> ... <Sample2/Ch63> (@t=0.2 ms)
    #                                                   ...
    #   @0x---- <SampleN/Ch0><SampleN/Ch1><SampleN/Ch2> ... <SampleN/Ch63> (@t=N*0.1e-3 s)

    ''' Number of chunks in requested data '''
    skip_chunks   = int(skip_s   * Fs) # Number of 128b chunks to skip - converts seconds to number of samples/time points
    length_chunks = int(length_s * Fs) # Number of 128b chunks to read - converts seconds to number of samples/time points
    
    """Each chunk = one time point × 64 channels × 2 bytes = 128 bytes
        So:
            skip_bytes = 20,000 × 128 = 2,560,000 bytes

            length_bytes = 30,000 × 128 = 3,840,000 bytes"""
    
    
    ''' Handle cases where skip_s is unspecified(0) and/or length_s is unspecified (0) '''
    if (length_chunks == 0) and (skip_chunks == 0):
        # If no length or skip specified, read the whole file
        length_chunks = int(filesize / (N_channels * data_width_bytes))
        
    if (length_chunks == 0) and (skip_chunks > 0):
        # If no length specified but non-zero skip, read until end of file
        length_chunks = int(filesize / (N_channels * data_width_bytes) - skip_chunks)
    
    ''' Number of bytes in requested data '''
    skip_bytes   = int(skip_chunks   * N_channels * data_width_bytes)
    length_bytes = int(length_chunks * N_channels * data_width_bytes)
    
    print(f"Reading {length_bytes}B from offset {skip_bytes} (0x{skip_bytes:X})")
    
    ''' Read file'''
    with open(filepath, 'rb') as fid:
        if skip_bytes > 0:
            fid.seek(skip_bytes)
        data = np.fromfile(fid, dtype='>i2', count=length_bytes // data_width_bytes)
    print("20 first points of data (binary values) : ")
    for x in data[:20]:
        print(x, end=',')
    print()
    
    data = data.reshape(length_chunks, N_channels)
    print("All done.")
    
    return data * quantum


def read_one_channel(filepath, channel_idx, quantum=0.0050863, skip_s=0, length_s=0):
    Fs = 10000
    N_channels = 64
    dtype = '>i2'

    filesize = os.path.getsize(filepath)
    total_samples = filesize // (2 * N_channels)

    skip_samples = int(skip_s * Fs)
    if length_s == 0:
        length_samples = total_samples - skip_samples
    else:
        length_samples = int(length_s * Fs)

    # Memmap: NO carga todo en memoria
    data = np.memmap(filepath, dtype=dtype, mode='r')

    # reshape lógico (sin copiar)
    data = data.reshape(-1, N_channels)

    # slicing directo del canal
    channel_data = data[skip_samples:skip_samples+length_samples, channel_idx]

    return channel_data * quantum


def read_multichannel_bin_data_mod(filepath, skip_s=0, length_s=0):

    # The first data the sent us has a  total length of 5700 seconds with a Fs of 10000Hz
    # filepath : path to the .bin file
    # quantum  : acquisition quantum -- scaling factor that converts the raw binary values (signed 16-bit integers, >i2) into physical units (likely volts or microvolts).
        #The file contains 2-byte signed integers representing the raw ADC (Analog-to-Digital Converter) counts from the acquisition hardware.

    # skip_s   : Time (in seconds) to skip (ie. ignore) at the beginning of the file
    # length_s : Time (in seconds) of data to read (leave 0 to read whole file)
    
    Fs               = 10000 # Hz
    N_channels       = 64 # Channnels -- electrodes, each channel is an electrode in the MEA
    data_width_bytes = 2  # 2 bytes per sample (>i2 : signed 2-byte integers, big-endian)
    filesize = os.path.getsize(filepath)



    total_samples = filesize // (N_channels * data_width_bytes)
    real_duration = total_samples / Fs

    print(f"Real samples: {total_samples}")
    print(f"Real duration (s): {real_duration}")
    
    # At every sampling time (1/Fs), 64 samples (1 for every channel) are recorded
    # They are recorded in a 128B chunk (2B per channel)
    #   @0x0000 <Sample0/Ch0><Sample0/Ch1><Sample0/Ch2> ... <Sample0/Ch63> (@t=0 s)
    #   @0x0080 <Sample1/Ch0><Sample1/Ch1><Sample1/Ch2> ... <Sample1/Ch63> (@t=0.1 ms)
    #   @0x0100 <Sample2/Ch0><Sample2/Ch1><Sample2/Ch2> ... <Sample2/Ch63> (@t=0.2 ms)
    #                                                   ...
    #   @0x---- <SampleN/Ch0><SampleN/Ch1><SampleN/Ch2> ... <SampleN/Ch63> (@t=N*0.1e-3 s)

    ''' Number of chunks in requested data '''
    skip_chunks   = int(skip_s   * Fs) # Number of 128b chunks to skip - converts seconds to number of samples/time points
    length_chunks = int(length_s * Fs) # Number of 128b chunks to read - converts seconds to number of samples/time points
    
    """Each chunk = one time point × 64 channels × 2 bytes = 128 bytes
        So:
            skip_bytes = 20,000 × 128 = 2,560,000 bytes

            length_bytes = 30,000 × 128 = 3,840,000 bytes"""
    
    
    max_chunks = filesize // (N_channels * data_width_bytes)
    print(max_chunks)
    
    if length_chunks > max_chunks - skip_chunks:
        print("⚠️ Requested length exceeds file size. Clipping...")
        length_chunks = max_chunks - skip_chunks

    ''' Handle cases where skip_s is unspecified(0) and/or length_s is unspecified (0) '''
    if (length_chunks == 0) and (skip_chunks == 0):
        # If no length or skip specified, read the whole file
        length_chunks = int(filesize / (N_channels * data_width_bytes))
        
    if (length_chunks == 0) and (skip_chunks > 0):
        # If no length specified but non-zero skip, read until end of file
        length_chunks = int(filesize / (N_channels * data_width_bytes) - skip_chunks)
    
    ''' Number of bytes in requested data '''
    skip_bytes   = int(skip_chunks   * N_channels * data_width_bytes)
    length_bytes = int(length_chunks * N_channels * data_width_bytes)
    
    print(f"Reading {length_bytes}B from offset {skip_bytes} (0x{skip_bytes:X})")
    
    ''' Read file'''
    with open(filepath, 'rb') as fid:
        if skip_bytes > 0:
            fid.seek(skip_bytes)
        data = np.fromfile(fid, dtype='>i2', count=length_bytes // data_width_bytes)
    print("20 first points of data (binary values) : ")
    for x in data[:20]:
        print(x, end=',')
    print()
    
    data = data.reshape(length_chunks, N_channels)
    print("All done.")

    #complete = data * quantum
    
    return data


