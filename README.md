# freqresp
Tool to get some insight about the frequency response of a device. Work in
progress!

todo:
- input filtering for the frequency response? (or otherwise more sophisticated way to measure)
- one frequency sweep 
- ...

Currently the frequency sweep is done in a such dummy way that the record/play is launched separately for every 
measured frequency. This causes overhead since a small part in the beginning of the recording is discarded to ensure 
stable conditions. 

### Configuration
All settings are placed in the file `settings.yaml`. The following settings can be tuned:

#### General properties
- sr: sample rate
- device: device to use (device name string)
- n\_waves: number of full waves to generate for each measurement point
- drop\_begin: length of the signal to drop out from the beginning of a
  measurement
- drop\_begin: length of the signal to drop out from the end of a measurement

#### Frequency sweep
Under `sweep` the following settings are available: 
- f0: beginning of the frequency sweep (default: 10)
- f1: end of the frequency sweep (default 10000)
- pid: points in decade 
- normalize: settings for incoming signal normalization. under this two alternatives can be given. by specifying 
`factor` a fixed division is done for every recorded signal. if frequency is set, a test signal is measured with the 
frequency, it is normalized to 1, and all the other measured frequencies are normalized with the same factor.   
- csv_filename: write the results in a csv file 

#### FFT settings
Under `fft` the following settings are available:
- fft_len: FFT length (default 4096)
- freq: frequency of the test signal
- csv_filename: write the results in a csv file

#### Example configuration
```
sr: 192000
device: "Scarlett 2i2 USB: Audio"

n_waves: 20
drop_begin: 0.1
drop_end: 0

sweep:
  f0: 1
  f1: 96000
  points_in_decade: 20

  normalize:
    #factor: 0.04
    frequency: 1000

  csv_filename: freqs.csv

fft:
  fft_len: 65536
  freq: 1000

  csv_filename: fft.csv
```
