# acars
Python code for downloading ACARS data. The data source is the MADIS server. The output format is the SPC text format. 

## Usage
Invoke the script to download all data currently on the MADIS server. Subsequent invocations only download what's been updated since the last time you called the script.
```
$ python acars.py --output-path /path/to/output --work-path /path/to/working/directory --time-granularity 600
```
* `--output-path` specifies the path to the final script output. The default is the current directory.
* `--work-path` specifies the path to the working directory. The default is the current directory.
* `--time-granularity` specifies the time granularity (in seconds) for the valid times on the output profiles (not applied to the data within each profile). The default is 600 seconds (10 minutes).
