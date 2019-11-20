# acars
Python code for downloading ACARS data. The data source is the MADIS server. The output format is the SPC text format. You'll probably need to modify some paths internally to get the script to work for you. 

## Usage
Invoke the script to download all data currently on the MADIS server. Subsequent invocations only download what's been updated since the last time you called the script.
```
$ python acars.py
```
