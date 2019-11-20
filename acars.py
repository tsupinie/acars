
from __future__ import print_function

import numpy as np

from netCDF4 import Dataset

from datetime import datetime, timedelta

try:
    import urllib2 as urlreq
except ImportError:
    import urllib.request as urlreq

import re
import os
import glob
import gzip

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

from stdatmos import std_atmosphere_pres

_epoch = datetime(1970, 1, 1, 0)
_missing = -9999.0
_base_url = "https://madis-data.ncep.noaa.gov/madisPublic1/data/point/acars/netcdf/"
_work_path = "/home/tsupinie/acars"
_output_path = "/data/soundings/http/soundings/acars"
_time_granularity = 600 # seconds



def load_meta(meta_fname=("%s/airport_info.dat" % _work_path)):
    """
    load_meta

    Load in our database of airport codes.
    """
    meta_cols = ['code', 'id', 'synop', 'lat', 'lon', 'elev', 'name']
    meta_types = {'code': int, 'id': str, 'synop': int, 'lat': float, 'lon': float, 'elev': int, 'name': str}

    meta_airport = {}
    with open(meta_fname) as fmeta:
        for line in fmeta:
            line_dict = {col: val for col, val in zip(meta_cols, line.strip().split(None, 6)) }
        
            for col in line_dict.keys():
                line_dict[col] = meta_types[col](line_dict[col])
        
            code = line_dict.pop('code')
            meta_airport[code] = line_dict
    return meta_airport


def get_times(marker_path=("%s/markers" % _work_path)):
    """
    get_times

    This function figures out what times need to be downloaded from the server. Profiles are in files by hour, and the 
    files are updated as new data comes in, so if we download a file once, we won't necessarily get all the data. So 
    this also keeps track of which files were last accessed when. It does this by touching a file on disk (in 
    marker_path) when it last accessed a file. If the file on the server is newer, the function knows that the file 
    needs to be downloaded again. The marker files are deleted when they become older than the oldest file on the 
    server.
    """
    def touch(fname, times=None):
        with open(fname, 'a'):
            os.utime(fname, times)

    # Figure out the files and their update times from the MADIS server
    txt = urlreq.urlopen(_base_url).read()
    files = re.findall(">([\d]{8}_[\d]{4}).gz<", txt)
    update_times = re.findall("([\d]{2}-[\w]{3}-[\d]{4} [\d]{2}:[\d]{2})", txt)

    # If a MADIS server file is newer than the corresponding marker file, add it to the list of times to download
    times_to_dl = []
    for file_time_str, update_time_str in zip(files, update_times):
        marker_fname = "%s/%s.txt" % (marker_path, file_time_str)

        update_time = datetime.strptime(update_time_str, '%d-%b-%Y %H:%M')
        file_time = datetime.strptime(file_time_str, '%Y%m%d_%H%M')

        if not os.path.exists(marker_fname) or (os.path.exists(marker_fname) 
            and _epoch + timedelta(seconds=os.path.getmtime(marker_fname)) < update_time):
            times_to_dl.append(file_time)
            touch(marker_fname)

    # Check for old marker files and delete them if they're older than the oldest file on the MADIS server
    earliest_time = datetime.strptime(files[0], '%Y%m%d_%H%M')
    for fname in glob.glob("%s/*.txt" % marker_path):
        ftime = datetime.strptime(os.path.basename(fname), "%Y%m%d_%H%M.txt")
        if ftime < earliest_time:
            os.unlink(fname)

    return times_to_dl   


def dl_profiles(dt, fname):
    """
    dl_profiles

    Download netCDF files from the MADIS server. They're actually gzipped on the remote server, so unzip them in memory
    and write plain netCDF.
    """
    url = "%s/%s.gz" % (_base_url, dt.strftime('%Y%m%d_%H%M'))

    # Download the file and put the contents in a memory buffer to unzip
    sio = StringIO(urlreq.urlopen(url).read())
    gzf = gzip.GzipFile(fileobj=sio)

    # Write the unzipped data
    with open(fname, 'wb') as fnc:
        fnc.write(gzf.read())


def load_profiles(fname):
    """
    load_profiles

    Load netCDF files and put the data into data structures. The data for each variable are in one array containing
    every ob for every profile, so we have to split the arrays up into distinct profiles ourselves. We also enforce the
    time granularity on the profile valid times (not the obs within a single profile) here.
    """
    stdatm = std_atmosphere_pres()

    load_vars = ['temperature', 'dewpoint', 'soundingSecs', 'sounding_airport_id', 'latitude', 'longitude',
                 'windSpeed', 'windDir', 'altitude']

    nc = Dataset(fname)
    profile_data = {var: nc.variables[var][:] for var in load_vars}
    nc.close()

    # Split the profile arrays wherever the valid time changes. I guess this will mess up if two adjacent profiles
    #   happen to be valid at the same time, but I'm not sure that throws out too many good profiles.
    splits = np.where(np.diff(profile_data['soundingSecs']))[0] + 1

    profile_data_split = {var: np.split(prof_var, splits) for var, prof_var in profile_data.items()}
    profiles = [{} for prof in profile_data_split['soundingSecs']]
    for var, prof_var in profile_data_split.items():
        for prof, split_prof in zip(profiles, prof_var):
            prof[var] = split_prof

    for prof in profiles:
        # The sensed altitude variable is pressure. The heights are computed from pressure using a standard atmosphere,
        #   but then they don't give pressure for some reason, so we need to get the pressures back from heights
        prof['pressure'] = stdatm(prof['altitude'])

    # Enforce the granularity on the profile valid times. The granularity is set by _time_granularity above.
    unique_profiles = {}
    for profile in profiles:
        ap_code = profile['sounding_airport_id'][0]
        snd_time = profile['soundingSecs'][0]
    
        if type(ap_code) == type(np.ma.masked) or type(snd_time) == type(np.ma.masked):
            continue
        
        snd_time = _time_granularity * np.round(snd_time / _time_granularity)
        key = (snd_time, ap_code)
        if key not in unique_profiles or len(unique_profiles[key]['soundingSecs']) < len(profile['soundingSecs']):
            unique_profiles[key] = profile
        
    profiles = list(unique_profiles.values())
    return profiles


def output_profile(path, profile, meta_airport):
    """
    output_profile

    Write a single profile out to a file on disk in the SPC text format. Several quality control items are done in this
    method.
    1) If the airport code is missing, ignore the profile.
    2) If the profile's airport code doesn't exist in our database, ignore the profile.
    3) If the surface height is missing, ignore the profile. (This gives SHARPpy problems.)
    4) If the profile's lat/lon data are too far away from the airport it claims to be from, ignore the profile.
    5) If the profile has fewer than 3 data points, ignore the profile.
    6) If there's already a file with the same name on disk and this profile would produce a smaller file, ignore the
        profile.
    7) Remove obs that create duplicate height values or out-of-order pressure values (which also give SHARPpy issues).
    """

    # Check for a missing airport code
    code = profile['sounding_airport_id'][0]
    if type(code) == type(np.ma.masked):
        return

    # Check for an airport code that's not in the database
    try:
        apid = meta_airport[code]['id']
    except KeyError:
        print("Skipping profile: unknown airport code '%d'" % code)
        return

    # Check for a missing surface height
    if type(profile['altitude'][0]) == type(np.ma.masked):
        print("Skipping profile: surface height at '%s' is qc'ed" % apid)
        return

    # Check for distance from the claimed source airport
    ap_lat = meta_airport[code]['lat']
    ap_lon = meta_airport[code]['lon']

    dist = np.hypot(ap_lat - profile['latitude'], ap_lon - profile['longitude'])
    if dist.min() > 1:
        print("Skipping profile: claims to be from '%s', but data are too far away" % apid)
        return

    # Get a datetime object with the profile time
    dt = _epoch + timedelta(seconds=float(profile['soundingSecs'][0]))

    # Fill any qc'ed values with the missing value
    pres_prof = profile['pressure'].filled(_missing)
    hght_prof = profile['altitude'].filled(_missing)
    tmpk_prof = profile['temperature'].filled(_missing)
    dwpk_prof = profile['dewpoint'].filled(_missing)
    wdir_prof = profile['windDir'].filled(_missing)
    wspd_prof = profile['windSpeed'].filled(_missing)
    
    # Sort by height
    sort_idxs = np.argsort(hght_prof)
    pres_prof = pres_prof[sort_idxs]
    hght_prof = hght_prof[sort_idxs]
    tmpk_prof = tmpk_prof[sort_idxs]
    dwpk_prof = dwpk_prof[sort_idxs]
    wdir_prof = wdir_prof[sort_idxs]
    wspd_prof = wspd_prof[sort_idxs]
    
    # Remove duplicate heights or out-of-order pressures
    bad_hghts = np.append(False, np.isclose(np.diff(hght_prof), 0))
    bad_press = np.append(False, np.diff(pres_prof) >= 0)
    
    keep = np.where(~(bad_hghts | bad_press))
    pres_prof = pres_prof[keep]
    hght_prof = hght_prof[keep]
    tmpk_prof = tmpk_prof[keep]
    dwpk_prof = dwpk_prof[keep]
    wdir_prof = wdir_prof[keep]
    wspd_prof = wspd_prof[keep]
    
    # Check for number of data points
    if len(hght_prof) < 3:
        return

    # Create the text output
    snd_lines = [
        "%TITLE%", 
        "%s   %s" % (apid.rjust(5), dt.strftime('%y%m%d/%H%M')),
        "",
        "   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD",
        "-------------------------------------------------------------------",
        "%RAW%"
    ]
    
    for pres, hght, tmpk, dwpk, wdir, wspd in zip(pres_prof, hght_prof, tmpk_prof, dwpk_prof, wdir_prof, wspd_prof):
        ob_str = "% 8.2f, % 9.2f, % 9.2f, % 9.2f, % 9.2f, % 9.2f" % (pres / 100., hght, tmpk - 273.15, dwpk - 273.15, 
                                                                     wdir, wspd * 1.94)
        snd_lines.append(ob_str)
    snd_lines.append('%END%')

    # Construct the file name (using the time granularity)
    dt_sec = round((dt - _epoch).total_seconds() / _time_granularity) * _time_granularity
    dt_round = _epoch + timedelta(seconds=dt_sec)
    tree_path = "%s/%s" % (path, dt_round.strftime("%Y/%m/%d/%H"))
    fname = "%s/%s_%s.txt" % (tree_path, apid, dt_round.strftime("%H%M"))

    try:
        os.makedirs(tree_path)
    except OSError:
        pass    

    snd_str = "\n".join(snd_lines)
    exist_size = os.path.getsize(fname) if os.path.exists(fname) else 0

    # Check for a smaller file that already exists. The idea is to avoid writing over a "good" file with a "bad" file,
    #   where file size is used as a proxy for "goodness". This may not be the best proxy, though.
    if len(snd_str) < exist_size:
        print("Skipping profile: refusing to overwrite existing '%s' with a smaller file" % fname)
    else:
        with open(fname, 'w') as fsnd:
            fsnd.write(snd_str)


def output_profiles(path, profiles, meta):
    """
    output_profiles

    Loop and dump out every profile. Not entirely sure this needs to be a separate function, but whatever.
    """
    for profile in profiles:
        output_profile(path, profile, meta)


def main():
    meta = load_meta()

    print("Retrieving profile times ...")
    dts = get_times()

    for dt in dts:
        print("New profiles for %s" % dt.strftime("%H%M UTC %d %b"))

        fname = "%s/%s.nc" % (_work_path, dt.strftime("%Y%m%d_%H%M"))

        print("Downloading profiles ...")
        dl_profiles(dt, fname)

        print("Parsing profiles ...")
        profiles = load_profiles(fname)

        print("Dumping files ...")
        output_profiles(_output_path, profiles, meta)

        os.unlink(fname)

    print("Done!")

if __name__ == "__main__":
    main()
