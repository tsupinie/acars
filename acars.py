
from __future__ import print_function

import numpy as np

from netCDF4 import Dataset

from datetime import datetime, timedelta

try:
    import urllib2 as urlreq
    from urllib2 import HTTPError
except ImportError:
    import urllib.request as urlreq
    from urllib.error import HTTPError

import re
import os
import glob
import gzip
import argparse

try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO

from stdatmos import std_atmosphere_pres

_epoch = datetime(1970, 1, 1, 0)
_missing = -9999.0
_base_url = "https://madis-data.ncep.noaa.gov/madisPublic1/data/point/acars/netcdf"

_stdatm = std_atmosphere_pres()

_meta_path = os.path.dirname(__file__)
if _meta_path == "":
    _meta_path = "."

class ACARSProfile(object):
    def __init__(self, apid, apcode, valid_dt, **prof_vars):
        self.apid = apid
        self.apcode = apcode
        self.dt = valid_dt
        self.prof_vars = prof_vars

        self.prof_vars['pressure'] = _stdatm(self.prof_vars['altitude'])

        self._sort()

    def apply_qc(self, meta_airport):
        # Check for a missing surface height
        if type(self.prof_vars['altitude'][0]) == type(np.ma.masked):
            print("Skipping profile: surface height at '%s' is qc'ed" % self.apid)
            return False

        # Check for distance from the claimed source airport
        ap_lat = meta_airport[self.apcode]['lat']
        ap_lon = meta_airport[self.apcode]['lon']

        dist = np.hypot(ap_lat - self.prof_vars['latitude'], ap_lon - self.prof_vars['longitude'])
        if dist.min() > 1:
            print("Skipping profile: claims to be from '%s', but data are too far away" % self.apid)
            return False
      
        # Remove duplicate heights or out-of-order pressures
        bad_hghts = np.append(False, np.isclose(np.diff(self.prof_vars['altitude']), 0))
        bad_press = np.append(False, np.diff(self.prof_vars['pressure']) >= 0)
        keep = np.where(~(bad_hghts | bad_press))
        
        for var, val in self.prof_vars.items():
            self.prof_vars[var] = val[keep]
 
        # Check for number of data points
        if len(self.prof_vars['altitude']) < 3:
            return False
        
        return True

    def append(self, other):
        if self.apid != other.apid or self.dt != other.dt:
            raise ValueError("Profile id and time in append() must be the same")

        for var, vals in other.prof_vars.items():
            self.prof_vars[var] = np.ma.append(self.prof_vars[var], vals)

        self._sort()

    def _sort(self):
        sort_idxs = np.argsort(self.prof_vars['altitude'])
        for var, vals in self.prof_vars.items():
            self.prof_vars[var] = vals[sort_idxs]

    def to_spc(self, path, time_granularity):
        # Fill any qc'ed values with the missing value
        pres_prof = (self.prof_vars['pressure'] / 100.).filled(_missing)
        hght_prof = self.prof_vars['altitude'].filled(_missing)
        tmpk_prof = (self.prof_vars['temperature'] - 273.15).filled(_missing)
        dwpk_prof = (self.prof_vars['dewpoint'] - 273.15).filled(_missing)
        wdir_prof = self.prof_vars['windDir'].filled(_missing)
        wspd_prof = (self.prof_vars['windSpeed'] * 1.94).filled(_missing)

        # Create the text output
        snd_lines = [
            "%TITLE%",
            "%s   %s" % (self.apid.rjust(5), self.dt.strftime('%y%m%d/%H%M')),
            "",
            "   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD",
            "-------------------------------------------------------------------",
            "%RAW%"
        ]

        for pres, hght, tmpk, dwpk, wdir, wspd in zip(pres_prof, hght_prof, tmpk_prof, dwpk_prof, wdir_prof, wspd_prof):
            ob_str = "% 8.2f, % 9.2f, % 9.2f, % 9.2f, % 9.2f, % 9.2f" % (pres, hght, tmpk, dwpk, wdir, wspd)
            snd_lines.append(ob_str)
        snd_lines.append('%END%')
        snd_str = "\n".join(snd_lines)

        # Construct the file name (using the time granularity)
        dt_sec = round((self.dt - _epoch).total_seconds() / time_granularity) * time_granularity
        dt_round = _epoch + timedelta(seconds=dt_sec)
        tree_path = "%s/%s" % (path, dt_round.strftime("%Y/%m/%d/%H"))
        fname = "%s/%s_%s.txt" % (tree_path, self.apid, dt_round.strftime("%H%M"))

        try:
            os.makedirs(tree_path)
        except OSError:
            pass    

        exist_size = os.path.getsize(fname) if os.path.exists(fname) else 0

        # Check for a smaller file that already exists. The idea is to avoid writing over a "good" file with a "bad" file,
        #   where file size is used as a proxy for "goodness". This may not be the best proxy, though.
        if len(snd_str) < exist_size:
            print("Skipping profile: refusing to overwrite existing '%s' with a smaller file" % fname)
        else:
            with open(fname, 'w') as fsnd:
                fsnd.write(snd_str)


def load_profiles(fname, supplemental, meta_airport):
    def _load_profiles(fname):
        load_vars = ['temperature', 'dewpoint', 'soundingSecs', 'sounding_airport_id', 'latitude', 'longitude',
                     'windSpeed', 'windDir', 'altitude']

        nc = Dataset(fname)
        profile_data = {var: nc.variables[var][:] for var in load_vars}
        nc.close()

        # Split the profile arrays wherever the valid time changes. I guess this will mess up if two adjacent profiles
        #   happen to be valid at the same time, but I'm not sure that throws out too many good profiles.
        splits = np.where(np.diff(profile_data['soundingSecs']))[0] + 1

        profile_data_split = {var: np.split(prof_var, splits) for var, prof_var in profile_data.items()}

        profiles = []
        for vals in zip(*(profile_data_split[var] for var in load_vars)):
            val_dict = dict(zip(load_vars, vals))

            if type(val_dict['soundingSecs'][0]) == type(np.ma.masked):
                continue

            time = val_dict.pop('soundingSecs')
            apid = val_dict.pop('sounding_airport_id')

            prof_dt = _epoch + timedelta(seconds=time[0])
            try:
                prof_id = meta_airport[apid[0]]['id']
            except KeyError:
                continue

            prof = ACARSProfile(prof_id, apid[0], prof_dt, **val_dict)
            profiles.append(prof)

        return profiles

    profiles = _load_profiles(fname)

    for fname in supplemental:
        supp_profiles = _load_profiles(fname)

        for supp_prof in supp_profiles:
            for prof in profiles:
                if supp_prof.apid == prof.apid and supp_prof.dt == prof.dt:
                    prof.append(supp_prof)
                    break

    return profiles


def load_meta(meta_fname=("%s/airport_info.dat" % _meta_path)):
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


def get_times(work_path):
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

    marker_path = "%s/markers" % work_path

    # Figure out the files and their update times from the MADIS server
    txt = urlreq.urlopen(_base_url).read().decode('utf-8')
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


def dl_profiles(path, dt):
    """
    dl_profiles

    Download netCDF files from the MADIS server. They're actually gzipped on the remote server, so unzip them in memory
    and write plain netCDF.
    """
    url = "%s/%s.gz" % (_base_url, dt.strftime('%Y%m%d_%H%M'))

    # Download the file and put the contents in a memory buffer to unzip
    bio = BytesIO(urlreq.urlopen(url).read())
    gzf = gzip.GzipFile(fileobj=bio)

    fname = "%s/%s.nc" % (path, dt.strftime("%Y%m%d_%H%M"))

    # Write the unzipped data
    with open(fname, 'wb') as fnc:
        fnc.write(gzf.read())

    return fname


def apply_granularity(profiles, granularity):
    unique_profiles = {}
    for profile in profiles:
        ap_code = profile.apid
        snd_time = (profile.dt - _epoch).total_seconds()

        snd_time = granularity * np.round(snd_time / granularity)
        key = (snd_time, ap_code)
        if key not in unique_profiles or len(unique_profiles[key].prof_vars['altitude']) < len(profile.prof_vars['altitude']):
            unique_profiles[key] = profile
        
    return list(unique_profiles.values())


def output_profiles(path, profiles, time_granularity):
    """
    output_profiles

    Loop and dump out every profile. Not entirely sure this needs to be a separate function, but whatever.
    """
    for profile in profiles:
        profile.to_spc(path, time_granularity)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work-path', dest="work_path", default=".")
    ap.add_argument('--output-path', dest="output_path", default=".")
    ap.add_argument('--time-granularity', dest='time_gran', type=int, default=600)

    args = ap.parse_args()

    meta = load_meta()

    print("Retrieving profile times ...")
    dts = get_times(args.work_path)

    dt_fnames = {}
    print("Downloading profiles ...")
    for dt in dts:
        dt_fnames[dt] = dl_profiles(args.work_path, dt)

        dt_prev = dt - timedelta(hours=1)
        dt_next = dt + timedelta(hours=1)

        if dt_prev not in dts:
            try:
                dt_fnames[dt_prev] = dl_profiles(args.work_path, dt_prev)
            except HTTPError:
                print("Could not find file for %s" % dt_prev.strftime("%d %b %Y %H%M UTC"))

        if dt_next not in dts:
            try:
                dt_fnames[dt_next] = dl_profiles(args.work_path, dt_next)
            except HTTPError:
                print("Could not find file for %s" % dt_next.strftime("%d %b %Y %H%M UTC"))

    for dt in dts:
        print("New profiles for %s" % dt.strftime("%H%M UTC %d %b"))

        dt_prev = dt - timedelta(hours=1)
        dt_next = dt + timedelta(hours=1)

        print("Parsing profiles ...")
        supplemental = []
        if dt_prev in dt_fnames:
            supplemental.append(dt_fnames[dt_prev])
        if dt_next in dt_fnames:
            supplemental.append(dt_fnames[dt_next])
        profiles = load_profiles(dt_fnames[dt], supplemental, meta)

        profiles_qc = [ profile for profile in profiles if profile.apply_qc(meta) ]

        profiles_gran = apply_granularity(profiles_qc, args.time_gran)

        print("Dumping files ...")
        output_profiles(args.output_path, profiles_gran, args.time_gran)

    for fname in dt_fnames.values():
        os.unlink(fname)

    print("Done!")

if __name__ == "__main__":
    main()
