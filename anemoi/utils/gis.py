import anemoi as an
import pandas as pd
import numpy as np

def distances_to_point(lat_point, lon_point, lats, lons):
    lat_point = np.deg2rad(lat_point)
    lon_point = np.deg2rad(lon_point)
    avg_earth_radius = 6373  # in km
    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)
    lat = lat_point - lats
    lon = lon_point - lons
    d = np.sin(lat * 0.5)**2 + np.cos(lat_point) * np.cos(lats) * np.sin(lon * 0.5)**2
    dist = 2 * avg_earth_radius * np.arcsin(np.sqrt(d))
    return dist
