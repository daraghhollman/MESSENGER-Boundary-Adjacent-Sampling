"""
A script to save sections of time either side of bow shock crossing intervals Loads the philpott boundaries list, and loads and saves solar wind and magnetosheath time series data in a fixed length window before/after each bow shock crossign interval, along with any other raw data such as ephemeris,lcoal time, latitude, etc.
"""

import datetime as dt
import multiprocessing

import numpy as np
import pandas as pd
import spiceypy as spice
from hermpy import boundaries, mag, trajectory, utils
from tqdm import tqdm

# How much data to attempt to save either side of the boundary.
sample_length = dt.timedelta(minutes=10)

# Load Philpott+ (2020) crossings
crossings = boundaries.Load_Crossings(
    utils.User.CROSSING_LISTS["Philpott"]
)

bow_shocks = crossings.loc[
    (crossings["Type"] == "BS_OUT") | (crossings["Type"] == "BS_IN")
]

def Get_Solar_Wind_Sample(row):
    """
    For a paricular solar wind sample (or 'event'),
    return a dictionary of all the relevant raw properties.
    """

    current_index = crossings.index.get_loc(row.name)
    
    # Find out if solar wind is before or after based on the type of the current crossing
    if row["Type"] == "BS_OUT":
        # Solar wind is after the crossing
        sample_start = row["End Time"]
        sample_end = row["End Time"] + sample_length

        if current_index < len(bow_shocks) - 1:
            # we need to enure we don't go past another boundary
            next_crossing = crossings.iloc[current_index + 1]
            if sample_end > next_crossing["Start Time"]:
                sample_end = next_crossing["Start Time"]

    elif row["Type"] == "BS_IN":
        # Solar wind is before the crossing
        sample_start = row["Start Time"] - sample_length
        sample_end = row["Start Time"]

        if current_index != 0:
            previous_crossing = crossings.iloc[current_index - 1]
            if sample_start < previous_crossing["End Time"]:
                sample_start = previous_crossing["End Time"]

    else:
        return None

    # Load sample data:
    sample = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG"], sample_start, sample_end, strip=True, aberrate=True
    )

    if len(sample['|B|']) - 600 > 2:
        raise ValueError(f"Samples larger than to 600: N={len(sample['|B|'])}\nSample Start: {sample_start}\nSample End: {sample_end}")

    # Remove data spikes
    mag.Remove_Spikes(sample)

    sample_middle = sample.iloc[round(len(sample) / 2)]
    sample_middle_position = np.array(
        [
            sample_middle["X MSM' (radii)"],
            sample_middle["Y MSM' (radii)"],
            sample_middle["Z MSM' (radii)"],
        ]
    )

    local_time = trajectory.Local_Time(sample_middle_position)
    latitude = trajectory.Latitude(sample_middle_position)
    magnetic_latitude = trajectory.Magnetic_Latitude(sample_middle_position)

    with spice.KernelPool(utils.User.METAKERNEL):
        et = spice.str2et(sample_middle["date"].strftime("%Y-%m-%d %H:%M:%S"))
        mercury_position, _ = spice.spkpos("MERCURY", et, "J2000", "NONE", "SUN")

        heliocentric_distance = np.sqrt(
            mercury_position[0] ** 2
            + mercury_position[1] ** 2
            + mercury_position[2] ** 2
        )
        heliocentric_distance = utils.Constants.KM_TO_AU(heliocentric_distance)


    return {
        # Time identifiers
        "Crossing Start": row["Start Time"],
        "Crossing End": row["End Time"],
        "Sample Start": sample_start,
        "Sample End": sample_end,
        # Data sample itself
        "UTC": sample["date"].tolist(),
        "|B|": sample["|B|"].tolist(),
        "Bx": sample["Bx"].tolist(),
        "By": sample["By"].tolist(),
        "Bz": sample["Bz"].tolist(),
        # The median local time of the sample
        "Local Time (hrs)": local_time,
        # The median latitude of the sample
        "Latitude (deg.)": latitude,
        # The median magnetic latitude of the sample
        "Magnetic Latitude (deg.)": magnetic_latitude,
        # Median heliocentric distance of the sample
        "Heliocentric Distance (AU)": heliocentric_distance,
        # Median Spacecraft position
        "X MSM' (radii)": sample_middle_position[0],
        "Y MSM' (radii)": sample_middle_position[1],
        "Z MSM' (radii)": sample_middle_position[2],
    }

def Get_Magnetosheath_Sample(row):
    """
    For a paricular solar wind sample (or 'event'),
    return a dictionary of all the relevant raw properties.
    """

    current_index = crossings.index.get_loc(row.name)
    
    # Find out if solar wind is before or after based on the type of the current crossing
    if row["Type"] == "BS_IN":
        # Solar wind is after the crossing
        sample_start = row["End Time"]
        sample_end = row["End Time"] + sample_length

        if current_index < len(bow_shocks) - 1:
            # we need to enure we don't go past another boundary
            next_crossing = crossings.iloc[current_index + 1]
            if sample_end > next_crossing["Start Time"]:
                sample_end = next_crossing["Start Time"]

    elif row["Type"] == "BS_OUT":
        # Solar wind is before the crossing
        sample_start = row["Start Time"] - sample_length
        sample_end = row["Start Time"]

        if current_index != 0:
            previous_crossing = crossings.iloc[current_index - 1]
            if sample_start < previous_crossing["End Time"]:
                sample_start = previous_crossing["End Time"]

    else:
        return None

    # Load sample data:
    sample = mag.Load_Between_Dates(
        utils.User.DATA_DIRECTORIES["MAG"], sample_start, sample_end, strip=True, aberrate=True
    )

    if len(sample['|B|']) - 600 > 2:
        raise ValueError(f"Samples larger than to 600: N={len(sample['|B|'])}\nSample Start: {sample_start}\nSample End: {sample_end}")

    # Remove data spikes
    mag.Remove_Spikes(sample)

    sample_middle = sample.iloc[round(len(sample) / 2)]
    sample_middle_position = np.array(
        [
            sample_middle["X MSM' (radii)"],
            sample_middle["Y MSM' (radii)"],
            sample_middle["Z MSM' (radii)"],
        ]
    )

    local_time = trajectory.Local_Time(sample_middle_position)
    latitude = trajectory.Latitude(sample_middle_position)
    magnetic_latitude = trajectory.Magnetic_Latitude(sample_middle_position)

    with spice.KernelPool(utils.User.METAKERNEL):
        et = spice.str2et(sample_middle["date"].strftime("%Y-%m-%d %H:%M:%S"))
        mercury_position, _ = spice.spkpos("MERCURY", et, "J2000", "NONE", "SUN")

        heliocentric_distance = np.sqrt(
            mercury_position[0] ** 2
            + mercury_position[1] ** 2
            + mercury_position[2] ** 2
        )
        heliocentric_distance = utils.Constants.KM_TO_AU(heliocentric_distance)


    return {
        # Time identifiers
        "Crossing Start": row["Start Time"],
        "Crossing End": row["End Time"],
        "Sample Start": sample_start,
        "Sample End": sample_end,
        # Data sample itself
        "UTC": sample["date"].tolist(),
        "|B|": sample["|B|"].tolist(),
        "Bx": sample["Bx"].tolist(),
        "By": sample["By"].tolist(),
        "Bz": sample["Bz"].tolist(),
        # The median local time of the sample
        "Local Time (hrs)": local_time,
        # The median latitude of the sample
        "Latitude (deg.)": latitude,
        # The median magnetic latitude of the sample
        "Magnetic Latitude (deg.)": magnetic_latitude,
        # Median heliocentric distance of the sample
        "Heliocentric Distance (AU)": heliocentric_distance,
        # Median Spacecraft position
        "X MSM' (radii)": sample_middle_position[0],
        "Y MSM' (radii)": sample_middle_position[1],
        "Z MSM' (radii)": sample_middle_position[2],
    }

# Create empty list of dictionaries
solar_wind_samples = []

# Iterrate through the crossings
process_items = [row for _, row in bow_shocks.iterrows()]
with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
    for result in tqdm(pool.imap(Get_Solar_Wind_Sample, process_items), total=len(process_items)):

        if result is not None:
            # Add row dictionary to list
            solar_wind_samples.append(result)


# Create dataframe from solar wind samples
solar_wind_samples = pd.DataFrame(solar_wind_samples)

print("")

solar_wind_samples.to_csv(
    f"/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock/solar_wind_sample_{int(sample_length.total_seconds() / 61)}_mins.csv"
)

# Create empty list of dictionaries
magnetosheath_samples = []

# Iterrate through the crossings
process_items = [row for _, row in bow_shocks.iterrows()]
with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
    for result in tqdm(pool.imap(Get_Magnetosheath_Sample, process_items), total=len(process_items)):

        if result is not None:
            # Add row dictionary to list
            magnetosheath_samples.append(result)


# Create dataframe from solar wind samples
magnetosheath_samples = pd.DataFrame(magnetosheath_samples)

print("")

magnetosheath_samples.to_csv(
    f"/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock/magnetosheath_sample_{int(sample_length.total_seconds() / 60)}_mins.csv"
)
