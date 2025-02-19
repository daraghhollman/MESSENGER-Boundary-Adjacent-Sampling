"""
A script to save sections of time either side of magnetopause crossing intervals Loads the philpott boundaries list, and loads and saves magnetosphere and magnetosheath time series data in a fixed length window before/after each bow shock crossign interval, along with any other raw data such as ephemeris,lcoal time, latitude, etc.
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

magnetopauses = crossings.loc[
    (crossings["Type"] == "MP_OUT") | (crossings["Type"] == "MP_IN")
]

def Get_Magnetosphere_Sample(row):
    """
    For a paricular solar wind sample (or 'event'),
    return a dictionary of all the relevant raw properties.
    """

    current_index = crossings.index.get_loc(row.name)
    
    # Find out if magnetosphere is before or after based on the type of the current crossing
    if row["Type"] == "MP_OUT":
        # Magnetosphere is before the crossing
        sample_start = row["Start Time"] - sample_length
        sample_end = row["Start Time"]

        if current_index != 0:
            # we need to enure we don't go past another boundary
            previous_crossing = crossings.iloc[current_index - 1]
            if sample_start < previous_crossing["End Time"]:
                sample_start = previous_crossing["End Time"]

    elif row["Type"] == "MP_IN":
        # Solar wind is before the crossing
        sample_start = row["End Time"]
        sample_end = row["End Time"] + sample_length

        if current_index < len(magnetopauses) - 1:
            next_crossing = crossings.iloc[current_index + 1]
            if sample_end > next_crossing["Start Time"]:
                sample_end = next_crossing["Start Time"]

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
    
    # Find out if magnetosheath is before or after based on the type of the current crossing
    if row["Type"] == "MP_OUT":
        # Magnetosheath is after the crossing
        sample_start = row["End Time"]
        sample_end = row["End Time"] + sample_length

        if current_index < len(magnetopauses) - 1:
            # we need to enure we don't go past another boundary
            next_crossing = crossings.iloc[current_index + 1]
            if sample_end > next_crossing["Start Time"]:
                sample_end = next_crossing["Start Time"]

    elif row["Type"] == "MP_IN":
        # Magnetosheath is before the crossing
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
magnetosphere_samples = []

# Iterrate through the crossings
process_items = [row for _, row in magnetopauses.iterrows()]
with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
    for result in tqdm(pool.imap(Get_Magnetosphere_Sample, process_items), total=len(process_items)):

        if result is not None:
            # Add row dictionary to list
            magnetosphere_samples.append(result)


# Create dataframe from solar wind samples
magnetosphere_samples = pd.DataFrame(magnetosphere_samples)

print("")

magnetosphere_samples.to_csv(
    f"/home/daraghhollman/Main/Work/mercury/DataSets/magnetopause/magnetosphere_sample_{int(sample_length.total_seconds() / 60)}_mins.csv"
)

"""
# Create empty list of dictionaries
magnetosheath_samples = []

# Iterrate through the crossings
process_items = [row for _, row in magnetopauses.iterrows()]
with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:
    for result in tqdm(pool.imap(Get_Magnetosheath_Sample, process_items), total=len(process_items)):

        if result is not None:
            # Add row dictionary to list
            magnetosheath_samples.append(result)


# Create dataframe from solar wind samples
magnetosheath_samples = pd.DataFrame(magnetosheath_samples)

print("")

magnetosheath_samples.to_csv(
    f"/home/daraghhollman/Main/Work/mercury/DataSets/magnetopause/magnetosheath_sample_{int(sample_length.total_seconds() / 60)}_mins.csv"
)
"""
