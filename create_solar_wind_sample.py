"""
Script to load the philpott boundaries list, and load and save the solar wind time series data in a fixed length window before/after the boundary, along with any other raw data such as ephemeris, local time, latitude, magnetic latitude
"""

import datetime as dt
import multiprocessing

import hermpy.boundary_crossings as boundaries
import hermpy.mag as mag
import hermpy.trajectory as traj
from hermpy.utils import User, Constants
import numpy as np
import pandas as pd
import spiceypy as spice
from tqdm import tqdm


root_dir = "/home/daraghhollman/Main/data/mercury/messenger/mag/avg_1_second/"
sample_length = dt.timedelta(minutes=10)


# Load Philpott+ (2020) crossings
crossings = boundaries.Load_Crossings(
    "/home/daraghhollman/Main/Work/mercury/DataSets/philpott_2020.xlsx"
)

bow_shocks = crossings.loc[ (crossings["Type"] == "BS_OUT") | (crossings["Type"] == "BS_IN") ]

def Get_Sample(row):
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
        root_dir, sample_start, sample_end, strip=True, aberrate=True
    )

    sample_middle = sample.iloc[round(len(sample) / 2)]
    sample_middle_position = np.array(
        [
            sample_middle["X MSM' (radii)"],
            sample_middle["Y MSM' (radii)"],
            sample_middle["Z MSM' (radii)"],
        ]
    )

    local_time = traj.Local_Time(sample_middle_position)
    latitude = traj.Latitude(sample_middle_position)
    magnetic_latitude = traj.Magnetic_Latitude(sample_middle_position)

    with spice.KernelPool(User.METAKERNEL):
        et = spice.str2et(sample_middle["date"].strftime("%Y-%m-%d %H:%M:%S"))
        mercury_position, _ = spice.spkpos("MERCURY", et, "J2000", "NONE", "SUN")

        heliocentric_distance = np.sqrt(
            mercury_position[0] ** 2
            + mercury_position[1] ** 2
            + mercury_position[2] ** 2
        ) * Constants.KM_TO_AU


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
with multiprocessing.Pool(int(input("# of cores? "))) as pool:
    for result in tqdm(pool.imap(Get_Sample, process_items), total=len(process_items)):

        if result is not None:
            # Add row dictionary to list
            solar_wind_samples.append(result)


# Create dataframe from solar wind samples
solar_wind_samples = pd.DataFrame(solar_wind_samples)

print("")

solar_wind_samples.to_csv(
    f"/home/daraghhollman/Main/Work/mercury/DataSets/solar_wind_sample_{int(sample_length.total_seconds() / 60)}_mins.csv"
)
