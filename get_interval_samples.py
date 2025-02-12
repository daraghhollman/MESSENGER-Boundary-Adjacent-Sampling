"""
A script to extract features of boundary crossing intervals as a whole, with the ultimate aim of using k-means clustering.
"""

import multiprocessing

import numpy as np
import pandas as pd
import scipy
from hermpy import boundaries, mag, trajectory, utils
from tqdm import tqdm

# Load boundary crossing intervals
print("Loading boundary crossing intervals")
crossings = boundaries.Load_Crossings(path=utils.User.CROSSING_LISTS["Philpott"])
bow_shock_crossings = crossings.loc[
    crossings["Type"].str.contains("BS")
]  # limit to just bow shocks

# Remove short crossing intervals (i.e. length < 60 seconds)
bow_shock_crossings = bow_shock_crossings[
    (bow_shock_crossings["End Time"] - bow_shock_crossings["Start Time"])
    > pd.Timedelta(seconds=60)
]

# Load entire mission at 1 second resolution
print("Loading MAG data for entire mission")
data = mag.Load_Mission("/home/daraghhollman/Main/Work/messenger_full_mission.pickle")


def Get_Features(input):
    window_start, window_end = input

    data_section = data.loc[data["date"].between(window_start, window_end)]

    if len(data_section) == 0:
        return

    # Find features
    features = dict()
    for component in ["|B|", "Bx", "By", "Bz"]:
        component_data = data_section[component]
        features.update(
            {
                f"Mean {component}": np.mean(component_data),
                f"Median {component}": np.median(component_data),
                f"Standard Deviation {component}": np.std(component_data),
                f"Skew {component}": scipy.stats.skew(component_data),
                f"Kurtosis {component}": scipy.stats.kurtosis(component_data),
            }
        )

    middle_data_point = data_section.iloc[len(data_section) // 2]
    middle_position = [
        middle_data_point["X MSM' (radii)"],
        middle_data_point["Y MSM' (radii)"],
        middle_data_point["Z MSM' (radii)"],
    ]
    middle_features = [
        "X MSM' (radii)",
        "Y MSM' (radii)",
        "Z MSM' (radii)",
    ]
    for feature in middle_features:
        features[feature] = middle_data_point[feature]

    features.update(
        {
            "Latitude (deg.)": trajectory.Latitude(middle_position),
            "Magnetic Latitude (deg.)": trajectory.Magnetic_Latitude(middle_position),
            "Local Time (hrs)": trajectory.Local_Time(middle_position),
            "Heliocentric Distance (AU)": utils.Constants.KM_TO_AU(
                trajectory.Get_Heliocentric_Distance(middle_data_point["date"])
            ),
        }
    )

    # Prediction
    X = pd.DataFrame([features])
    column_names = list(X.columns.values)
    column_names.sort()
    X = X[column_names]

    return X


# Iterate through each crossing and extract the data
# Use multiprocessing
process_items = [
    (row["Start Time"], row["End Time"]) for _, row in bow_shock_crossings.iterrows()
]

samples = []
with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as pool:

    for result in tqdm(
        pool.imap(Get_Features, process_items), desc="Finding features", total=len(process_items)
    ):

        samples.append(result)

samples = pd.concat(samples)
samples = samples.reset_index(drop=True)
samples.to_csv(
    "/home/daraghhollman/Main/Work/mercury/DataSets/bow_shock_crossing_intervals.csv"
)
