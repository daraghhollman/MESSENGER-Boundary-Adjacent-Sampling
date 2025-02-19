"""
Script to load the philpott boundaries list, and load and save the solar wind time series data in a fixed length window before/after the boundary, along with any other raw data such as ephemeris, local time, latitude, magnetic latitude
"""

import datetime as dt
import multiprocessing

import diptest
from hermpy import trajectory
import numpy as np
import pandas as pd
import scipy.stats
import spiceypy as spice
from hermpy.utils import Constants


def main():

    # Load samples csv
    ms_samples_data_set = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/magnetosheath_sample_10_mins.csv",
        parse_dates=["Crossing Start", "Crossing End"],
    )
    sw_samples_data_set = pd.read_csv(
        "/home/daraghhollman/Main/Work/mercury/DataSets/solar_wind_sample_10_mins.csv",
        parse_dates=["Crossing Start", "Crossing End"],
    )

    outputs = [
        "/home/daraghhollman/Main/Work/mercury/DataSets/magnetosheath_features.csv",
        "/home/daraghhollman/Main/Work/mercury/DataSets/solar_wind_features.csv",
    ]

    for samples_data_set, label, output in zip(
        [ms_samples_data_set, sw_samples_data_set], ["MS", "SW"], outputs
    ):

        # Fix loading issues to do with an element being a series itself
        samples_data_set["|B|"] = samples_data_set["|B|"].apply(
            lambda x: list(map(float, x.strip("[]").split(",")))
        )
        samples_data_set["Bx"] = samples_data_set["Bx"].apply(
            lambda x: list(map(float, x.strip("[]").split(",")))
        )
        samples_data_set["By"] = samples_data_set["By"].apply(
            lambda x: list(map(float, x.strip("[]").split(",")))
        )
        samples_data_set["Bz"] = samples_data_set["Bz"].apply(
            lambda x: list(map(float, x.strip("[]").split(",")))
        )

        # Create empty list of dictionaries
        features_data = []

        # Iterrate through the crossings
        count = 0
        process_items = [(row, label) for _, row in samples_data_set.iterrows()]
        with multiprocessing.Pool() as pool:
            for result in pool.imap(Get_Features, process_items):

                if result is not None:
                    # Add row dictionary to list
                    features_data.append(result)

                count += 1
                print(f"{count} / {len(samples_data_set)}", end="\r")

        # Create dataframe from solar wind samples
        features_data_set = pd.DataFrame(features_data)

        print("")

        features_data_set.to_csv(output)


def Get_Features(input):
    """
    For a paricular solar wind sample (or 'event'),
    return a dictionary of all the relevant raw properties.
    """

    row, label = input

    try:
        spice.furnsh(
            "/home/daraghhollman/Main/SPICE/messenger/metakernel_messenger.txt"
        )

        # Each feature return will be a list with the calculation for each component
        mean = np.mean([row["|B|"], row["Bx"], row["By"], row["Bz"]], axis=1)
        median = np.median([row["|B|"], row["Bx"], row["By"], row["Bz"]], axis=1)
        std = np.std([row["|B|"], row["Bx"], row["By"], row["Bz"]], axis=1)
        skew = scipy.stats.skew([row["|B|"], row["Bx"], row["By"], row["Bz"]], axis=1)
        kurtosis = scipy.stats.kurtosis(
            [row["|B|"], row["Bx"], row["By"], row["Bz"]], axis=1
        )

        dip = np.array(
            [
                diptest.diptest(np.array(row[component]))
                for component in ["|B|", "Bx", "By", "Bz"]
            ]
        )

        grazing_angle = Get_Grazing_Angle(row)

        try:
            crossing_start = dt.datetime.strptime(
                row["Crossing Start"], "%Y-%m-%d %H:%M:%S.%f"
            )
        except:
            crossing_start = dt.datetime.strptime(
                row["Crossing Start"], "%Y-%m-%d %H:%M:%S"
            )

        try:
            sample_start = dt.datetime.strptime(
                row["Sample Start"], "%Y-%m-%d %H:%M:%S.%f"
            )
        except:
            sample_start = dt.datetime.strptime(
                row["Sample Start"], "%Y-%m-%d %H:%M:%S"
            )

        if label == "SW":
            # Solar wind sample is before the crossing
            if crossing_start > sample_start:
                is_inbound = 1

            # Solar wind sample is after the crossing
            else:
                is_inbound = 0

        elif label == "MS":
            # Magnetosheath sample is before the crossing
            if crossing_start > sample_start:
                is_inbound = 0

            # Magnetosheath sample is after the crossing
            else:
                is_inbound = 1

        else:
            raise Exception(f"Unknown sample label: {label}, expected 'SW' or 'MS'")

    finally:
        spice.kclear()

    return {
        # Time identifiers
        "Crossing Start": row["Crossing Start"],
        "Crossing End": row["Crossing End"],
        "Sample Start": row["Sample Start"],
        "Sample End": row["Sample End"],
        # Parameters
        "Mean": mean,
        "Median": median,
        "Standard Deviation": std,
        "Skew": skew,
        "Kurtosis": kurtosis,
        "Heliocentric Distance (AU)": row["Heliocentric Distance (AU)"],
        "Local Time (hrs)": row["Local Time (hrs)"],
        "Latitude (deg.)": row["Latitude (deg.)"],
        "Magnetic Latitude (deg.)": row["Magnetic Latitude (deg.)"],
        "X MSM' (radii)": row["X MSM' (radii)"],
        "Y MSM' (radii)": row["Y MSM' (radii)"],
        "Z MSM' (radii)": row["Z MSM' (radii)"],
        "Dip Statistic": dip[:, 0],
        "Dip P-Value": dip[:, 1],
        "Grazing Angle (deg.)": grazing_angle,
        "Is Inbound?": is_inbound,
    }


def Get_Grazing_Angle(row, function="bow shock"):
    """
    We find the closest position on the Winslow (2013) average BS model
    Assuming any expansion / compression occurs parallel to the normal vector
    of the curve, the vector to the closest point on the bow shock to MESSENGER
    is parallel with the bow shock normal at that closest point

    We then compare this vector, with the velocity vector of the spacecraft.
    """
    try:
        start_time = dt.datetime.strptime(row["Crossing Start"], "%Y-%m-%d %H:%M:%S.%f")

    except:
        start_time = dt.datetime.strptime(row["Crossing Start"], "%Y-%m-%d %H:%M:%S")

    try:
        end_time = dt.datetime.strptime(row["Crossing End"], "%Y-%m-%d %H:%M:%S.%f")

    except:
        end_time = dt.datetime.strptime(row["Crossing End"], "%Y-%m-%d %H:%M:%S")

    start_position = (
        trajectory.Get_Position(
            "MESSENGER",
            start_time + (end_time - start_time) / 2,
            frame="MSM",
            aberrate=True,
        )
        / Constants.MERCURY_RADIUS_KM
    )

    next_position = (
        trajectory.Get_Position(
            "MESSENGER",
            start_time + (end_time - start_time) / 2 + dt.timedelta(seconds=1),
            frame="MSM",
            aberrate=True,
        )
        / Constants.MERCURY_RADIUS_KM
    )

    cylindrical_start_position = np.array(
        [start_position[0], np.sqrt(start_position[1] ** 2 + start_position[2] ** 2)]
    )
    cylindrical_next_position = np.array(
        [next_position[0], np.sqrt(next_position[1] ** 2 + next_position[2] ** 2)]
    )

    cylindrical_velocity = cylindrical_next_position - cylindrical_start_position

    # normalise velocity
    cylindrical_velocity /= np.sqrt(np.sum(cylindrical_velocity**2))

    match function:
        case "bow shock":
            # Winslow+ (2013) parameters
            initial_x = 0.5
            psi = 1.04
            p = 2.75

            L = psi * p

            phi = np.linspace(0, 2 * np.pi, 10000)
            rho = L / (1 + psi * np.cos(phi))

            # Cylindrical coordinates (X, R)
            bow_shock_x_coords = initial_x + rho * np.cos(phi)
            bow_shock_r_coords = rho * np.sin(phi)

            boundary_positions = np.array([bow_shock_x_coords, bow_shock_r_coords]).T

        case "magnetopause":
            # Winslow+ (2013) parameters

            sub_solar_point = 1.45  # radii
            alpha = 0.5

            phi = np.linspace(0, 2 * np.pi, 10000)
            rho = sub_solar_point * (2 / (1 + np.cos(phi))) ** alpha

            # Cylindrical coordinates (X, R)
            magnetopause_x_coords = rho * np.cos(phi)
            magnetopause_r_coords = rho * np.sin(phi)

            boundary_positions = np.array(
                [magnetopause_x_coords, magnetopause_r_coords]
            ).T

        case _:
            raise ValueError(
                f"Invalid function choice: {function}. Options are 'bow shock', 'magnetopause'."
            )

    # We need to determine which point on the boundary curve is closest to the spacecraft
    # This method, utilising a k-d tree is computationally faster than iterrating through
    # the points and determining the distance.
    # O(logN) vs O(N)
    kd_tree = scipy.spatial.KDTree(boundary_positions)
    
    _, closest_position = kd_tree.query(cylindrical_start_position)

    # Get the normal vector of the BS at this point
    # This is just the normalised vector between the spacecraft and the closest point,
    # as the vector between an arbitrary point and the closest point on an arbitrary
    # curve is parallel to the normal vector of that curve at that closest point.
    normal_vector = boundary_positions[closest_position] - cylindrical_start_position

    normal_vector = normal_vector / np.sqrt(np.sum(normal_vector**2))

    grazing_angle = np.arccos(
                        np.dot(normal_vector, cylindrical_velocity)
                        / (np.sqrt(np.sum(normal_vector**2)) * np.sqrt( np.sum(cylindrical_velocity ** 2) ))
                    )
    grazing_angle = Constants.RADIANS_TO_DEGREES(grazing_angle)

    # If the grazing angle is greater than 90, then we take 180 - angle as its from the other side
    # This occurs as we don't make an assumption as to what side of the model boundary we are.
    # i.e. we could be referencing the normal, or the anti-normal.
    if grazing_angle > 90:
        # If the angle is greater than 90 degrees, we have the normal vector
        # the wrong way around. i.e. the inward pointing normal.
        grazing_angle = 180 - grazing_angle

    return grazing_angle


if __name__ == "__main__":
    main()
