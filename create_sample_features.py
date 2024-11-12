"""
Script to load the philpott boundaries list, and load and save the solar wind time series data in a fixed length window before/after the boundary, along with any other raw data such as ephemeris, local time, latitude, magnetic latitude
"""

import datetime as dt
import multiprocessing

import diptest
import hermpy.trajectory as traj
import numpy as np
import pandas as pd
import scipy.stats
import spiceypy as spice


def main():

    # Load samples csv
    ms_samples_data_set = pd.read_csv(
        "./magnetosheath_sample_10_mins.csv",
        parse_dates=["crossing_start", "crossing_end"],
    )
    sw_samples_data_set = pd.read_csv(
        "./solar_wind_sample_10_mins.csv",
        parse_dates=["crossing_start", "crossing_end"],
    )

    outputs = ["./magnetosheath_features.csv", "./solar_wind_features.csv"]

    for samples_data_set, label, output in zip(
        [ms_samples_data_set, sw_samples_data_set], ["MS", "SW"], outputs
    ):

        # Fix loading issues to do with an element being a series itself
        samples_data_set["|B|"] = samples_data_set["|B|"].apply(
            lambda x: list(map(float, x.strip("[]").split(",")))
        )
        samples_data_set["B_x"] = samples_data_set["B_x"].apply(
            lambda x: list(map(float, x.strip("[]").split(",")))
        )
        samples_data_set["B_y"] = samples_data_set["B_y"].apply(
            lambda x: list(map(float, x.strip("[]").split(",")))
        )
        samples_data_set["B_z"] = samples_data_set["B_z"].apply(
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
        mean = np.mean([row["|B|"], row["B_x"], row["B_y"], row["B_z"]], axis=1)
        median = np.median([row["|B|"], row["B_x"], row["B_y"], row["B_z"]], axis=1)
        std = np.std([row["|B|"], row["B_x"], row["B_y"], row["B_z"]], axis=1)
        skew = scipy.stats.skew(
            [row["|B|"], row["B_x"], row["B_y"], row["B_z"]], axis=1
        )
        kurtosis = scipy.stats.kurtosis(
            [row["|B|"], row["B_x"], row["B_y"], row["B_z"]], axis=1
        )

        dip = np.array(
            [
                diptest.diptest(np.array(row[component]))
                for component in ["|B|", "B_x", "B_y", "B_z"]
            ]
        )

        grazing_angle = Get_Grazing_Angle(row)


        try:
            crossing_start = dt.datetime.strptime(row["crossing_start"], "%Y-%m-%d %H:%M:%S.%f")
        except:
            crossing_start = dt.datetime.strptime(row["crossing_start"], "%Y-%m-%d %H:%M:%S")

        try:
            sample_start = dt.datetime.strptime(row["sample_start"], "%Y-%m-%d %H:%M:%S.%f")
        except:
            sample_start = dt.datetime.strptime(row["sample_start"], "%Y-%m-%d %H:%M:%S")

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


    finally:
        spice.kclear()

    return {
        # Time identifiers
        "crossing_start": row["crossing_start"],
        "crossing_end": row["crossing_end"],
        "sample_start": row["sample_start"],
        "sample_end": row["sample_end"],
        # Parameters
        "mean": mean,
        "median": median,
        "std": std,
        "skew": skew,
        "kurtosis": kurtosis,
        "RH": row["RH"],
        "LT": row["LT"],
        "Lat": row["Lat"],
        "MLat": row["MLat"],
        "x_msm": row["x_msm"],
        "y_msm": row["y_msm"],
        "z_msm": row["z_msm"],
        "dip_stat": dip[:, 0],
        "dip_p_value": dip[:, 1],
        "grazing_angle": grazing_angle,
        "is_inbound": is_inbound,
    }


def Get_Grazing_Angle(row):
    # Find grazing angle
    try:
        start_time = dt.datetime.strptime(row["crossing_start"], "%Y-%m-%d %H:%M:%S.%f")

    except:
        start_time = dt.datetime.strptime(row["crossing_start"], "%Y-%m-%d %H:%M:%S")

    start_position = traj.Get_Position(
        "MESSENGER",
        start_time,
        frame="MSM",
    )
    next_position = traj.Get_Position(
        "MESSENGER",
        start_time + dt.timedelta(seconds=1),
        frame="MSM",
    )

    velocity = next_position - start_position

    # We find the closest position on the Winslow (2013) BS model
    initial_x = 0.5
    psi = 1.04
    p = 2.75

    L = psi * p

    phi = np.linspace(0, 2 * np.pi, 1000)
    rho = L / (1 + psi * np.cos(phi))

    bow_shock_x_coords = initial_x + rho * np.cos(phi)
    bow_shock_y_coords = rho * np.sin(phi)
    bow_shock_z_coords = rho * np.sin(phi)

    bow_shock_positions = np.array(
        [bow_shock_x_coords, bow_shock_y_coords, bow_shock_z_coords]
    ).T

    # Initialise some crazy big value
    shortest_distance = 10000
    closest_position = 0

    for i, bow_shock_position in enumerate(bow_shock_positions):

        distance_to_position = np.sqrt(
            (start_position[0] - bow_shock_position[0]) ** 2
            + (start_position[1] - bow_shock_position[1]) ** 2
            + (start_position[2] - bow_shock_position[2]) ** 2
        )

        if distance_to_position < shortest_distance:
            shortest_distance = distance_to_position
            closest_position = i

        else:
            continue

    # Get the normal vector of the BS at this point
    # This is just the normalised vector between the spacecraft and the closest point
    normal_vector = bow_shock_positions[closest_position] - start_position
    normal_vector = normal_vector / np.sqrt(np.sum(normal_vector**2))

    grazing_angle = (
        np.arccos(
            np.dot(normal_vector, velocity)
            / (np.sqrt(np.sum(normal_vector**2)) + np.sqrt(np.sum(velocity**2)))
        )
        * 180
        / np.pi
    )

    # If the grazing angle is greater than 90, then we take 180 - angle as its from the other side
    # This occurs as we don't make an assumption as to what side of the model boundary we are

    if grazing_angle > 90:
        grazing_angle = 180 - grazing_angle

    return grazing_angle


if __name__ == "__main__":
    main()
