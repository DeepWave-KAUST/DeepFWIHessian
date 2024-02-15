import numpy as np
import matplotlib.pyplot as plt


def create_triangular_mask(nt: int, dt: float, max_time: float, offsets: np.ndarray) -> np.ndarray:
    """
    Create a 2D mask for a shot gather with a triangular mute shape.
    
    Parameters
    ----------
    nt : int
        Number of time samples.
    dt : float
        Time sampling interval in seconds.
    max_time : float
        Maximum time to mute at the farthest offset in seconds.
    offsets : np.ndarray
        An array of offsets (distances) for each receiver from the shot point.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (nt, len(offsets)) representing the mask.
    """
    mask = np.ones((nt, len(offsets)), dtype=np.float32)  # Initialize the mask with ones (no muting).

    # Calculate the slope of the mute triangle based on the farthest offset.
    farthest_offset = np.max(np.abs(offsets))
    slope = max_time / farthest_offset

    # Apply the triangular mute to the mask.
    for i, offset in enumerate(offsets):
        mute_time = slope * np.abs(offset)  # Calculate mute time for each offset.
        mute_samples = int(np.floor(mute_time / dt))  # Convert mute time to number of samples.
        mask[:mute_samples, i] = 0  # Apply mute by setting the initial samples to 0.

    return mask


def create_upside_down_triangular_mask(nt: int, nx: int, shot_index: int, dx: float, dt: float, max_offset_time: float, time_shift: float) -> np.ndarray:
    """
    Create an upside-down triangular mask for a seismic shot gather with vertical time shift.

    :param nt: int
        Number of time samples.
    :param nx: int
        Number of receivers.
    :param shot_index: int
        Index of the shot position among the receivers.
    :param dx: float
        Receiver interval in meters.
    :param dt: float
        Time sampling interval in seconds.
    :param max_offset_time: float
        The time at the farthest offset to start muting.
    :param time_shift: float
        The vertical shift of the mute window in seconds.
    :return: np.ndarray
        A 2D numpy array representing the mask.
    """
    # Initialize the mask with ones
    mask = np.ones((nt, nx))
    
    # The farthest receiver from the shot point
    farthest_receiver = max(shot_index, nx - shot_index - 1)
    
    # Calculate the slope based on the farthest receiver
    slope = max_offset_time / (farthest_receiver * dx)
    
    # Apply the upside-down triangular mute with vertical shift
    for ix in range(nx):
        # Calculate the absolute offset from the shot
        offset = np.abs(ix - shot_index) * dx
        
        # Calculate the start time of the mute for this trace with time shift
        start_mute_time = slope * offset + time_shift
        
        # Ensure the start mute time is within bounds
        start_mute_time = max(0, min(nt * dt, start_mute_time))
        
        # Calculate the number of samples to mute at the start of this trace
        start_mute_samples = int(start_mute_time / dt)
        
        # Apply the mute
        mask[:(nt - start_mute_samples), ix] = 0

    return mask

def calculate_symmetric_arrival_times(nx: int, nt: int, dx: float, dt: float, velocity: float, shot_index: int, time_shift: float = 0.0) -> np.ndarray:
    """
    Calculate symmetric arrival times for a seismic shot gather.

    :param nx: int
        Number of receivers.
    :param nt: int
        Number of time samples.
    :param dx: float
        Receiver interval in meters.
    :param dt: float
        Time sampling interval in seconds.
    :param velocity: float
        Wave propagation velocity in meters per second.
    :param shot_index: int
        Index of the shot position among the receivers.
    :param time_shift: float, optional
        Time shift to be added to the arrival times, default is 0.0 seconds.
    :return: np.ndarray
        Array of arrival times in sample indices for each receiver.
    """
    # Calculate the offset for each receiver from the shot point
    offsets = (np.arange(nx) - shot_index) * dx
    # Calculate the absolute travel distance from the shot point
    travel_distances = np.abs(offsets)
    # Calculate the arrival times based on the travel distances
    arrival_times = travel_distances / velocity + time_shift
    arrival_samples = np.round(arrival_times / dt).astype(int)
    return arrival_samples

def cosine_taper_mask(nx: int, nt: int, arrival_samples: np.ndarray, width: int = 10) -> np.ndarray:
    """
    Create a cosine taper mask for a seismic shot gather.

    :param nx: int
        Number of receivers.
    :param nt: int
        Number of time samples.
    :param arrival_samples: np.ndarray
        Array of arrival time samples for each receiver.
    :param width: int, optional
        Width of the cosine taper window in samples, default is 10.
    :return: np.ndarray
        A 2D numpy array representing the taper mask.
    """
    mask = np.ones((nt, nx))

    for ix in range(nx):
        arrival_sample = arrival_samples[ix]
        start = max(0, arrival_sample - width // 2)
        end = min(nt, arrival_sample + width // 2)
        taper_length = end - start
        
        # Create a cosine taper window
        taper_window = (1 + np.cos(np.linspace(-np.pi, np.pi, taper_length))) / 2
        
        # Apply the taper window to the mask
        mask[start:end, ix] = 1 - taper_window

    return mask