import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler

def butter_bandpass(lowcut, highcut, frequency, order=5):
    """
    Design Butterworth band-pass filter.

    Args:
        lowcut: low cutoff frequency
        highcut: high cutoff frequency
        frequency: sampling rate
        order: order of the filter, by default defined to be 5.

    Returns:
        a: Numerator of the IIR filter
        b: Denominator of the IIR filter
    """
    # calculate nyquist frequency
    nyq = 0.5 * frequency

    # design filter
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order,[low, high], btype='band')

    # return filter coefficients
    return b,a

def butter_bandpass_filter(data, b, a):
    """
    Apply Butterworth band-pass filter.

    Args:
        data: n-dimensional input array containg the data vectors
        a: Numerator of the IIR filter
        b: Denominator of the IIR filter

    Returns:
        y: filtered n-dimensional array containg the data vectors
    """
    y = lfilter(b, a, data)

    return y

def scaler(data):
    """
    Design the scaler.

    Args:
        data: n-dimensional input array used to compute the mean and standard deviation used for later scaling

    Returns:
        fitted_scaler: fitted scaler
    """
    scaler = StandardScaler()

    fitted_scaler = scaler.fit(data)

    return fitted_scaler

def standardize_data(scaler, data):
    """
    Perform standardization by centering and scaling

    Args:
        scaler: scaler used to scale the data along the features axis
        data: n-dimensional input array used to scale along the features axis.

    Returns:
        scaled_data: standardized n-dimensional array
    """
    scaled_data = scaler.transform(data)

    return scaled_data

def sliding_window_generator(data, windowsize):
    """
    Design a generator that iterates through 1d input array with
    predefiend chunks.

        Args:
            data: 1d input array containg all datasamples
            windowsize: size of the non-overlapping sliding window (must be
            a divisor of the integer defining the size of the input array)
    """
    for i in range(0, data.size, windowsize):
        yield data[i:i+windowsize]

def sliding_window(data, labels, windowsize):
    """
    Apply a non-overlapping sliding window.

        Args:
            data: n-dimensional input array containg the data vectors
            lables: 1d input array containg the target values
            windowsize: size of the non-overlapping sliding window (must be
            a divisor of the integer defining the size of the input array)

        Returns:
            updated_data: numpy array of shape(num_of_recs_new, windowsize) containg the updated data vectors
            updated_labels: numpy array of shape(num_of_recs_new) containing the updated target values
    """
    num_of_chunks = int(data.shape[1]/windowsize)
    num_of_recs_new = int(data.shape[0]*num_of_chunks)

    updated_data = list()
    updated_labels = np.zeros(num_of_recs_new,)

    curr_chunks = 0
    chunks = sliding_window_generator(data.flatten(), windowsize)

    for chunk in chunks:
        updated_data.append(chunk)
        updated_labels[curr_chunks] = labels[curr_chunks//num_of_chunks]
        curr_chunks += 1

    return np.asarray(updated_data), updated_labels
