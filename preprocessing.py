import numpy as np
from scipy.signal import butter, lfilter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

def standard_scaler(data):
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

def sliding_window(data, windowsize):
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

def windowing(data, labels, windowsize):
    """
    Apply a non-overlapping sliding window.

        Args:
            data: n-dimensional input array containg the data vectors
            lables: 1d input array containg the target values
            windowsize: size of the non-overlapping sliding window (must be
            a divisor of the integer defining the size of the input array)

        Returns:
            windowed_data: numpy array of shape(num_of_recs_new, windowsize) containg the windowed data vectors
            windowed_labels: numpy array of shape(num_of_recs_new) containing the windowed target values
    """
    num_of_chunks = int(data.shape[1]/windowsize)
    num_of_recs_new = int(data.shape[0]*num_of_chunks)

    windowed_data = list()
    windowed_labels = np.zeros(num_of_recs_new,)

    curr_chunks = 0
    chunks = sliding_window(data.flatten(), windowsize)

    for chunk in chunks:
        windowed_data.append(chunk)
        windowed_labels[curr_chunks] = labels[curr_chunks//num_of_chunks]
        curr_chunks += 1

    return np.asarray(windowed_data), windowed_labels

def split_dataset(X, y):
    """
    Split the data in a train, test and validation set using a 70:15:15 ratio.

        Args:
            X: n-dimensional input array containg the data vectors
            y: 1d input array containg the target values

        Returns:
            data: list containg the training, test and validation sets of data vectors
            labels: list containing the training, test and validation sets of target values
    """
    # split the dataset in train and test set while keeping the
    # ratio between the target classes (as the data is imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.7,
                                                        random_state=42,
                                                        stratify=y)

    # split the test set in test and validation set while keeping
    # the ratio between the target classes
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                    train_size=0.5,
                                                    random_state=42,
                                                    stratify=y_test)

    data = [X_train, X_test, X_val]
    labels = [y_train, y_test, y_val]

    return data, labels

def prepare_data(X, y):
    """
    Apply all preproccesing steps to the specified data set. This includes splitting the
    data into a training, test and validation set, filtering the data, standardizing the
    data and finally the windowing of the data.

        Args:
            X: n-dimensional input array containg the data vectors
            y: 1d input array containg the target values

        Returns:
            data: list containg the preprocessed training, test and validation sets of data vectors
            labels: list containing the preprocessed training, test and validation sets of target values
    """

    # split the data in a train, test and validation set in a 70:15:15 ratio
    data, labels = split_dataset(X, y)

    # define the scaler (on the test set in order to prevent data leakage)
    scaler = standard_scaler(data[0])

    # define the filter coefficients for the butterworth band-pass filter
    LOWCUT = 0.5
    HIGHCUT = 40
    SAMPLINGRATE = 173.61

    b, a = butter_bandpass(LOWCUT,HIGHCUT,SAMPLINGRATE)

    #define the windowsize
    WINDOWSIZE = 241

    prepared_data = list()
    prepared_labels = list()

    # apply the preprocessing pipeline to the different sets
    for X, y in zip(data, labels):
        filtered_X = butter_bandpass_filter(X,b,a)
        standardized_X = standardize_data(scaler, filtered_X)
        prepared_X, prepared_y = windowing(standardized_X, y, WINDOWSIZE)
        prepared_data.append(prepared_X)
        prepared_labels.append(prepared_y)

    return prepared_data, prepared_labels
