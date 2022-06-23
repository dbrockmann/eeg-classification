import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, lfilter
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.restoration import denoise_wavelet

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

def design_scaler(data, type):
    """
    Design the scaler.

    Args:
        data: n-dimensional input array used to compute the mean and standard deviation used for later scaling

    Returns:
        fitted_scaler: fitted scaler
    """
    if type == 'StandardScaler':
        scaler = StandardScaler()
        fitted_scaler = scaler.fit(data)

    if type == 'MinMaxScaler':
        scaler = MinMaxScaler()
        fitted_scaler = scaler.fit(data)

    return fitted_scaler

def scale_data(data, scaler):
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

def prepare_data_autoencoder(X, y):
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
    scaler = design_scaler(data[0], 'MinMaxScaler')


    # define the filter coefficients for the butterworth band-pass filter
    LOWCUT = 0.5
    HIGHCUT = 40
    SAMPLINGRATE = 173.61

    b, a = butter_bandpass(LOWCUT,HIGHCUT,SAMPLINGRATE)

    #define the windowsize
    WINDOWSIZE = 256

    prepared_data = list()
    prepared_labels = list()

    # apply the preprocessing pipeline to the different sets
    for X, y in zip(data, labels):
        filtered_X = butter_bandpass_filter(X,b,a)
        standardized_X = scale_data(filtered_X, scaler)
        prepared_X, prepared_y = windowing(standardized_X, y, WINDOWSIZE)
        prepared_data.append(prepared_X)
        prepared_labels.append(prepared_y)

    return prepared_data, prepared_labels

def wavelet_denoise(data):
    """
    Perform Daubechies wavelet threshold denoising.

    Args:
        data: n-dimensional input array containg the data vectors

    Returns:
        y: denoised n-dimensional array containg the data vectors
    """
    y = denoise_wavelet(data, wavelet='db4', mode='soft', wavelet_levels=5, method='VisuShrink', rescale_sigma='True')
    return y

def extract_features(X):

    recordings = {}
    feature_names = ['min_data', 'max_data', 'mav_data', 'mean_data', 'avp_data', 'std_data', 'var_data', 'skew_data']
    subband_names = ['A5', 'D5', 'D4', 'D3', 'D2', 'D1']
    names = [f'{feature}_{subband}' for subband in subband_names for feature in feature_names]

    for k in range(X.shape[0]):
        feature_list = []
        coeffs = pywt.wavedecn(X[k], wavelet='db4', level=5, mode='per')
        for coeff in coeffs:
            if isinstance(coeff, dict):
                features = get_features(coeff['d'])
                feature_list.extend(features.values())
            else:
                features = get_features(coeff)
                feature_list.extend(features.values())
        recordings['recording_' + str(k + 1)] = feature_list

    return pd.DataFrame.from_dict(recordings, orient='index', columns=names)

def get_features(X):

    return {'min_data': X.min(),
            'max_data': X.max(),
            'mav_data': np.mean(abs(X)),
            'mean_data': np.mean(X),
            'avp_data': np.mean(abs(X)**2),
            'std_data': X.std(),
            'var_data': X.var(),
            'skew_data': skew(X)}

def fit_pca_model(X, components):
    pca = PCA(n_components = components)
    pca_model = pca.fit(X)

    return pca_model

def pca_dim_reduction(X, pca, components):

    columns = [f'pc_{num}' for num in range(1,components+1)]
    df_pca = pd.DataFrame(pca.transform(X), columns=columns, index=X.index)

    return df_pca

def prepare_data_features(X, y, COMPONENTS):

    # split the data in a train, test and validation set in a 70:15:15 ratio
    data, labels = split_dataset(X, y)

    features_data = list()
    features_labels = list()

    # extract statistical features in the different sets
    for X, y in zip(data, labels):
        windowed_X, windowed_y = windowing(X, y, 256)
        denoised_X = wavelet_denoise(windowed_X)
        df_features = extract_features(denoised_X)
        features_data.append(df_features)
        features_labels.append(windowed_y)

    # define the scaler (on the features extracted from the test set in order to prevent data leakage)
    scaler = design_scaler(features_data[0].to_numpy(), 'StandardScaler')

    # standardize the features in the different sets
    for i, df in enumerate(features_data):
        features_data[i] = pd.DataFrame(scale_data(df.to_numpy(), scaler), columns=list(df.columns), index=df.index)

    # fit pca model (on the features extracted from the test set in order to prevent data leakage)
    pca = fit_pca_model(features_data[0], COMPONENTS)

    # reduce dimensions to the size of COMPONENTS
    for i, df in enumerate(features_data):
        features_data[i] = pca_dim_reduction(df, pca, COMPONENTS).to_numpy()

    return features_data, features_labels
