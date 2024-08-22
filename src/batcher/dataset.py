import torch
import itertools
import os
import numpy as np
from torch.utils.data import Dataset
from random import randint
from brian2 import Hz, kHz
from brian2hears import Sound, erbspace, Gammatone, Filterbank
import scipy.signal
import pandas as pd

center_frequencies = erbspace(80 * Hz, 3 * kHz, 12)
channels_name = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
                 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
                 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz',
                 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4',
                 'O2']
CG1 = ['AF3', 'F3', 'F5', 'FC3', 'FC5', 'T7', 'C5', 'TP7', 'CP5', 'P5'] # 10
CG2 = ['FT7', 'FT8', 'T7', 'T8', 'TP7', 'TP8'] # 6
CG3 = ['F1', 'F3', 'FC3', 'FC1', 'Fz', 'F2', 'F4', 'FC4', 'FC2', 'Cz'] # 10
CG4 = ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'F5', 'F6', 'F7', 'F8'] # 9
CG5 = ['Cz', 'C1', 'C2', 'C3', 'C4', 'CPz', 'CP1', 'CP2', 'CP3', 'CP4']
CG6 = ['Fp1', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FC1', 'FC3', 'FC5', 'FT7', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3', 'CP5',
       'TP7', 'P1', 'P3', 'P5', 'P7', 'PO3', 'P9', 'PO7', 'Iz', 'O1'] # 27
CG7 = ['Fp2', 'AF4', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 'FC6', 'FT8', 'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4', 'CP6',
       'TP8', 'P2', 'P4', 'P6', 'P8', 'PO4', 'P10', 'PO8', 'AFz', 'O2'] # 27


class RegressionDataset(Dataset):
    """Generate data for the regression task."""

    def __init__(
            self,
            files,  # files containing EEG and envelope
            input_length,  # length of EEG/envelope segments
            channels,  # channels of EEG electrodes
            task,  # str, train/val/test
            g_con,
            high_pass_freq=None,
            low_pass_freq=None,
    ):

        self.input_length = input_length
        self.files = self.group_recordings(files)
        self.channels = channels
        self.task = task
        self.g_con = g_con

        self.fs = 64
        self.high_pass_freq = high_pass_freq
        self.low_pass_freq = low_pass_freq

        if self.high_pass_freq and self.low_pass_freq:
            self.filter_ = scipy.signal.butter(N=1,
                                               Wn=[self.high_pass_freq, self.low_pass_freq],
                                               btype="bandpass",
                                               fs=64,
                                               output="sos")
        if self.high_pass_freq and not self.low_pass_freq:
            self.filter_ = scipy.signal.butter(N=1,
                                               Wn=self.high_pass_freq,
                                               btype="highpass",
                                               fs=64,
                                               output="sos")
        if not self.high_pass_freq and self.low_pass_freq:
            self.filter_ = scipy.signal.butter(N=1,
                                               Wn=self.low_pass_freq,
                                               btype="lowpass",
                                               fs=64,
                                               output="sos")
        if not self.high_pass_freq and not self.low_pass_freq:
            self.filter_ = None

    def group_recordings(self, files):

        # new files are lists:[ ...[eeg,envelope]... ]

        new_files = []
        grouped = itertools.groupby(sorted(files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))
        # make sure eeg.npy always comes before envelope.npy
        for recording_name, feature_paths in grouped:
            new_files += [sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)]

        return new_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, recording_index):

        # 1. For within subject, return eeg, envelope and subject ID
        # 2. For held-out subject, return eeg, envelope

        if self.task == "train":
            x, y, sub_id, feature_path = self.__train_data__(recording_index)

        else:
            x, y, sub_id, feature_path = self.__test_data__(recording_index)

        return x, y, sub_id, feature_path

    def __train_data__(self, recording_index):

        framed_data = []

        for idx, feature in enumerate(self.files[recording_index]):
            # operate on one single group: for idx, feature in [eeg, envelope]
            if idx == 0:
                data = np.load(feature)  # load .npy to array
                df = pd.DataFrame(data, columns=channels_name)
                if self.channels != 64:
                    data = np.array(df[CG7])
                    # data=self.prepare_data(data)
                # get both eeg and envelope random segments of input_length
                start_idx = randint(0, len(data) - self.input_length)
                framed_data += [data[start_idx:start_idx + self.input_length]]  # lists of array of input_length
            if idx == 1:
                data = np.load(feature)  # load .npy to array
                # data = self.prepare_data(data)
                # filter envelope into 12 subbands using GammatoneFilter
                sound = Sound(data, samplerate=self.fs * Hz)
                filter_bank = Gammatone(sound, center_frequencies)
                multi_channel_envelope = filter_bank.process()  # output: numpy array with shape[input_length,12]
                framed_data += [multi_channel_envelope[start_idx:start_idx + self.input_length]]
            # print(feature)

        if self.g_con == True:
            sub_idx = feature.split('/')[-1].split('_-_')[1].split('-')[-1]
            sub_idx = int(sub_idx) - 1

        else:
            sub_idx = torch.FloatTensor([0])

            # return torch.FloatTensor(framed_data[0]), torch.FloatTensor(framed_data[1]), sub_idx

        return torch.FloatTensor(framed_data[0].copy()), torch.FloatTensor(framed_data[1].copy()), sub_idx, feature

    def __test_data__(self, recording_index):
        """
        return: list of segments [[eeg, envelope] ...] depending on self.input_length 
                e.g.,for 10 second-long input signal and input_length==5, return [[5, 5], [5, 5]]
        
        """
        framed_data = []

        for idx, feature in enumerate(self.files[recording_index]):
            if idx == 0:
                data = np.load(feature)
                df = pd.DataFrame(data, columns=channels_name)
                if self.channels != 64:
                    data = np.array(df[CG7])
            if idx == 1:
                data = np.load(feature)
                # filter envelope into 12 subbands using GammatoneFilter
                sound = Sound(data, samplerate=self.fs * Hz)
                filter_bank = Gammatone(sound, center_frequencies)
                multi_channel_envelope = filter_bank.process()  # output: numpy array with shape[input_length,12]
                data = multi_channel_envelope

            nsegment = data.shape[0] // self.input_length  # num(int) of segments of EEG/envelope
            data = data[:int(nsegment * self.input_length)]  # make sure data can be divided into int segments
            segment_data = [torch.FloatTensor(data[i:i + self.input_length].copy()).unsqueeze(0) for i in
                            range(0, data.shape[0], self.input_length)]
            # output:[torch.FloatTensor[],[],[]...[]], tensor.shape:(1, input_length)
            segment_data = torch.cat(segment_data)
            # segment_data.shape:(input_length, nsegments)
            framed_data += [segment_data]
            # [[], []], each list is a FloatTensor

        if self.g_con == True:
            sub_idx = feature.split('/')[-1].split('_-_')[1].split('-')[-1]
            sub_idx = int(sub_idx) - 1

        else:
            sub_idx = torch.FloatTensor([0])

        return framed_data[0], framed_data[1], sub_idx, feature

    def prepare_data(self, data):
        """ If specified, filter the data between highpass and lowpass
        :param data:  list of numpy arrays, eeg and envelope
        :return: filtered data

        """
        if self.filter_ is not None:
            # assuming time is the first dimension and channels the second
            resulting_data = scipy.signal.sosfiltfilt(self.filter_, data, axis=0)
        else:
            resulting_data = data

        return resulting_data
