import librosa
import torch
from torch.utils.data import Dataset, DataLoader

def make_noised(data, noise, dB=0):
    noise, rate = librosa.load(noise, sr=16000)
    noise *= 10**(dB / 10)
    if len(data) >= len(noise):
        data = data[:len(noise)]
        noised_data = data + noise
    else:
        noised_data = data + noise[:len(data)]
    return (noised_data, data)

def make_wav(data, path, rate=16000, sec=10):
    data_length = rate * sec
    for i in range(len(data) // data_length):
        small_data = data[i*data_length:(i+1)*data_length]
        librosa.output.write_wav(path+'_'+str(i)+'.wav', small_data, rate)


class Dataset_for_speech(Dataset):
    def __init__(self, clean_speech_paths, noised_speech_paths, sr=16000):
        self.clean_speech_paths = clean_speech_paths
        self.noised_speech_paths = noised_speech_paths

    def __len__(self):
        return len(self.clean_speech_paths)

    def __getitem__(self, index):
        clean, rate = librosa.load(self.clean_speech_paths[index], sr=self.sr)
        noised, rate = librosa.load(self.noised_speech_paths[index], sr=self.sr)
        clean, noised = torch.from_numpy(clean), torch.from_numpy(noised)

        return clean, noised


def make_dataloaders(cleaned_tosave_path_train, noised_tosave_path_train, cleaned_tosave_path_val, noised_tosave_path_val, batch_size):
    clean_speech_paths_train = listdir_fullpath(cleaned_tosave_path_train)
    noised_speech_paths_train = listdir_fullpath(noised_tosave_path_train)
    clean_speech_paths_val = listdir_fullpath(cleaned_tosave_path_val)
    noised_speech_paths_val = listdir_fullpath(noised_tosave_path_val)

    train_set = Dataset_for_speech(clean_speech_paths_train, noised_speech_paths_train)
    val_set = Dataset_for_speech(clean_speech_paths_val, noised_speech_paths_val)
    image_datasets = {'train': train_set, 'val': val_set}

    dataloaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True),
                   'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)}
    return dataloaders
