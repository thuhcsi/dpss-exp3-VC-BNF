import librosa
import os
import re
import subprocess
from scipy.io import wavfile
import soundfile
import numpy as np
from scipy import signal
from scipy import interpolate
from config.hparams import Hparams

audio_hp = Hparams().Audio


def load_wav(path):
    return librosa.core.load(path, sr=audio_hp.sample_rate)[0]


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, audio_hp.sample_rate, wav.astype(np.int16))
    return


def spectrogram(y, clip_norm=True):
    D = _stft(y)
    S = _amp_to_db(np.abs(D)) - audio_hp.ref_level_db
    if clip_norm:
        S = _normalize(S)
    return S


def melspectrogram(y, clip_norm=True):
    D = _stft(y)
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - audio_hp.ref_level_db
    if clip_norm:
        S = _normalize(S)
    return S




def inv_spectrogram(spectrogram):
    S = _db_to_amp(_denormalize(spectrogram) + audio_hp.ref_level_db)
    return _griffin_lim(S ** audio_hp.power)


def inv_mel_spectrogram(mel_spectrogram):
    S = _mel_to_linear(_db_to_amp( #1e-10 25
        _denormalize(mel_spectrogram)+audio_hp.ref_level_db))
    return _griffin_lim(S ** audio_hp.power)


def find_endpoint(wav, threshold_db=-40.0, min_silence_sec=0.8):
    window_length = int(audio_hp.sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x: x + window_length]) < threshold:
            return x + hop_length
    return len(wav)


def _griffin_lim(S):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(audio_hp.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    if len(y.shape) == 1:  # [time_steps]
        return librosa.stft(y=y, n_fft=n_fft,
                            hop_length=hop_length,
                            win_length=win_length,
                            center=audio_hp.center)
    elif len(y.shape) == 2:  # [batch_size, time_steps]
        if y.shape[0] == 1:  # batch_size=1
            return np.expand_dims(librosa.stft(y=y[0], n_fft=n_fft,
                                               hop_length=hop_length,
                                               win_length=win_length,
                                               center=audio_hp.center),
                                  axis=0)
        else:  # batch_size > 1
            spec_list = list()
            for wav in y:
                spec_list.append(librosa.stft(y=wav, n_fft=n_fft,
                                              hop_length=hop_length,
                                              win_length=win_length,
                                              center=audio_hp.center))
            return np.concatenate(spec_list, axis=0)
    else:
        raise Exception('Wav dimension error in stft function!')


def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    if len(y.shape) == 2:  # spectrogram shape: [n_frame, n_fft]
        return librosa.istft(y, hop_length=hop_length,
                             win_length=win_length,
                             center=audio_hp.center)
    elif len(y.shape) == 3:  # spectrogram shape: [batch_size, n_frame, n_fft]
        if y.shape[0] == 1:  # batch_size = 1
            return np.expand_dims(librosa.istft(y[0],
                                                hop_length=hop_length,
                                                win_length=win_length,
                                                center=audio_hp.center),
                                  axis=0)
        else:  # batch_size > 1
            wav_list = list()
            for spec in y:
                wav_list.append(librosa.istft(spec,
                                              hop_length=hop_length,
                                              win_length=win_length,
                                              center=audio_hp.center))
                return np.concatenate(wav_list, axis=0)
    else:
        raise Exception('Spectrogram dimension error in istft function!')


def _stft_parameters():
    n_fft = (audio_hp.num_freq - 1) * 2
    hop_length = int(audio_hp.frame_shift_ms / 1000 * audio_hp.sample_rate)
    win_length = int(audio_hp.frame_length_ms / 1000 * audio_hp.sample_rate)
    return n_fft, hop_length, win_length


def _linear_to_mel(spectrogram):
    _mel_basis = _build_mel_basis()
    # print(_mel_basis[0:5, 0:5])
    # print("mel basis:",_mel_basis[10, 50:100])
    # print("spect:", spectrogram[50, 0:5])
    return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(mel_spectrogram):
    _inv_mel_basis = np.linalg.pinv(_build_mel_basis())
    linear_spectrogram = np.dot(_inv_mel_basis, mel_spectrogram)
    if len(linear_spectrogram.shape) == 3:
        # for 3-dimension mel, the shape of
        # inverse linear spectrogram will be [num_freq, batch_size, n_frame]
        linear_spectrogram = np.transpose(linear_spectrogram, [1, 0, 2])
    return np.maximum(1e-10, linear_spectrogram)


def _build_mel_basis():
    n_fft = (audio_hp.num_freq - 1) * 2
    return librosa.filters.mel(
        audio_hp.sample_rate,
        n_fft=n_fft,
        n_mels=audio_hp.num_mels,
        fmin=audio_hp.min_mel_freq,
        fmax=audio_hp.max_mel_freq)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    if audio_hp.symmetric_specs:
        return np.clip(
            (2 * audio_hp.max_abs_value) * (
                    (S - audio_hp.min_level_db) / (-audio_hp.min_level_db)
            ) - audio_hp.max_abs_value,
            -audio_hp.max_abs_value, audio_hp.max_abs_value)
    else:
        return np.clip(audio_hp.max_abs_value * (
                (S - audio_hp.min_level_db) / (-audio_hp.min_level_db)),
                       0, audio_hp.max_abs_value)


def _denormalize(S):
    if audio_hp.symmetric_specs:
        return ((np.clip(S, -audio_hp.max_abs_value, audio_hp.max_abs_value)
                 + audio_hp.max_abs_value) * (-audio_hp.min_level_db)
                / (2 * audio_hp.max_abs_value)
                + audio_hp.min_level_db)
    else:
        return ((np.clip(S, 0, audio_hp.max_abs_value) * (-audio_hp.min_level_db)
                 / audio_hp.max_abs_value)
                + audio_hp.min_level_db)


def _preemphasize(x):
    if len(x.shape) == 1:  # [time_steps]
        return signal.lfilter([1, -audio_hp.preemphasize], [1], x)
    elif len(x.shape) == 2:  # [batch_size, time_steps]
        if x.shape[0] == 1:
            return np.expand_dims(
                signal.lfilter([1, -audio_hp.preemphasize], [1], x[0]), axis=0)
        wav_list = list()
        for wav in x:
            wav_list.append(signal.lfilter([1, -audio_hp.preemphasize], [1], wav))
        return np.concatenate(wav_list, axis=0)
    else:
        raise Exception('Wave dimension error in pre-emphasis')


def inv_preemphasize(x):
    if audio_hp.preemphasize is None:
        return x
    if len(x.shape) == 1:  # [time_steps]
        return signal.lfilter([1], [1, -audio_hp.preemphasize], x)
    elif len(x.shape) == 2:  # [batch_size, time_steps]
        if x.shape[0] == 1:
            return np.expand_dims(
                signal.lfilter([1], [1, -audio_hp.preemphasize], x[0]), axis=0)
        wav_list = list()
        for wav in x:
            wav_list.append(signal.lfilter([1], [1, -audio_hp.preemphasize], wav))
        return np.concatenate(wav_list, axis=0)
    else:
        raise Exception('Wave dimension error in inverse pre-emphasis')


def mfcc(y):
    from scipy.fftpack import dct
    preemphasized = _preemphasize(y)
    D = _stft(preemphasized)
    S = librosa.power_to_db(_linear_to_mel(np.abs(D)**2))
    mfcc = dct(x=S, axis=0, type=2, norm='ortho')[:audio_hp.n_mfcc]
    deltas = librosa.feature.delta(mfcc)
    delta_deltas = librosa.feature.delta(mfcc, order=2)
    mfcc_feature = np.concatenate((mfcc, deltas, delta_deltas), axis=0)
    return mfcc_feature.T


def hyper_parameters_estimation(wav_dir):
    from tqdm import tqdm
    wavs = []
    for root, dirs, files in os.walk(wav_dir):
        for f in files:
            if re.match(r'.+\.wav', f):
                wavs.append(os.path.join(root, f))
    mel_db_min = 100.0
    mel_db_max = -100.0
    for f in tqdm(wavs):
        wav_arr = load_wav(f)
        pre_emphasized = _preemphasize(wav_arr)
        D = _stft(pre_emphasized)
        S = _amp_to_db(_linear_to_mel(np.abs(D)))
        mel_db_max = np.max(S) if np.max(S) > mel_db_max else mel_db_max
        mel_db_min = np.min(S) if np.min(S) < mel_db_min else mel_db_min
    return mel_db_min, mel_db_max


def _magnitude_spectrogram(audio, clip_norm):
    preemp_audio = _preemphasize(audio)
    mel_spec = melspectrogram(preemp_audio, clip_norm)
    linear_spec = spectrogram(preemp_audio, clip_norm)
    return mel_spec.T, linear_spec.T


def _energy_spectrogram(audio):
    preemp_audio = _preemphasize(audio)
    linear_spec = np.abs(_stft(preemp_audio)) ** 2
    mel_spec = _linear_to_mel(linear_spec)
    return mel_spec.T, linear_spec.T


def _extract_min_max(wav_path, mode, post_fn=lambda x: x):
    num_mels = audio_hp.num_mels
    num_linears = audio_hp.num_freq

    wavs = []
    for root, dirs, files in os.walk(wav_path):
        for f in files:
            if re.match(r'.+\.wav', f):
                wavs.append(os.path.join(root, f))

    num_wavs = len(wavs)
    mel_mins_per_wave = np.zeros((num_wavs, num_mels))
    mel_maxs_per_wave = np.zeros((num_wavs, num_mels))
    linear_mins_per_wave = np.zeros((num_wavs, num_linears))
    linear_maxs_per_wave = np.zeros((num_wavs, num_linears))

    for i, wav in enumerate(post_fn(wavs)):
        audio, sr = soundfile.read(wav)
        if mode == 'magnitude':
            mel, linear = _magnitude_spectrogram(audio, clip_norm=False)
        elif mode == 'energy':
            mel, linear = _energy_spectrogram(audio)
        else:
            raise Exception('Only magnitude or energy is supported!')

        mel_mins_per_wave[i, ] = np.amin(mel, axis=0)
        mel_maxs_per_wave[i, ] = np.amax(mel, axis=0)
        linear_mins_per_wave[i, ] = np.amin(linear, axis=0)
        linear_maxs_per_wave[i, ] = np.amax(linear, axis=0)

    mel_mins = np.reshape(np.amin(mel_mins_per_wave, axis=0), (1, num_mels))
    mel_maxs = np.reshape(np.amax(mel_maxs_per_wave, axis=0), (1, num_mels))
    linear_mins = np.reshape(np.amin(linear_mins_per_wave, axis=0), (1, num_mels))
    linear_maxs = np.reshape(np.amax(linear_mins_per_wave, axis=0), (1, num_mels))
    min_max = {
        'mel_min': mel_mins,
        'mel_max': mel_maxs,
        'linear_mins': linear_mins,
        'linear_max': linear_maxs
    }
    return min_max


def _normalize_min_max(spec, maxs, mins, max_value=1.0, min_value=0.0):
    spec_dim = len(spec.T)
    num_frame = len(spec)

    max_min = maxs - mins
    max_min = np.reshape(max_min, (1, spec_dim))
    max_min[max_min <= 0.0] = 1.0

    target_max_min = np.zeros((1, spec_dim))
    target_max_min.fill(max_value - min_value)
    target_max_min[max_min <= 0.0] = 1.0

    spec_min = np.tile(mins, (num_frame, 1))
    target_min = np.tile(min_value, (num_frame, spec_dim))
    spec_range = np.tile(max_min, (num_frame, 1))
    norm_spec = np.tile(target_max_min, (num_frame, 1)) / spec_range
    norm_spec = norm_spec * (spec - spec_min) + target_min
    return norm_spec


def _denormalize_min_max(spec, maxs, mins, max_value=1.0, min_value=0.0):
    spec_dim = len(spec.T)
    num_frame = len(spec)

    max_min = maxs - mins
    max_min = np.reshape(max_min, (1, spec_dim))
    max_min[max_min <= 0.0] = 1.0

    target_max_min = np.zeros((1, spec_dim))
    target_max_min.fill(max_value - min_value)
    target_max_min[max_min <= 0.0] = 1.0

    spec_min = np.tile(mins, (num_frame, 1))
    target_min = np.tile(min_value, (num_frame, spec_dim))
    spec_range = np.tile(max_min, (num_frame, 1))
    denorm_spec = spec_range / np.tile(target_max_min, (num_frame, 1))
    denorm_spec = denorm_spec * (spec - target_min) + spec_min
    return denorm_spec


def rescale(mel):
    x = np.linspace(1, mel.shape[0], mel.shape[0])
    xn = np.linspace(1, mel.shape[0], int(mel.shape[0] * 1.25))
    f = interpolate.interp1d(x, mel, kind='cubic', axis=0)
    rescaled_mel = f(xn)
    return rescaled_mel
