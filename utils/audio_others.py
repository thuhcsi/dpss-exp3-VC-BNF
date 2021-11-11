# from HU Jiankun
import librosa
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import dct
import matplotlib.pyplot as plt

# 超参数个数：16
hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 160,
    'win_length': 400,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 30.,
    'fmax': 7600.,
    'ref_db': 20,
    'min_db': -80.0,
    'griffin_lim_power': 1.5,
    'griffin_lim_iterations': 60,
    'silence_db': -28.0,
    'center': True,  # 不知道为什么提取ppg要是True
}

_mel_basis = None
_inv_mel_basis = None


# 超参数个数：1
def load_wav(wav_f, sr=hparams['sample_rate']):
    wav_arr, _ = librosa.load(wav_f, sr=sr)
    return wav_arr


# 超参数个数：1
def write_wav(write_path, wav_arr, sr=hparams['sample_rate']):
    wav_arr *= 32767 / max(0.01, np.max(np.abs(wav_arr)))
    wavfile.write(write_path, sr, wav_arr.astype(np.int16))
    return


# 超参数个数：1
def split_wav(wav_arr, top_db=-hparams['silence_db']):
    intervals = librosa.effects.split(wav_arr, top_db=top_db)
    return intervals


# 超参数个数：12
def wav2unnormalized_mfcc(wav_arr, sr=hparams['sample_rate'], preemphasis=hparams['preemphasis'],
                          n_fft=hparams['n_fft'], hop_len=hparams['hop_length'],
                          win_len=hparams['win_length'], num_mels=hparams['num_mels'],
                          n_mfcc=hparams['n_mfcc'], window=hparams['window'], fmin=0.0,
                          fmax=None, ref_db=hparams['ref_db'],
                          center=hparams['center']):
    emph_wav_arr = _preempahsis(wav_arr, pre_param=preemphasis)
    power_spec = _power_spec(emph_wav_arr, n_fft=n_fft, hop_len=hop_len, win_len=win_len, window=window, center=center)
    power_mel = _power_spec2power_mel(power_spec, sr=sr, n_fft=n_fft, num_mels=num_mels, fmin=fmin, fmax=fmax)
    db_mel = _power2db(power_mel, ref_db=ref_db)
    # 没有进行norm

    mfcc = dct(x=db_mel.T, axis=0, type=2, norm='ortho')[:n_mfcc]
    deltas = librosa.feature.delta(mfcc)
    delta_deltas = librosa.feature.delta(mfcc, order=2)
    mfcc_feature = np.concatenate((mfcc, deltas, delta_deltas), axis=0)
    return mfcc_feature.T


# 超参数个数：12
def wav2normalized_db_mel(wav_arr, sr=hparams['sample_rate'], preemphasis=hparams['preemphasis'],
                          n_fft=hparams['n_fft'], hop_len=hparams['hop_length'],
                          win_len=hparams['win_length'], num_mels=hparams['num_mels'],
                          window=hparams['window'], fmin=hparams['fmin'],
                          fmax=hparams['fmax'], ref_db=hparams['ref_db'], min_db=hparams['min_db'],
                          center=hparams['center']):
    emph_wav_arr = _preempahsis(wav_arr, pre_param=preemphasis)
    power_spec = _power_spec(emph_wav_arr, n_fft=n_fft, hop_len=hop_len, win_len=win_len, window=window,
                             center=center)  # (time, n_fft/2+1)
    power_mel = _power_spec2power_mel(power_spec, sr=sr, n_fft=n_fft, num_mels=num_mels, fmin=fmin, fmax=fmax)
    db_mel = _power2db(power_mel, ref_db=ref_db)
    normalized_db_mel = _db_normalize(db_mel, min_db=min_db)
    return normalized_db_mel


# 超参数个数：9
def wav2normalized_db_spec(wav_arr, sr=hparams['sample_rate'], preemphasis=hparams['preemphasis'],
                           n_fft=hparams['n_fft'], hop_len=hparams['hop_length'],
                           win_len=hparams['win_length'],
                           window=hparams['window'], ref_db=hparams['ref_db'], min_db=hparams['min_db'],
                           center=hparams['center']):
    emph_wav_arr = _preempahsis(wav_arr, pre_param=preemphasis)
    power_spec = _power_spec(emph_wav_arr, n_fft=n_fft, hop_len=hop_len, win_len=win_len, window=window,
                             center=center)  # (time, n_fft/2+1)
    # power_mel = _power_spec2power_mel(power_spec, sr=sr, n_fft=n_fft, num_mels=num_mels, fmin=fmin, fmax=fmax)
    db_spec = _power2db(power_spec, ref_db=ref_db)
    normalized_db_spec = _db_normalize(db_spec, min_db=min_db)
    return normalized_db_spec


# inv操作
# 超参数个数：14
def normalized_db_mel2wav(normalized_db_mel, sr=hparams['sample_rate'], preemphasis=hparams['preemphasis'],
                          n_fft=hparams['n_fft'], hop_len=hparams['hop_length'],
                          win_len=hparams['win_length'], num_mels=hparams['num_mels'],
                          window=hparams['window'], fmin=hparams['fmin'],
                          fmax=hparams['fmax'],
                          ref_db=hparams['ref_db'], min_db=hparams['min_db'],
                          center=hparams['center'], griffin_lim_power=hparams['griffin_lim_power'],
                          griffin_lim_iterations=hparams['griffin_lim_iterations']):
    db_mel = _db_denormalize(normalized_db_mel, min_db=min_db)
    power_mel = _db2power(db_mel, ref_db=ref_db)
    power_spec = _power_mel2power_spec(power_mel, sr=sr, n_fft=n_fft, num_mels=num_mels, fmin=fmin,
                                       fmax=fmax)  # 矩阵求逆猜出来的spec
    magnitude_spec = power_spec ** 0.5  # (time, n_fft/2+1)
    # print('-----1:', magnitude_spec.shape)
    # magnitude_spec_t = magnitude_spec.T
    griffinlim_powered_magnitude_spec = magnitude_spec ** griffin_lim_power  # (time, n_fft/2+1)
    # print('-----2:', griffinlim_powered_magnitude_spec.shape)
    # 送入griffinlim的是正常的 (time, n_fft/2+1)
    emph_wav_arr = _griffin_lim(griffinlim_powered_magnitude_spec, gl_iterations=griffin_lim_iterations,
                                n_fft=n_fft, hop_len=hop_len, win_len=win_len, window=window, center=center)

    wav_arr = _deemphasis(emph_wav_arr, pre_param=preemphasis)
    return wav_arr


# inv操作
# 超参数个数：11
def normalized_db_spec2wav(normalized_db_spec, sr=hparams['sample_rate'], preemphasis=hparams['preemphasis'],
                           n_fft=hparams['n_fft'], hop_len=hparams['hop_length'],
                           win_len=hparams['win_length'],
                           window=hparams['window'], ref_db=hparams['ref_db'], min_db=hparams['min_db'],
                           center=hparams['center'], griffin_lim_power=hparams['griffin_lim_power'],
                           griffin_lim_iterations=hparams['griffin_lim_iterations']):
    db_spec = _db_denormalize(normalized_db_spec, min_db=min_db)
    power_spec = _db2power(db_spec, ref_db=ref_db)  # (time, n_fft/2+1)
    magnitude_spec = power_spec ** 0.5  # (time, n_fft/2+1)
    # magnitude_spec_t = magnitude_spec.T #(n_fft/2+1, time)
    griffinlim_powered_magnitude_spec = magnitude_spec ** griffin_lim_power
    emph_wav_arr = _griffin_lim(griffinlim_powered_magnitude_spec, gl_iterations=griffin_lim_iterations,
                                n_fft=n_fft, hop_len=hop_len, win_len=win_len, window=window, center=center)

    wav_arr = _deemphasis(emph_wav_arr, pre_param=preemphasis)
    return wav_arr


# 超参数个数：1
def _preempahsis(wav_arr, pre_param):
    return signal.lfilter([1, -pre_param], [1], wav_arr)


# 超参数个数：1
def _deemphasis(wav_arr, pre_param):
    return signal.lfilter([1], [1, -pre_param], wav_arr)


# 超参数个数：5
# 注意center的参数
# return shape: [n_freqs, time]
def _stft(wav_arr, n_fft, hop_len, win_len, window, center):
    return librosa.core.stft(wav_arr, n_fft=n_fft, hop_length=hop_len,
                             win_length=win_len, window=window, center=center)


# 超参数个数：3
# stft_matrix shape [n_freqs, time]，复数
def _istft(stft_matrix, hop_len, win_len, window):
    return librosa.core.istft(stft_matrix, hop_length=hop_len,
                              win_length=win_len, window=window)


# 超参数个数：5
# 注意center的参数
# 以后只用power谱了，统一起来，都用stft之后先算平方，然后转换log后乘以10，但是其实不懂区别，哪一个更好？
# return shape: [time, n_freqs]
def _power_spec(wav_arr, n_fft, hop_len, win_len, window, center):
    s = _stft(wav_arr, n_fft=n_fft, hop_len=hop_len, win_len=win_len, window=window, center=center).T
    power = np.abs(s) ** 2
    return power


# 超参数个数：5
# input shape: [time, n_freqs]
# return shape: [time, n_mels]
def _power_spec2power_mel(power_spec, sr, n_fft, num_mels, fmin, fmax):
    power_spec_t = power_spec.T

    global _mel_basis
    _mel_basis = (librosa.filters.mel(sr, n_fft, n_mels=num_mels, fmin=fmin,
                                      fmax=fmax) if _mel_basis is None else _mel_basis)  # [n_mels, 1+n_fft/2]
    power_mel_t = np.dot(_mel_basis, power_spec_t)  # [n_mels, time]
    power_mel = power_mel_t.T

    return power_mel


# inv操作
# 超参数个数：5
# input shape: [time, n_mels]
# return shape: [time, n_freqs]
def _power_mel2power_spec(power_mel, sr, n_fft, num_mels, fmin, fmax):
    power_mel_t = power_mel.T

    global _mel_basis, _inv_mel_basis
    _mel_basis = (librosa.filters.mel(sr, n_fft, n_mels=num_mels, fmin=fmin,
                                      fmax=fmax) if _mel_basis is None else _mel_basis)  # [n_mels, 1+n_fft/2]
    _inv_mel_basis = (np.linalg.pinv(_mel_basis) if _inv_mel_basis is None else _inv_mel_basis)
    power_spec_t = np.dot(_inv_mel_basis, power_mel_t)
    power_spec_t = np.maximum(1e-10, power_spec_t)
    power_spec = power_spec_t.T

    return power_spec


# 超参数个数：1
# returned value: (10. * log10(power_spec) - ref_db)
def _power2db(power_spec, ref_db, tol=1e-5):
    return 10. * np.log10(power_spec + tol) - ref_db


# inv操作
# 超参数个数：1
def _db2power(power_db, ref_db):
    return np.power(10.0, 0.1 * (power_db + ref_db))


# 超参数个数：1
# return: db normalized to [0., 1.]
def _db_normalize(db, min_db):
    return np.clip((db - min_db) / -min_db, 0., 1.)


# inv操作
# 超参数个数：1
def _db_denormalize(normalized_db, min_db):
    return np.clip(normalized_db, 0., 1.) * -min_db + min_db


# 超参数个数：6
# input: magnitude spectrogram of shape [time, n_freqs]
# return: waveform array
def _griffin_lim(magnitude_spec, gl_iterations, n_fft, hop_len, win_len, window, center):
    # # 在这里进行gl的power，输入的是正常的magnitude_spec
    # magnitude_spec = magnitude_spec ** gl_power
    mag = magnitude_spec.T  # transpose to [n_freqs, time]
    # print('-----3:', magnitude_spec.shape)
    # print('-----4:', mag.shape)
    angles = np.exp(2j * np.pi * np.random.rand(*mag.shape))
    complex_mag = np.abs(mag).astype(np.complex)
    stft_0 = complex_mag * angles
    y = _istft(stft_0, hop_len=hop_len, win_len=win_len, window=window)
    for _i in range(gl_iterations):
        angles = np.exp(
            1j * np.angle(_stft(y, n_fft=n_fft, hop_len=hop_len, win_len=win_len, window=window, center=center)))
        y = _istft(complex_mag * angles, hop_len=hop_len, win_len=win_len, window=window)
    return y


def _wav2unnormalized_mfcc_test(wav_path, mfcc_path):
    wav_arr = load_wav(wav_path)
    mfcc = wav2unnormalized_mfcc(wav_arr)
    mfcc_label = np.load(mfcc_path)
    print(mfcc.min(), mfcc_label.min())
    print(mfcc.max(), mfcc_label.max())
    print(mfcc.mean(), mfcc_label.mean())
    print(np.abs(mfcc - mfcc_label))
    print(np.mean(np.abs(mfcc - mfcc_label)))

    plt.figure()
    plt.subplot(211)
    plt.imshow(mfcc.T, origin='lower')
    # plt.colorbar()
    plt.subplot(212)
    plt.imshow(mfcc_label.T, origin='lower')
    # plt.colorbar()
    plt.tight_layout()
    plt.show()
    return


def _wav2normalized_db_mel_test(wav_path, wav_rec_path):
    wav_arr = load_wav(wav_path)
    spec = wav2normalized_db_spec(wav_arr)
    wav_arr_rec = normalized_db_spec2wav(spec)
    write_wav(wav_rec_path, wav_arr_rec)


def _wav2normalized_db_spec_test(wav_path, wav_rec_path):
    wav_arr = load_wav(wav_path)
    mel = wav2normalized_db_mel(wav_arr)
    wav_arr_rec = normalized_db_mel2wav(mel)
    write_wav(wav_rec_path, wav_arr_rec)


if __name__ == '__main__':
    _wav2unnormalized_mfcc_test('test.wav', 'test_mfcc.npy')
    _wav2normalized_db_mel_test('test.wav', 'test_mel_rec.wav')
    _wav2normalized_db_spec_test('test.wav', 'test_spec_rec.wav')
