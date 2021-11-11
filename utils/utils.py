import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
import librosa
import subprocess
from scipy.io import wavfile
import parselmouth
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
def repeat_mat(mat, k):
  m, n = np.shape(mat)
  new_mat = np.zeros([m * k, n])
  for i in range(m):
    new_mat[i * k:(i + 1) * k] = mat[i, :]
  return new_mat

def reshape_enc_mat_ctc(enc_npy, target_len,dims=256):
  enc_len = np.shape(enc_npy)[0]
  half_pad = (target_len - enc_len * 4) // 2
  new_enc = np.zeros([target_len, dims])
  new_enc[half_pad:enc_len * 4 + half_pad] = repeat_mat(enc_npy, 4)
  return new_enc

def load_wav(path, sr):
  return librosa.core.load(path, sr=sr)[0]


def save_wav(wav, path, sr, k=None):
  if k:
    norm_wav = wav * 32767 / max(0.01, np.max(np.abs(wav))) * k
  else:
    norm_wav = wav * 32767
  wavfile.write(path, sr, norm_wav.astype(np.int16))
  return


def change_sample_rate(input_path, sample_rate, output_path):
  cmd = "ffmpeg -i {} -ac 1 -ar {} -y {} -loglevel 8".format(
    input_path, sample_rate, output_path)
  subprocess.call(cmd, shell=True)
  return


def reform_input_audio(source_wav_path, reform_wav_path,sr = 16000):

  change_sample_rate(source_wav_path, sr, reform_wav_path)
  signal, _ = librosa.load(reform_wav_path, sr)
  pad_length = 0 * 240  # pad zero frame in training
  pad_sig = np.pad(signal, (pad_length, pad_length))
  pad_sig = pad_sig / np.max(np.abs(pad_sig)) * 0.75
  save_wav(pad_sig, reform_wav_path, sr=sr)
  return

def masked_mse_loss(inputs: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor):
    """
    :param inputs: [batch, time, dim]
    :param targets: [batch, time, dim]
    :param lengths: [batch]
    :return:
    """
    if lengths is None:
        return nn.MSELoss()(inputs, targets)
    else:
        max_len = max(lengths.cpu().numpy())
        mask = torch.arange(max_len).expand([len(lengths), max_len]).to(device) < lengths.unsqueeze(1)
        mask = mask.to(dtype=torch.float32)
        mse_loss = torch.mean(
            torch.sum(torch.mean((inputs - targets) ** 2, dim=2) * mask,
                      dim=1) / lengths.to(dtype=torch.float32))
        return mse_loss


def softmax(x, axis=-1):
    assert len(x.shape) == 2
    _max = np.max(x)
    probs = np.exp(x - _max) / np.sum(np.exp(x - _max), axis=axis, keepdims=True)
    return probs


def read_lines(data_path, log=True):
  lines = []
  with open(data_path, encoding="utf-8") as fr:
    for line in fr.readlines():
      if len(line.strip().replace(" ", "")):
        lines.append(line.strip())
  if log:
    print("read {} lines from {}".format(len(lines), data_path))
    print("example(last) {}\n".format(lines[-1]))
  return lines


def write_lines(data_path, lines, log=True):
  with open(data_path, "w", encoding="utf-8") as fw:
    for line in lines:
      fw.write("{}\n".format(line))
  if log:
    print("write {} lines to {}".format(len(lines), data_path))
    print("example(last line): {}\n".format(lines[-1]))
  return


class F0Extractor:
  def __init__(self, method_type, sample_rate=16000, hop_time=0.01, min_f0=20,
               max_f0=600):
    """ FO EXTRACTOR

    Args:
      method_type: must be in [praat, sptk, pyworld], recommend to use praat.
      sample_rate: sr
      hop_time: frame shift time, hop_size=hop_time*sr
      min_f0: min threshold of f0 (Hz)
      max_f0: max threshold of f0 (Hz)
    """
    self._max_f0 = max_f0
    self._min_f0 = min_f0
    self._sr = sample_rate
    self._hop_size = int(sample_rate * hop_time)
    self.hop_time=hop_time
    self._method_type = method_type
    print("init {}-f0 extractor with {}/{}".format(method_type, min_f0, max_f0))
    return

  def _basic_analysis(self, signal):
    assert -1 <= np.min(signal) <= np.max(signal) <= 1.0
      
    if self._method_type == "praat":
      sound = parselmouth.Sound(signal.astype(np.float64), self._sr, 0.0)
      time_step = 0.0025
      pitch = sound.to_pitch(
        time_step=time_step, pitch_floor=self._min_f0,
        pitch_ceiling=self._max_f0)
      f0 = pitch.selected_array['frequency']
      time = pitch.xs()
      unvoiced_value = 0
    elif self._method_type == "sptk":
      signal = signal * 32767
      signal = signal.astype(np.int16)
      pm_times, pm, f0_times, f0, corr = pyreaper.reaper(
        signal, self._sr, frame_period=0.0025, maxf0=self._max_f0,
        minf0=self._min_f0, unvoiced_cost=1.1)
      unvoiced_value = -1
      time = f0_times
    elif self._method_type == "world":
      frame_period = 10
      f0, t = pw.dio(
        signal.astype(np.float64), self._sr, frame_period=frame_period)
      unvoiced_value = 0
      time = t
    else:
      raise Exception("unvalid method type")
    return f0.reshape((-1)), time, unvoiced_value

  @staticmethod
  def _extract_vuv(signal, unvoiced_value):
    is_unvoiced = np.isclose(signal,
                             unvoiced_value * np.ones_like(signal),
                             atol=1e-2)
    is_voiced = np.logical_not(is_unvoiced)
    return is_voiced
    
  @staticmethod
  def _interpolate(signal, is_voiced):
    """Linearly interpolates the signal in unvoiced regions such that there are
       no discontinuities.

      Args:
          signal (np.ndarray[n_frames, feat_dim]): Temporal signal.
          is_voiced (np.ndarray[n_frames]<bool>): Boolean array indicating if each frame is voiced.

      Returns:
          (np.ndarray[n_frames, feat_dim]): Interpolated signal, same shape as signal.
      """
    n_frames = signal.shape[0]
    feat_dim = signal.shape[1]

    # Initialize whether we are starting the search in voice/unvoiced.
    in_voiced_region = is_voiced[0]

    last_voiced_frame_i = None
    for i in range(n_frames):
      if is_voiced[i]:
        if not in_voiced_region:
          # Current frame is voiced, but last frame was unvoiced.
          # This is the first voiced frame after an unvoiced sequence,
          # interpolate the unvoiced region.

          # If the signal starts with an unvoiced region then
          # `last_voiced_frame_i` will be None.
          # Bypass interpolation and just set this first unvoiced region
          # to the current voiced frame value.
          if last_voiced_frame_i is None:
            signal[:i + 1] = signal[i]

          # Use `np.linspace` to create a interpolate a region that
          # includes the bordering voiced frames.
          else:
            start_voiced_value = signal[last_voiced_frame_i]
            end_voiced_value = signal[i]

            unvoiced_region_length = (i + 1) - last_voiced_frame_i
            interpolated_region = np.linspace(start_voiced_value,
                                              end_voiced_value,
                                              unvoiced_region_length)
            interpolated_region = interpolated_region.reshape(
              (unvoiced_region_length, feat_dim))

            signal[last_voiced_frame_i:i + 1] = interpolated_region

        # Move pointers forward, and waiting to find another unvoiced section.
        last_voiced_frame_i = i

      in_voiced_region = is_voiced[i]

    # If the signal ends with an unvoiced region then it would not have been
    # caught in the loop. Similar to the case with an unvoiced region at the
    # start we can bypass the interpolation.
    if not in_voiced_region:
      signal[last_voiced_frame_i:] = signal[last_voiced_frame_i]
    return signal

  def extract_f0_by_frame(self, signal, interpolate):
    """ extract f0 and vuv by frame

    Args:
      signal: np.array, [length]
      interpolate: using vuv to interpolate

    Returns:
      f0 and vuv by frame

    """
    f0, x0, unvoiced_value = self._basic_analysis(signal)
    f0 = f0.reshape((-1, 1))
    vuv = self._extract_vuv(f0, unvoiced_value=unvoiced_value)
    if interpolate:
      f0 = self._interpolate(f0, vuv).reshape((-1))

    f0 = f0.reshape((-1))
    frame_num = len(signal) // self._hop_size + 1
    frame_time = np.arange(frame_num) * self._hop_size / self._sr
    time = x0
    assert abs(frame_time[-1] - time[-1]) < 0.1, "{}/{}".format(
      frame_time[-1], time[-1])
    f0_by_frame = np.interp(frame_time, x0, f0)
    vuv_by_frame = np.interp(frame_time, x0, vuv.reshape((-1)))

    # norm
    f0_mean = np.mean(f0_by_frame)
    f0_std = np.std(f0_by_frame)
    f0_by_frame = (f0_by_frame - f0_mean)/(f0_std+1e-3)

    f0_by_frame = f0_by_frame[:,np.newaxis]
    vuv_by_frame = vuv_by_frame[:,np.newaxis]
    f0_with_vuv = np.concatenate([f0_by_frame,vuv_by_frame],axis=-1)
    return f0_with_vuv

