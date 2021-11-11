import copy
from random import sample
import subprocess
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml
import numpy as np
from models.wenet.transformer.asr_model import init_asr_model
from models.wenet.utils.checkpoint import load_checkpoint
from utils.utils import read_lines,reshape_enc_mat_ctc


class AsrReco:
  def __init__(self, config_path, checkpoint_path,use_cuda=False):
    with open(config_path, 'r') as fin:
      configs = yaml.load(fin, Loader=yaml.FullLoader)

      # Init dataset and data loader
      # Init dataset and data loader
    test_collate_conf = copy.deepcopy(configs['collate_conf'])
    test_collate_conf['spec_aug'] = False
    test_collate_conf['spec_sub'] = False
    test_collate_conf['feature_dither'] = False
    test_collate_conf['speed_perturb'] = False
    test_collate_conf['wav_distortion_conf']['wav_distortion_rate'] = 0

    dataset_conf = configs.get('dataset_conf', {})
    dataset_conf['batch_size'] = 1
    dataset_conf['batch_type'] = 'static'
    dataset_conf['sort'] = False

    # Init asr model from configs
    model = init_asr_model(configs)

    load_checkpoint(model, checkpoint_path)
    #use_cuda = False
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()
    self._model = model
    self._device = device
    # self._test_collate_conf = test_collate_conf
    self._feature_extraction_conf = test_collate_conf['feature_extraction_conf']

    self.sample_rate = 16000
    dict_path = './pretrained_model/asr_model/dict.txt'
    self._char_dict = {}
    for line in read_lines(dict_path):
      arr = line.strip().split()
      assert len(arr) == 2
      self._char_dict[int(arr[1])] = arr[0]
    return

  def _extract_feature(self, wav_path):
    """ Extract acoustic fbank feature from origin waveform.

      Speed perturbation and wave amplitude distortion is optional.

      Args:
          wav_path:

      Returns:
          (keys, feats, labels)
      """
    feature_extraction_conf = self._feature_extraction_conf
    waveform, sample_rate = torchaudio.load(wav_path,normalize=False)
    waveform=waveform.float()
    mat = kaldi.fbank(
      waveform,
      num_mel_bins=feature_extraction_conf['mel_bins'],
      frame_length=feature_extraction_conf['frame_length'],
      frame_shift=feature_extraction_conf['frame_shift'],
      dither=0.0,
      energy_floor=0.0,
      sample_frequency=sample_rate
    )
    mat = mat.detach().cpu().numpy()
    return mat, mat.shape[0]




  def recognize(self, wave_path):
    with torch.no_grad():
      
      feats, feats_lengths = self._extract_feature(wave_path)

      feats = torch.from_numpy(feats).to(self._device).unsqueeze(0)
      feats_lengths = torch.tensor([feats.size(1)]).to(self._device)
      target_len = feats.size(1) + 2 
      enc, hyps , ctc_probs= self._model.ctc_greedy_search(
        feats,
        feats_lengths,
        decoding_chunk_size=-1,
        num_decoding_left_chunks=10,
        simulate_streaming=False)
      
      enc_numpy = enc.cpu().numpy()[0]
      enc = reshape_enc_mat_ctc(enc_numpy, target_len,dims=enc_numpy.shape[-1])
      feats_lengths = feats_lengths.cpu().numpy()[0]
      # print(hyps)

    content = []
    # for w in hyps[0]:
    #   content += self._char_dict[w]
    # print(" ".join(content))
    subprocess.call('rm {}'.format(wave_path), shell=True)
    return enc, feats_lengths ,ctc_probs


  


  