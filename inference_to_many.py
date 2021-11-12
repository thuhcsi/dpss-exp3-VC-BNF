import os

import numpy as np

import argparse
from models.wenet.bin.recognize import AsrReco
from models import BLSTMToManyConversionModel
from config import Hparams
from utils import load_wav, _preemphasize, melspectrogram,inv_mel_spectrogram, inv_preemphasize, save_wav, F0Extractor,reform_input_audio

import torch
def main():
    hps = Hparams
    parser = argparse.ArgumentParser('VC inference')
    parser.add_argument('--src_wav', type=str, help='source wav file path')
    parser.add_argument('--ckpt', type=str, help='model ckpt path')
    parser.add_argument('--tgt_spk', type=str, help='target speaker name',
                        choices=hps.SPEAKERS.spk_to_inds)
    parser.add_argument('--save_dir', type=str, help='synthesized wav save directory')
    args = parser.parse_args()
    # 0.
    src_wav_arr = load_wav(args.src_wav)
    src_wav_arr[src_wav_arr>1]=1.
    pre_emphasized_wav = _preemphasize(src_wav_arr)
    # Set up network
    bnf_config = './config/asr_config.yaml'
    asr_checkpoint_path = './pretrained_model/asr_model/final.pt'
    print('Loading BNFs extractor from {}'.format(bnf_config))
    bnf_extractor = AsrReco(bnf_config, asr_checkpoint_path,False)

    fid = args.src_wav.split('/')[-1].split('.wav')[0]
    reform_input_audio(args.src_wav,fid+'-temp.wav')
    BNFs, feat_lengths, PPGs = bnf_extractor.recognize(fid+'-temp.wav')



    # 2. extract normed_f0, mel-spectrogram
    pitch_ext = F0Extractor("praat",sample_rate=16000)
    f0 = pitch_ext.extract_f0_by_frame(src_wav_arr,True)
    # mel-spectrogram is extracted for comparison
    mel_spec = melspectrogram(pre_emphasized_wav).astype(np.float32).T

    # 3. prepare inputs
    min_len = min(f0.shape[0], BNFs.shape[0])
    vc_inputs = np.concatenate([BNFs[:min_len, :], f0[:min_len, :]], axis=1)
    vc_inputs = np.expand_dims(vc_inputs, axis=1)  # [time, batch, dim]

    # 4. setup vc model and do the inference
    model = BLSTMToManyConversionModel(
        in_channels=hps.Audio.bn_dim + 2,
        out_channels=hps.Audio.num_mels,
        num_spk=hps.SPEAKERS.num_spk,
        embd_dim=hps.BLSTMToManyConversionModel.spk_embd_dim,
        lstm_hidden=hps.BLSTMToManyConversionModel.lstm_hidden)
    device = torch.device('cpu')
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    tgt_spk_ind = torch.LongTensor([hps.SPEAKERS.spk_to_inds.index(args.tgt_spk)])
    predicted_mels = model(torch.tensor(vc_inputs).to(torch.float), tgt_spk_ind)
    predicted_mels = np.squeeze(predicted_mels.detach().numpy(), axis=1)

    # 5. synthesize wav
    synthesized_wav = inv_preemphasize(inv_mel_spectrogram(predicted_mels.T))
    resynthesized_wav = inv_preemphasize(inv_mel_spectrogram(mel_spec.T))
    ckpt_name = args.ckpt.split('/')[-1].split('.')[0].split('-')[-1]
    wav_name = args.src_wav.split('/')[-1].split('.')[0]
    save_wav(synthesized_wav, os.path.join(args.save_dir, '{}-to-{}-converted-{}.wav'.format(wav_name, args.tgt_spk, ckpt_name)))
    save_wav(resynthesized_wav, os.path.join(args.save_dir, '{}-src-cp-syn-{}.wav'.format(wav_name, ckpt_name)))
    return


if __name__ == '__main__':
    main()
