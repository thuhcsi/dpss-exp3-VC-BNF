import os
import random
import argparse
import numpy as np


from tqdm import tqdm
from utils import load_wav, _preemphasize, melspectrogram, spectrogram, F0Extractor,reform_input_audio
from models.wenet.bin.recognize import AsrReco
from config import Hparams


hps = Hparams


def length_validate(features):
    min_len = 1000000
    for feat in features:
        if feat.shape[0] < min_len:
            min_len = feat.shape[0]
    new_feats = (feat[:min_len, :] for feat in features)
    return new_feats


def main():
    parser = argparse.ArgumentParser('PreprocessingParser')
    parser.add_argument('--data_dir', type=str, help='data root directory')
    parser.add_argument('--save_dir', type=str, help='extracted feature save directory')
    parser.add_argument('--dev_rate', type=float, help='dev set rate', default=0.05)
    parser.add_argument('--test_rate', type=float, help='test set rate', default=0.05)
    parser.add_argument('--use_cuda', type=bool, help='use cuda or not', default=False)
    args = parser.parse_args()
    # args validation
    if args.dev_rate < 0 or args.dev_rate >= 1:
        raise ValueError('dev rate should be in [0, 1)')
    if args.test_rate < 0 or args.test_rate >= 1:
        raise ValueError('dev rate should be in [0, 1)')
    if args.test_rate + args.dev_rate >= 1:
        raise ValueError('dev rate + test rate should not be >= 1.')
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError('Directory {} not found!'.format(args.data_dir))
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    mel_dir = os.path.join(args.save_dir, 'mels')
    os.makedirs(mel_dir, exist_ok=True)
    linear_dir = os.path.join(args.save_dir, 'linears')
    os.makedirs(linear_dir, exist_ok=True)
    f0_dir = os.path.join(args.save_dir, 'f0s')
    os.makedirs(f0_dir, exist_ok=True)
    bnf_dir = os.path.join(args.save_dir, 'BNFs')
    os.makedirs(bnf_dir, exist_ok=True)
    for mode in ['train', 'dev', 'test']:
        if os.path.isfile(os.path.join(args.save_dir, "{}_meta.csv".format(mode))):
            os.remove(os.path.join(args.save_dir, "{}_meta.csv".format(mode)))
    wav_files = []
    
    for rootdir, subdir, files in os.walk(args.data_dir):
        for f in files:
            if f.endswith('.wav'):
                wav_files.append(os.path.join(rootdir, f))
    
    random.shuffle(wav_files)

    print('Set up BNFs extraction network')
    # Set up network
    bnf_config = './config/asr_config.yaml'
    asr_checkpoint_path = './pretrained_model/asr_model/final.pt'

    print('Loading BNFs extractor from {}'.format(bnf_config))
    bnf_extractor = AsrReco(bnf_config, asr_checkpoint_path,args.use_cuda)


    print('Extracting mel-spectrograms, spectrograms and f0s...')
    pitch_ext = F0Extractor("praat",sample_rate=16000)
    train_set = []
    dev_set = []
    test_set = []
    dev_start_idx = int(len(wav_files) * (1 - args.dev_rate - args.test_rate))
    test_stat_idx = int(len(wav_files) * (1 - args.test_rate))

    error=[]
    for i, wav_f in tqdm(enumerate(wav_files)):
        speaker = wav_f.split('/')[-2]
        # print(speaker)
        # exit()
        try:
            wav_arr = load_wav(wav_f)
        except:
            continue
        pre_emphasized_wav = _preemphasize(wav_arr)
        fid = '{}_{}'.format(speaker, wav_f.split('/')[-1].split('.')[0])
        # print(fid)
        # continue
        # extract mel-spectrograms
        mel_fn = os.path.join(mel_dir, '{}.npy'.format(fid))
        try:
            mel_spec = melspectrogram(pre_emphasized_wav).astype(np.float32).T
        except:
            continue
        # extract spectrograms
        linear_fn = os.path.join(linear_dir, '{}.npy'.format(fid))
        try:
            linear_spec = spectrogram(pre_emphasized_wav).astype(np.float32).T
        except:
            continue
        # new_wave_arr = inv_mel_spectrogram(mel_spec.T)
        # new_wave_arr = inv_preemphasize(new_wave_arr)
        # save_wav(new_wave_arr,'./0.wav')
        # save_wav(wav_arr,'./1.wav')
        # continue
        # extract f0s with vuv
        f0_fn = os.path.join(f0_dir, '{}.npy'.format(fid))
        try:
            f0 = pitch_ext.extract_f0_by_frame(wav_arr,True)
        except AssertionError as e:
            print(wav_f)
            error.append(wav_f)
            continue
        
        # extract ppgs
        
        reform_input_audio(wav_f,fid+'-temp.wav')
        BNFs, feat_lengths, PPGs = bnf_extractor.recognize(fid+'-temp.wav')

        BNFs_fn = os.path.join(bnf_dir, '{}.npy'.format(fid))
        
        # save features to respective directory
        mel_spec, linear_spec, f0, BNFs = length_validate((mel_spec, linear_spec, f0, BNFs))
        np.save(mel_fn, mel_spec)
        np.save(linear_fn, linear_spec)
        np.save(f0_fn, f0)
        np.save(BNFs_fn, BNFs)
        
        # write to csv
        if i < dev_start_idx:
            train_set.append(fid)
            with open(os.path.join(args.save_dir, 'train_meta.csv'),
                      'a', encoding='utf-8') as train_f:
                train_f.write('{}|BNFs/{}.npy|mels/{}.npy|linears/{}.npy|f0s/{}.npy\n'.format(fid, fid, fid, fid, fid))
        elif i < test_stat_idx:
            dev_set.append(fid)
            with open(os.path.join(args.save_dir, 'dev_meta.csv'),
                      'a', encoding='utf-8') as dev_f:
                dev_f.write('{}|BNFs/{}.npy|mels/{}.npy|linears/{}.npy|f0s/{}.npy\n'.format(fid, fid, fid, fid, fid))
        else:
            test_set.append(fid)
            with open(os.path.join(args.save_dir, 'test_meta.csv'),
                      'a', encoding='utf-8') as test_f:
                test_f.write('{}|BNFs/{}.npy|mels/{}.npy|linears/{}.npy|f0s/{}.npy\n'.format(fid, fid, fid, fid, fid))
    print('Done extracting features!')
    print(error)
    return


if __name__ == '__main__':
    main()
