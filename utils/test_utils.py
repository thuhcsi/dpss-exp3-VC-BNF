import os
import threading
import matplotlib

import matplotlib.pyplot as plt
from .audio import inv_preemphasize, inv_mel_spectrogram, save_wav


def draw_melspectrograms(save_dir, step, mel_batch, mel_lengths, ids, prefix=''):
    matplotlib.use('agg')
    for i, mel in enumerate(mel_batch):
        plt.imshow(mel[:mel_lengths[i], :].T, aspect='auto', origin='lower')
        plt.tight_layout()
        idx = ids[i].decode('utf-8') if type(ids[i]) is bytes else ids[i]
        plt.savefig(save_dir + '/{}-{}-{}.png'.format(prefix, idx, step))
        plt.close()
    return