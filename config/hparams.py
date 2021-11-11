class Hparams:
    class Audio:
        num_mels = 80
        ppg_dim = 351
        bn_dim = 256
        num_freq = 1025  
        min_mel_freq = 30.
        max_mel_freq = 7600.
        sample_rate = 16000
        frame_length_ms = 25
        frame_shift_ms = 10
        upper_f0 = 500.
        lower_f0 = 30.
        n_mfcc = 13
        preemphasize = 0.97
        min_level_db = -80.0
        ref_level_db = 20.0
        max_abs_value = 1.
        symmetric_specs = False
        griffin_lim_iters = 60
        power = 1.5
        center = True

    class SPEAKERS:
        num_spk = 3
        spk_to_inds = ['bzn', 'mst-female', 'mst-male']

    class TrainToOne:
        dev_set_rate = 0.1
        test_set_rate = 0.05
        epochs = 60
        train_batch_size = 32
        test_batch_size = 1
        shuffle_buffer = 128
        shuffle = True
        learning_rate = 1e-3
        num_workers = 16

    class TrainToMany:
        dev_set_rate = 0.1
        test_set_rate = 0.05
        epochs = 60
        train_batch_size = 32
        test_batch_size = 1
        shuffle_buffer = 128
        shuffle = True
        learning_rate = 1e-3
        num_workers = 16

    class BLSTMConversionModel:
        lstm_hidden = 256

    class BLSTMToManyConversionModel:
        lstm_hidden = 256
        spk_embd_dim = 64
