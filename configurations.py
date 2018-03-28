import os


class CONFIG(object):
    run_index = 'gru9k_inv'

    # when set to true, uses inverted data for training / beam search.
    # in evaluation mode, set to False
    inverted_training = False

    final_folder = os.path.join('..', 'rlts')

    # TODO: paths, data
    metric_weights = {'bf': 0., 'bb': 0., 'edf': 0.0, 'edb': 0.0, 'wb': .25, 'mlf': 0.6, 'mlb': 0.4}

    data_folder = os.path.join('..', 'data')

    if inverted_training is True:
        data_folder = os.path.join('..', 'data_inv')

    file_list = os.path.join('..', 'file_lists')

    # data_path = r'.\data_aug_6'
    log_path = os.path.join('.', 'log_' + run_index + '.csv')
    model_path = os.path.join(os.path.join('..', 'final'), 'gru9k_inv')

    forward_result_path = os.path.join(os.path.join('..', 'final'), 'beam5_forw_lstm9k10_result.csv')
    backward_result_path = os.path.join(os.path.join('..', 'final'), 'beam5_inv_gru9m10_result.csv')

    val_proportion = 0.05
    num_category = 2

    # TODO: training
    max_train_epochs = 20

    # TODO: model hypers

    drop_out = 0.29
    clip_threshold = 12
    learning_rate = 4.8e-5
    batch_size = 32

    embedding_size = 256
    RNN_size = 512
    vocabulary_size = 28
    # the letters, '+' and '#'.
