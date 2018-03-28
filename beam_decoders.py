import numpy as np
from GPU_dataloader import augmentation_simple
import os
from configurations import CONFIG
from GPU_dataloader import rgb2grey
import matplotlib.image as mpimg
import scipy.stats


def plain_beam(file_list, H, W, model, k):
    """
    plain beam search.
    NOTE: due to differences in rnn decoder initial states, this beam search is different for GRU and LSTM.
    :param file_list:
    :param model:
    :param k:
    :return: dictionary original_label:[beam_output]
    """

    vocabulary = 'abcdefghijklmnopqrstuvwxyz+#'
    dict_c2i = dict()
    for char in vocabulary:
        dict_c2i[char] = len(dict_c2i)
    # img: NHWC; x_context: NL

    results = dict()

    for file in file_list:
        beam_agenda = list()
        # agenda: [(seq_in_text, logprob)]
        batch_img = np.zeros(shape=(1, H, W, 1))
        batch_img[0, :, :, 0] = augmentation_simple(file, 0, H, W)

        # initialize beam search
        beam_agenda.append(('+', 0.))
        context_length = 1
        while True:
            # beam search

            # process each agenda item
            cache_agenda = list()
            for seq, logprob in beam_agenda:
                batch_x_context = np.zeros(shape=(1, context_length))
                for i in range(context_length):
                    batch_x_context[0, i] = dict_c2i[seq[i]]
                input_to_model = [batch_img, batch_x_context]
                predictions = model.predict(input_to_model, batch_size=1)
                predictions = np.reshape(predictions, newshape=(-1,))
                top_k = np.argsort(predictions)[-k:][::-1]

                # update cache agenda
                for i in range(k):
                    seq_extended = seq + vocabulary[top_k[i]]
                    logprob_new = logprob + np.log10(predictions[top_k[i]])
                    cache_agenda.append((seq_extended, logprob_new))

            # update agenda
            cache_agenda_sorted = sorted(cache_agenda, key=(lambda x: x[1]))
            beam_agenda = cache_agenda_sorted[-k:]

            # update iteration
            context_length += 1
            terminate_beam = False
            for seq, _ in beam_agenda:
                if '#' in seq:
                    terminate_beam = True
            if terminate_beam:
                break

        finalised_seqs = [item for item in beam_agenda if '#' in item[0]]
        final_seq = sorted(finalised_seqs, key=(lambda x: x[1]))[-1][0]

        results[file[:-4]] = [final_seq[1:-1]]
        print(file + ' beam search finished.')

    # filter the results to remove excessive repititions i.e. >= 3 times
    for label, seqs in results.items():
        seq_filtered = ''.join([seqs[0][i] for i in range(len(seqs[0]) - 2)
                                if not (seqs[0][i] == seqs[0][i+1] and seqs[0][i+1] == seqs[0][i+2])]) \
                       + seqs[0][-2:]
        results[label] = [seq_filtered]
    return results


hs, ws = list(), list()
ars = list()
files = os.listdir(CONFIG.data_folder)
for file in files:
    image = rgb2grey(mpimg.imread(os.path.join(CONFIG.data_folder, file)))
    ss = np.shape(image)
    hs.append(ss[0])
    ws.append(ss[1])
    ars.append(ss[0] * ss[1])

mean_area = np.mean(ars)
prop = mean_area / (131.*214.)
pr = scipy.stats.pearsonr(hs, ws)

a = 1

'''
def merging_results(result_path=r'D:\gregg\rlts\eval_10_result.csv'):
    results = after_beam.load_results()
    with open(result_path, 'r') as rin:
        buffer = rin.readline()
        buffer = rin.readline()
        while buffer != '':
            s_b = buffer.split(',')
            ref = s_b[0]
            hyp = s_b[1]
            results[ref].append(hyp)
            buffer = rin.readline()
    with open('finally.csv', 'w') as rout:
        for (ref, seqs) in results.items():
            rout.write(ref + ',' + seqs[0] + ',' + seqs[1] + ',' + seqs[2] + '\n')


merging_results()
'''