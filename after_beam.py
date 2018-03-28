from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.metrics.distance import edit_distance
import numpy as np
from utils import modified_levenshtein_distance

from configurations import CONFIG


def best_candidate(hypothesis_forward, hypothesis_backward, full_candidate_list, metric_weights,
                   forward_bleu_weights=list([.25, .25, .25, 0.25]), backward_bleu_weights=list([.25, .25, .25, 0.25])):
    """
    returns a candidate that admits to the highest score wrt the hypothesis.
    the score is a weighted average of the forward/backward edit_similarity, bleu1 to bleu4 and
    weighted_bleu.
    :param hypothesis_forward:
    :param hypothesis_backward:
    :param full_candidate_list:
    :param metric_weights: dict that specifies the weights of each metric.
        bf, bb, edf, edb, wb
    :return:
    """
    candidate_scores = dict()
    sm = SmoothingFunction().method3
    for candidate in full_candidate_list:
        score = 0.
        score += metric_weights['bf'] * sentence_bleu([list(candidate)], list(hypothesis_forward),
                                                      weights=forward_bleu_weights, smoothing_function=sm)
        score += metric_weights['bb'] * sentence_bleu([list(candidate)], list(hypothesis_backward),
                                                      weights=backward_bleu_weights, smoothing_function=sm)
        if metric_weights['edf'] != 0:
            score += metric_weights['edf'] * max(0, 1 - edit_distance(candidate, hypothesis_forward)/len(candidate))

        if metric_weights['edb'] != 0:
            score += metric_weights['edb'] * max(0, 1 - edit_distance(candidate, hypothesis_backward)/len(candidate))

        if metric_weights['wb'] != 0:
            score += metric_weights['wb'] * weighted_blue(candidate, [hypothesis_forward, hypothesis_backward])

        if metric_weights['mlf'] != 0:
            ldist = modified_levenshtein_distance(candidate, hypothesis_forward)
            score += metric_weights['mlf'] * max(0, 1 - 2 * ldist / len(candidate))

        if metric_weights['mlb'] != 0:
            ldist = modified_levenshtein_distance(candidate[::-1], hypothesis_backward[::-1])
            score += metric_weights['mlb'] * max(0, 1 - 2 * ldist / len(candidate))

        candidate_scores[candidate] = score

    candidate_list = sorted(full_candidate_list, key=(lambda x: candidate_scores[x]))
    return candidate_list[-1]


def weighted_blue(reference, hypotheses, bleu_weights=list([.25, .25, .25, .25]), direction_weights=list([0.62, 0.38])):
    """
    evaluated bleu in a novel manner.
    we have two hypotheses, the bleu is evaluated as a fine-grained weighted sum such that it relies
    more on the forward hypothesis in earlier part of the sequence, and more on the other one otherwise.
    nltk smoothing method3 is applied.
    :param hypotheses: [forward hypothesis, backward hypothesis]
    :param direction_weights: weight the forward hypothesis and the backward hypothesis
    :return: weighted blue-4. if one hypothesis has at most 4 characters, return the average standard bleu-3
    """
    if len(hypotheses[0]) < 5 or len(hypotheses[1]) < 5:
        return 0.5 * sentence_bleu(reference, hypotheses[0],
                                   smoothing_function=SmoothingFunction().method3) + \
               0.5 * sentence_bleu(reference, hypotheses[1],
                                   smoothing_function=SmoothingFunction().method3)

    # evaluate acc
    p_n = list([0., 0., 0., 0.])
    for n in range(4):
        precision = 0.
        for direction, hypothesis in enumerate(hypotheses):
            for i in range(len(hypothesis) - n):
                ngram = hypothesis[i:i+n+1]
                if reference.find(ngram) != -1:
                    if direction == 0:
                        positional_weight = 1 - i/(len(hypothesis) - n - 1)
                    elif direction == 1:
                        positional_weight = i/(len(hypothesis) - n - 1)
                    weight = positional_weight * direction_weights[direction] \
                        / ((1 - positional_weight) * (1 - direction_weights[direction]) +
                           positional_weight * direction_weights[direction])
                    precision += 1/(len(hypothesis) - n) * weight
        p_n[n] = precision

    # smoothing method3
    n = 0
    while p_n[n] != 0. and n < 3:
        n += 1
    k = 1
    while n < 4:
        p_n[n] = np.power(2., -k) / (len(hypotheses[0]) + len(hypotheses[1]))
        k += 1
        n += 1

    return np.exp(np.sum([np.log(p_n[i]) * bleu_weights[i] for i in range(4)]))


def load_results(forward_path=CONFIG.forward_result_path, backward_path=CONFIG.backward_result_path):
    """
    returns dictionary formulated as ref: [forward_generation, backward_generation]
    :param forward_path:
    :param backward_path:
    :return: dictionary formulated as ref: [forward_generation, backward_generation]
    """
    result_dict = dict()
    with open(forward_path, 'r') as forward_in:
        # discard headers
        buffer = forward_in.readline()
        buffer = forward_in.readline()
        while buffer != '':
            seqs = buffer.split(',')
            result_dict[seqs[0]] = [seqs[1]]
            buffer = forward_in.readline()

    with open(backward_path, 'r') as backward_in:
        # discard headers
        buffer = backward_in.readline()
        buffer = backward_in.readline()
        while buffer != '':
            seqs = buffer.split(',')
            reference = seqs[0][::-1]
            result_dict[reference].append(seqs[1][::-1])
            buffer = backward_in.readline()

    return result_dict


def evaluation(results_ori):
    """
    extends results list with evaluation metrics and evaluate the mean on those metrics
    edit_dist_score, bleu1 to bleu4. bleu scores are evaluated with nltk smoothing method 3.
    metrics:
    edit_dict_score: max(1 - edit_dist/length(ref), 0)
    :param results_ori: label:[output, whatever...]
    :return:
    """
    results = dict(results_ori)
    scores = dict()
    bleu1_scores = dict()
    bleu2_scores = dict()
    bleu3_scores = dict()
    bleu4_scores = dict()
    for label, seqs in results.items():
        if seqs[0] == label:
            scores[label] = 1.
            bleu1_scores[label] = 1.
            bleu2_scores[label] = 1.
            bleu3_scores[label] = 1.
            bleu4_scores[label] = 1.
        else:
            edit_dist = edit_distance(label, seqs[0])
            bleu1 = sentence_bleu([list(label)], list(seqs[0]),
                                  weights=[1., 0, 0, 0], smoothing_function=SmoothingFunction().method3)
            bleu2 = sentence_bleu([list(label)], list(seqs[0]),
                                  weights=[.5, .5, 0, 0], smoothing_function=SmoothingFunction().method3)
            bleu3 = sentence_bleu([list(label)], list(seqs[0]),
                                  weights=[.33, .33, .33, 0], smoothing_function=SmoothingFunction().method3)
            bleu4 = sentence_bleu([list(label)], list(seqs[0]), smoothing_function=SmoothingFunction().method3)
            scores[label] = 1 - edit_dist/(max(len(label), len(seqs[0])))
            scores[label] = max(scores[label], 0)
            bleu1_scores[label] = bleu1
            bleu2_scores[label] = bleu2
            bleu3_scores[label] = bleu3
            bleu4_scores[label] = bleu4
        seqs.extend([scores[label], bleu1_scores[label], bleu2_scores[label], bleu3_scores[label], bleu4_scores[label]])
    metrics = dict()
    # metrics['ed_scores'] = scores
    # metrics['b1_scores'] = bleu1_scores
    # metrics['b2_scores'] = bleu2_scores
    # metrics['b3_scores'] = bleu3_scores
    # metrics['b4_scores'] = bleu4_scores
    metrics['sharp_acc'] = len([s for _, s in scores.items() if s == 1.]) / len(scores)
    metrics['mean_ed_score'] = np.mean(list(scores.values()))
    metrics['mean_b1_score'] = np.mean(list(bleu1_scores.values()))
    metrics['mean_b2_score'] = np.mean(list(bleu2_scores.values()))
    metrics['mean_b3_score'] = np.mean(list(bleu3_scores.values()))
    metrics['mean_b4_score'] = np.mean(list(bleu4_scores.values()))
    return results, metrics


if __name__ == '__main__':
    # results_dict = load_results()
    # results_extended_dict, eval_metrics = evaluation(results_dict)
    s = weighted_blue('refer', ['refer', 'zzzzz'], [1., 0, 0, 0], [0.6, 0.4])
    print(s)
    pass















def max_bleu_candidate(seq, full_candidate_list):
    """

    :param seq:
    :param full_candidate_list: candidates of form '+dafdesa#'
    :return:
    """
    candidate_list = sorted(full_candidate_list,
                            key=(lambda x: sentence_bleu([list(x)], list(seq), weights=(0.5, 0.5, 0, 0),
                                                         smoothing_function=SmoothingFunction().method3)))
    return candidate_list[-1]


def min_edi_dist_candidate(seq, full_candidate_list):
    """

    :param seq:
    :param full_candidate_list: candidates of form '+dafdesa#'
    :return:
    """
    candidate_list = sorted(full_candidate_list, key=(lambda x: edit_distance(''.join(x), ''.join(seq))))
    return candidate_list[0]