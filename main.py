import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import GPU_dataloader
import dill
import os
import beam_decoders
import after_beam

import numpy as np

from configurations import CONFIG


rr = after_beam.load_results()
s1, s2 = 0, 0
cf, cfc = dict(), dict()
cb, cbc = dict(), dict()
for label, seq in rr.items():
    if label[0] not in cf:
        cf[label[0]] = [1, 0]
    else:
        cf[label[0]][0] += 1
    if label[-1] not in cb:
        cb[label[-1]] = [1, 0]
    else:
        cb[label[-1]][0] += 1
    if label[0] == seq[0][0]:
        s1 += 1
        cf[label[0]][1] += 1
    if label[-1] == seq[1][-1]:
        s2 += 1
        cb[label[-1]][1] += 1
s1 /= len(rr)
s2 /= len(rr)

acc1 = np.mean([cf[label][1]/cf[label][0] for label in cf])
acc2 = np.mean([cb[label][1]/cb[label][0] for label in cb])


# train_files, val_files, test_files, max_H, max_W, max_seq_length = GPU_dataloader.data_split()
# max_W += 10
# max_H += 10

# dilled is expanded w and h. for 6-training, you don't really have to revert them.
train_files, val_files, test_files, max_H, max_W, max_seq_length = dill.load(open(CONFIG.file_list, 'rb'))

# for invert training, flip the file names to access the inverted data
if CONFIG.inverted_training is True:
    t_f, v_f, te_f = list(), list(), list()
    for file in train_files:
        t_f.append(file[:-4][::-1] + file[-4:])
    for file in val_files:
        v_f.append(file[:-4][::-1] + file[-4:])
    for file in test_files:
        te_f.append(file[:-4][::-1] + file[-4:])
    train_files, val_files, test_files = t_f, v_f, te_f

train_sequence = GPU_dataloader.ShorthandGenerationSequence(train_files, max_H, max_W, aug_types=9,
                                                            max_label_leng=max_seq_length, channels=1)
val_sequence = GPU_dataloader.ShorthandGenerationSequence(val_files, max_H, max_W, aug_types=1,
                                                          max_label_leng=max_seq_length, channels=1)
test_sequence = GPU_dataloader.ShorthandGenerationSequence(test_files, max_H, max_W, aug_types=1,
                                                           max_label_leng=max_seq_length, channels=1)


def def_model(H, W):
    # feature_extractor = keras.applications.Xception(weights=None, include_top=False, input_shape=(131, 214, 3))

    # all in NWHC
    feature_extractor = keras.models.Sequential()
    # input: 168 * 101 images with 1 channel -> (168, 101, 1) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    feature_extractor.add(Conv2D(32, (3, 3), activation='relu', input_shape=(H, W, 1), padding='same',
                                 kernel_initializer='he_normal'))
    feature_extractor.add(keras.layers.BatchNormalization())
    feature_extractor.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    feature_extractor.add(MaxPooling2D(pool_size=2))
    # model.add(Dropout(0.1))

    feature_extractor.add(keras.layers.BatchNormalization())
    feature_extractor.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    feature_extractor.add(keras.layers.BatchNormalization())
    feature_extractor.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    feature_extractor.add(MaxPooling2D(pool_size=2))

    feature_extractor.add(keras.layers.BatchNormalization())
    feature_extractor.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    feature_extractor.add(keras.layers.BatchNormalization())
    feature_extractor.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    feature_extractor.add(MaxPooling2D(pool_size=2))

    feature_extractor.add(keras.layers.BatchNormalization())
    feature_extractor.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    feature_extractor.add(keras.layers.BatchNormalization())
    feature_extractor.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    feature_extractor.add(MaxPooling2D(pool_size=2))

    feature_extractor.add(keras.layers.BatchNormalization())
    feature_extractor.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    feature_extractor.add(keras.layers.BatchNormalization())
    feature_extractor.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
    feature_extractor.add(keras.layers.GlobalMaxPooling2D())

    feature_extractor.add(keras.layers.core.Reshape(target_shape=[-1, 1, 512]))

    feature_extractor.add(Flatten())
    feature_extractor.add(Dropout(CONFIG.drop_out))
    feature_extractor.add(Dense(2048, activation='relu'))
    feature_extractor.add(Dropout(CONFIG.drop_out))
    feature_extractor.add(Dense(CONFIG.RNN_size, activation='relu'))
#     feature_extractor.add(Dropout(0.25))

#     feature_extractor.add(Dense(CONFIG.num_category, activation='softmax'))

    img = keras.layers.Input(shape=(H, W, 1), dtype='float32')

    img_f = feature_extractor(img)
#     img_f = keras.layers.GlobalAveragePooling2D()(img_f)
#     img_f = Dense(CONFIG.RNN_size, activation='relu')(img_f)

    x_context = keras.layers.Input(shape=[None, ], dtype='int32', name='x_context')
    x_embedding_layer = keras.layers.Embedding(input_dim=CONFIG.vocabulary_size,
                                               output_dim=CONFIG.embedding_size,
                                               trainable=True)
    x_seq_embedding = x_embedding_layer(x_context)
    h_t = keras.layers.GRU(CONFIG.RNN_size, name='h_t', kernel_initializer='xavier')\
        (x_seq_embedding, initial_state=img_f)
    h_t_dropped = keras.layers.Dropout(rate=CONFIG.drop_out)(h_t)
    predictions = Dense(CONFIG.vocabulary_size, activation='softmax')(h_t_dropped)
    model = keras.Model(inputs=[img, x_context], outputs=predictions)

    return model


def train_model(model):
    """

    :param model:
    :return:
    """

    print("train on " + str(len(train_files)) + " images. val on " + str(len(val_files)) + " images.")
    print("max size:" + str(max_H) + " * " + str(max_W))
    # print("contains: " + CONFIG.contains)
    # print("positive samples %:" + str(GPU_dataloader.pos_tag_proportion(train_files)))

    optimizer = keras.optimizers.adam(lr=CONFIG.learning_rate, clipnorm=CONFIG.clip_threshold)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    csvlogger = keras.callbacks.CSVLogger(CONFIG.log_path, separator=',', append=False)

    check_point_path = os.path.join(os.path.join('.', CONFIG.run_index), r'{epoch:02d}.hdf5')
    checkpointer = keras.callbacks.ModelCheckpoint(check_point_path,
                                                   monitor='loss', period=1, save_best_only=False, verbose=1)

    history = model.fit_generator(generator=train_sequence, epochs=CONFIG.max_train_epochs,
                                  validation_data=val_sequence, callbacks=[csvlogger, checkpointer], shuffle=True)

    dill.dump(history.history, open(CONFIG.run_index + '_history', 'wb'))


def evaluate_from_file():
    beam_results = after_beam.load_results()
    """
    with open(CONFIG.run_index + '_full_result.csv', 'w') as fout:
        fout.write('label,forward,backward \n')
        for label, seqs in beam_results.items():
            fout.write(label + ',' + seqs[0] + ',' + seqs[1] + '\n')
    """

    candidate_list = [filename[:-4] for filename in train_files + val_files + test_files]

    final_result = dict()
    metric_weights = CONFIG.metric_weights
    for i, (label, seqs) in enumerate(beam_results.items()):
        forw_hyp = beam_results[label][0]
        back_hyp = beam_results[label][1]
        final_result[label] = [after_beam.best_candidate(forw_hyp, back_hyp, candidate_list,
                                                         metric_weights=metric_weights)]
        print(str(i + 1) + ' seqs decoded. previous seq:' + label)

    # evaluate
    eval_f_results, eval_f_metrics = after_beam.evaluation(beam_results)
    beam_results_back = {label: [beam_results[label][1]] for label in beam_results}
    eval_b_results, eval_b_metrics = after_beam.evaluation(beam_results_back)

    # eval_results, eval_metrics = after_beam.evaluation(beam_results)
    eval_results, eval_metrics = after_beam.evaluation(final_result)

    # export result
    with open(CONFIG.run_index + '_result.csv', 'w') as fout:
        fout.write('label,hypothesis,edit_dist,bleu1,bleu2,bleu3,bleu4 \n')
        for label, seqs in eval_results.items():
            fout.write(label + ',' + seqs[0] + ',' + str(seqs[1]) + ',' + str(seqs[2]) + ',' + str(seqs[3]) + ',' +
                       str(seqs[4]) + ',' + str(seqs[5]) + '\n')
    with open('rawscores.txt', 'w') as raw_out:
        raw_out.write('metrics:')
        raw_out.write(str(eval_metrics))
        raw_out.write('\nraw_forward_metrics:')
        raw_out.write(str(eval_f_metrics))
        raw_out.write('\nraw_backward_metrics:')
        raw_out.write(str(eval_b_metrics))

    pass


# evaluate_from_file()
# print('evaluation terminated.')
# random.seed(43970)
'''
rlts = after_beam.load_results()
rr = dict()
for label, seq in rlts.items():
    rr[label] = [seq[1]]
_, metrics = after_beam.evaluation(rr)
'''
files = os.listdir(CONFIG.final_folder)
ss = dict()
for file in files:
    rlts = after_beam.load_results(os.path.join(CONFIG.final_folder, file), CONFIG.backward_result_path)
    _, metrics = after_beam.evaluation(rlts)
    ss[file] = metrics['mean_ed_score']
    print('++')

pass



print('start running execution indexed:' + CONFIG.run_index)

modell = None

if not os.path.exists(os.path.join('.', CONFIG.run_index)):
    os.mkdir(os.path.join('.', CONFIG.run_index))

# acquire model
if not os.path.exists(CONFIG.model_path):
    modell = def_model(max_H, max_W)
    # 131, 214 +10
    train_model(modell)
    print('model defined.')
    modell.save(filepath=CONFIG.model_path)
    evals = modell.evaluate_generator(test_sequence)
    print('evaluation after training:')
    print(str(evals))
    print('model saved.')
else:
    print('loading serialized model...')
    modell = keras.models.load_model(filepath=CONFIG.model_path)
    print('model loaded.')
    evals = modell.evaluate_generator(test_sequence)
    print('evaluations of model:')
    print(str(evals))

results = beam_decoders.plain_beam(val_files, max_H, max_W, modell, k=5)
# export result
with open(CONFIG.run_index + '_result.csv', 'w') as fout:
    fout.write('label,output,edit_best,score \n')
    for label, seqs in results.items():
        fout.write(label + ',' + seqs[0] + '\n')
