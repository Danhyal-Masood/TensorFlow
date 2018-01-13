from __future__ import absolute_import, division, print_function

import os
from six import moves
import ssl
import tflearn
from tflearn.data_utils import *
import urllib.request

path = "British_Cities.txt"
if not os.path.isfile(path):
    context = ssl._create_unverified_context()
    urllib.request.urlretrieve("https://raw.githubusercontent.com/Krypto12/TensorFlow/master/City_Names.txt", path)

maxlen = 20

X, Y, char_idx =textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

g = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')
g = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

m = tflearn.SequenceGenerator(g, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_British_Cities')

ls=[]
for i in range(100):
    seed = random_sequence_from_textfile(path, maxlen)
    m.fit(X, Y, validation_set=0.1, batch_size=128,
          n_epoch=1, run_id='British_Cities')
    print("-- TESTING...")
    print("-- Test with temperature of 1.2 --")
    ls.append(m.generate(30, temperature=1.2, seq_seed=seed))
    print("-- Test with temperature of 1.0 --")
    ls.append(m.generate(30, temperature=1.0, seq_seed=seed))
    print("-- Test with temperature of 0.5 --")
    ls.append(m.generate(30, temperature=0.5, seq_seed=seed))

with open("Generated_city_names.txt","w") as e:
    for x in ls:
        e.write("{}\n".format(x))
