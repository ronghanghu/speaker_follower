import pickle
from tasks.R2R.utils import read_vocab, Tokenizer
from tasks.R2R.vocab import TRAIN_VOCAB

vocab = read_vocab(TRAIN_VOCAB)
tok = Tokenizer(vocab=vocab)

with open('data/working_data/train_objects_by_word.pickle', 'rb') as file:
  data = pickle.load(file)

print(tok.encode_sentence(data[5487][0]['words_objects'][0][0]['name']))

for key in data:
  for elem in data[key]:
    for j in elem['words_objects']:
      for z in j:
        if 1 in tok.encode_sentence(z['name'])[0]:
          print(z['name'])
