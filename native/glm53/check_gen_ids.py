"""What do the candidate generated ids decode to? Pin down the junk-reply source."""
from tokenizers import Tokenizer

tok = Tokenizer.from_file("/mnt/apex/rick/vybforge/glm53-checkpoint/tokenizer.json")
for ids in ([576], [1283], [576, 576, 576], [1283, 1283], [31], [63], [30]):
    print(ids, "->", repr(tok.decode(ids)))
# and the vocab size + whether 576/1283 are real tokens
print("vocab size:", tok.get_vocab_size())
for t in (576, 1283, 6722, 315):
    print(t, "=", repr(tok.id_to_token(t)))
