from skipthoughts import skipthoughts as st
import sys
from sklearn.neighbors import KDTree


def process_text(text):
    s = text.replace("   ", "")
    sentences = s.split(".")[1:]
    return sentences


def get_sentences(fn):
    with open(fn) as f:
        s = f.read()
    return process_text(s)


def setup_kdtree(sentences, encoder):
    print 'embedding'
    embeds = encoder.encode(sentences)
    print 'making tree'
    return KDTree(embeds)

if __name__ == "__main__":
    encoder = st.Encoder(st.load_model())
    sents = get_sentences(sys.argv[1])
    tree = setup_kdtree(sents, encoder)
    while True:
        user_in = raw_input("enter text: ")
        s = process_text(user_in)
    embeds = encoder.encode(s)
    dist, ind = tree.query(embeds, k=1)
    new_text = [sents[i] for i in ind]
    print ". ".join(new_text)
