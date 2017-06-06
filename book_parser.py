'''
Splits a file into sentences, one for each line
'''
import sys
from nltk import sent_tokenize, RegexpTokenizer
import nltk

def main(args):
    filename = args[1]
    tokenizer = RegexpTokenizer(r'\w+')
    output_sentences = []
    with open(filename, 'r', encoding="latin-1") as f:
        raw = f.read()
        # raw = raw.decode('utf-8', 'ignore')
        raw = raw.replace('\n', ' ')
        sentences = sent_tokenize(raw)
        for sent in sentences:
            sent = tokenizer.tokenize(sent)
            output_sentences.append(' '.join(sent))

    with open(filename + str(".lines"), 'w') as o:
        for sent in output_sentences:
            if len(sent) > 0:
                o.writelines(sent + '\n')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python book_parser.py <filename>")
    else:
        main(sys.argv)
