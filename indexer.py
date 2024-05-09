import argparse
from pypdf import PdfReader
from pypdf.errors import PdfStreamError
import os
import sys
import math

class Lexer:
    def __init__(self, content):
        self.content = content.strip()
        self.ptr = 0

    def next_token(self):
        while self.ptr < len(self.content):
            start = -1
            # find start of token
            while self.ptr < len(self.content) and not self.content[self.ptr].isalnum():
                self.ptr += 1
            start = self.ptr
            # find end of token
            while self.ptr < len(self.content) and self.content[self.ptr].isalnum():
                self.ptr += 1
            token = self.content[start:self.ptr].upper()
            if not token:
                break
            yield token

def top_n(index, n):
    if n > len(index):
        raise IndexError("ERROR: n is out of bounds")
    li = []
    for key, value in index.items():
        li.append((key, value))

    li.sort(key = lambda x: x[1], reverse=True)
    for i in li[:n]:
        yield i

def block_stderr():
    sys.stderr = open(os.devnull, "w")

def unblock_stderr():
    sys.stderr = sys.__stderr__

def tf_from_file(filepath):
    block_stderr()
    try:
        reader = PdfReader(filepath)
    except PdfStreamError:
        unblock_stderr()
        return None
    index = {}
    for page in reader.pages:
        content = page.extract_text()
        lexer = Lexer(content)
        for token in lexer.next_token():
            index[token] = index.get(token, 0) + 1
    unblock_stderr()
    total = sum(index.values())
    for key in index:
        index[key] /= total
    return index

def build_global_index(index):
    print("INFO: building global index")
    global_index = {}
    for tf_table in index.values():
        for key, val in tf_table.items():
            global_index[key] = global_index.get(key, 0) + 1
    return global_index

# tf-idf
def reassign_weights(index, global_index, num_documents):
    print("INFO: calculating tf-idf for index")
    for file in index.keys():
        tf_table = index[file]
        for key in tf_table.keys():
            if key in global_index:
                tf_table[key] = tf_table[key] * math.log(num_documents / global_index[key])
        index[file] = tf_table

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", dest="max_size", type=str, action="store")
    parser.add_argument("-r", dest="max_results", type=int, action="store")
    parser.add_argument("prompt", nargs='+', action="store")
    return parser.parse_args()


def get_num_bytes(max_size):
    try:
        num, base = int(max_size[:-1]), max_size[-1]
    except ValueError:
        print("ERROR: bad size")
        sys.exit(1)
    if base == "B":
        base = 1
    elif base == "K":
        base = 2**10
    elif base == "M":
        base = 2**20
    elif base == "G":
        base = 2 ** 30
    else:
        raise ValueError("ERROR: bad size")
    return num * base



def main():
    args = parse_args()
    index = {}
    num_documents = 0

    max_size = get_num_bytes(args.max_size)
    print(f"INFO: skipping documents over: {max_size}B")
    print(f"INFO: received prompt: {args.prompt}")
    prompt = [word.upper() for word in args.prompt]

    for root, subdir, files in os.walk("./papers-we-love", topdown=False):
        for file in files:
            filepath = os.path.join(root, file)
            # only read docs smaller than 1Mb
            if os.path.getsize(filepath) > max_size:
                continue
            if not filepath.endswith(".pdf"):
                continue
            num_documents += 1
            tf_table = tf_from_file(filepath)
            if not tf_table:
                print("ERROR: cannot read pdf stream for: ", file)
            else:
                index[file] = tf_table

    print(f"INFO: number of documents to index: {num_documents}")
    global_index = build_global_index(index)
    reassign_weights(index, global_index, num_documents)

    results = []
    for filepath, tf_idf_table in index.items():
        tf_idf_sum = 0
        for word in prompt:
            tf_idf = tf_idf_table.get(word, 0)
            tf_idf_sum += tf_idf
        results.append((filepath, tf_idf_sum))
    results.sort(key = lambda x: x[1], reverse=True)
    print(f"RESULT: top {args.max_results} results for prompt: {' '.join(prompt)}")
    for i, j in enumerate(results[:args.max_results], 1):
        print(f"{i}. {j[0]}: {round(j[1], 4)}")
    
if __name__ == "__main__":
    main()
