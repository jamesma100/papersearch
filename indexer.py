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
            token = self.content[start:self.ptr]
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

def build_index(filepath):
    try:
        reader = PdfReader(filepath)
    except PdfStreamError:
        return None
    index = {}
    for page in reader.pages:
        content = page.extract_text()
        lexer = Lexer(content)
        for token in lexer.next_token():
            index[token] = index.get(token, 0) + 1
    return index

def tf_from_file(filepath):
    try:
        reader = PdfReader(filepath)
    except PdfStreamError:
        return None
    index = {}
    for page in reader.pages:
        content = page.extract_text()
        lexer = Lexer(content)
        for token in lexer.next_token():
            index[token] = index.get(token, 0) + 1
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


def main():
    index = {}
    num_documents = 0

    for root, subdir, files in os.walk("./papers-we-love", topdown=False):
        for file in files:
            filepath = os.path.join(root, file)
            # only read docs smaller than 400KB
            if os.path.getsize(filepath) > 400 * 2**10:
                continue
            if not filepath.endswith(".pdf"):
                continue
            num_documents += 1
            tf_table = tf_from_file(filepath)
            if not tf_table:
                print("ERROR: cannot read pdf stream for: ", file)
            else:
                index[file] = tf_table

    global_index = build_global_index(index)
    reassign_weights(index, global_index, num_documents)
    for filepath, tf_table in index.items():
        print(f"{filepath}:")
        for i, j in top_n(tf_table, 10):
            print(f"    {i}: {round(j, 3)}")
    
if __name__ == "__main__":
    main()
