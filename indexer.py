from pypdf import PdfReader
from pypdf.errors import PdfStreamError
import os
import sys


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

# tf-idf
def build_global_index(indexes):
    print("DEBUG: building global index")
    global_index = {}
    for index in indexes.values():
        for key, val in index.items():
            global_index[key] = global_index.get(key, 0) + 1
    return global_index

def reassign_weights(indexes, global_index):
    print("DEBUG: reassigning weights")
    for file in indexes.keys():
        index = indexes[file]
        for key in index.keys():
            if key in global_index:
                index[key] = index[key] // global_index[key]
        indexes[file] = index


def main():
    indexes = {}

    for root, subdir, files in os.walk("./papers-we-love", topdown=False):
        for file in files:
            filepath = os.path.join(root, file)
            # only read docs smaller than 400KB
            if os.path.getsize(filepath) > 400 * 2**10:
                continue
            if not filepath.endswith(".pdf"):
                continue
            index = build_index(filepath)
            if not index:
                print("ERROR: cannot read pdf stream for: ", file)
            else:
                indexes[file] = index

    global_index = build_global_index(indexes)
    reassign_weights(indexes, global_index)
    for filepath, index in indexes.items():
        print(f"{filepath}:")
        for i in top_n(index, 10):
            print(f"    {i}")
    
if __name__ == "__main__":
    main()
