import argparse
from pypdf import PdfReader
from pypdf.errors import PdfStreamError
from multiprocessing import Pool
import os
import sys
import time
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
            token = self.content[start : self.ptr].upper()
            if not token:
                break
            yield token


def top_n(index, n):
    if n > len(index):
        raise IndexError(
            f"ERROR: requested {n} documents, but index only has {len(index)}"
        )
    li = []
    for key, value in index.items():
        li.append((key, value))

    li.sort(key=lambda x: x[1], reverse=True)
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


def tfs_from_files(filepaths):
    """Takes in list of filepaths, return list of
    tuples in the form (filename, tf hashmap)
    """

    def basename(filepath):
        return filepath.split("/")[-1]

    res = []
    for filepath in filepaths:
        res.append((basename(filepath), tf_from_file(filepath)))
    return res


def build_local_index(repo_path, max_size_in_bytes, process_cnt):
    """Parse all documents, calls `tf_from_file()` for each document,
    and return an index mapping filepath to its "local" tf hashmap
    """

    def split_task(filepaths, process_cnt):
        divider = len(filepaths) // process_cnt
        tasks = []
        for i in range(process_cnt):
            start = i * divider
            end = start + divider
            tasks.append(filepaths[start:end])
        tasks[-1].extend(filepaths[end:])
        return tasks

    print(f"INFO: parsing documents...")
    index = {}
    filepaths = []
    for root, subdir, files in os.walk(repo_path, topdown=False):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.getsize(filepath) > max_size_in_bytes:
                continue
            if not filepath.endswith(".pdf"):
                continue
            filepaths.append(filepath)

    tasks = split_task(filepaths, process_cnt)
    with Pool(process_cnt) as pool:
        res = pool.map(tfs_from_files, tasks)
    for item in res:
        for file, tf in item:
            index[file] = tf

    print(f"INFO: number of documents indexed: {len(index)}")
    return index


def get_results(index, prompt, max_results):
    """Given a prompt, return sorted list of relevant documents
    in the form (file path, tf-idf score)
    """
    results = []
    for filepath, tf_idf_table in index.items():
        tf_idf_sum = 0
        for word in prompt:
            tf_idf = tf_idf_table.get(word, 0)
            tf_idf_sum += tf_idf
        results.append((filepath, tf_idf_sum))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:max_results]


def build_global_index(index):
    print("INFO: building global index...")
    global_index = {}
    for tf_table in index.values():
        for key, val in tf_table.items():
            global_index[key] = global_index.get(key, 0) + 1
    return global_index


# tf-idf
def reassign_weights(index, global_index):
    print("INFO: calculating tf-idf for index...")
    num_documents = len(index)
    for file in index.keys():
        tf_table = index[file]
        for key in tf_table.keys():
            if key in global_index:
                tf_table[key] = tf_table[key] * math.log(
                    num_documents / global_index[key]
                )
        index[file] = tf_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", dest="max_size", type=str, action="store")
    parser.add_argument("-r", dest="max_results", type=int, action="store")
    parser.add_argument("-p", dest="process_cnt", type=int, action="store", default=1)
    parser.add_argument("prompt", nargs="+", action="store")
    return parser.parse_args()


def get_num_bytes(max_size):
    try:
        num, base = float(max_size[:-1]), max_size[-1]
    except ValueError:
        print(f"ERROR: bad size: {max_size}")
        sys.exit(1)
    if base == "B":
        base = 1
    elif base == "K":
        base = 2**10
    elif base == "M":
        base = 2**20
    elif base == "G":
        base = 2**30
    else:
        raise ValueError(f"ERROR: bad size: {max_size}")
    return num * base


def main():
    args = parse_args()
    max_size_in_bytes = get_num_bytes(args.max_size)

    print(f"INFO: skipping documents over: {max_size_in_bytes} bytes")
    print(f"INFO: received prompt: {args.prompt}")
    print(f"INFO: number of processes to start: {args.process_cnt}")

    prompt = [word.upper() for word in args.prompt]
    index = build_local_index("./papers-we-love", max_size_in_bytes, args.process_cnt)
    global_index = build_global_index(index)
    reassign_weights(index, global_index)
    results = get_results(index, prompt, args.max_results)

    print(f"RESULT: top {args.max_results} results for prompt: {' '.join(prompt)}")
    for i, j in enumerate(results[: args.max_results], 1):
        print(f"{i}. {j[0]}: {round(j[1], 4)}")


if __name__ == "__main__":
    main()
