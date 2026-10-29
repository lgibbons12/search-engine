from part1 import dir_size


# each entry: (absolute file path, size in bytes)
starter_files = [
    ("/a/b/x.txt", 100),
    ("/a/b/y.txt", 50),
    ("/a/c/z.txt", 30),
    ("/a/q.txt", 5),
    ("/ab/v.txt", 7),
    ("/d/w.txt", 10),
]

empty_files = []

if __name__ == "__main__":
    assert dir_size(starter_files, "/a") == 185
    assert dir_size(starter_files, "/a/b") == 150
    assert dir_size(starter_files, "/") == 202
    assert dir_size(starter_files, "/zzz") == 0
    assert dir_size(empty_files, '/') == 0
    