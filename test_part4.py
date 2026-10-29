from part1 import create_filetree, dir_size

starter_files = [
    ("/a/b/x.txt", 100),
    ("/a/b/y.txt", 50),
    ("/a/c/z.txt", 30),
    ("/a/q.txt", 5),
    ("/ab/v.txt", 7),
    ("/d/w.txt", 10),
]

if __name__ == "__main__":
    root = create_filetree(starter_files)
    root.move("/a/c", '/d')
    assert 
    