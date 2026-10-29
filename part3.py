from part1 import create_filetree

def common_dir(path1: str, path2: str) -> str:
    files = [(path1, 0), (path2, 0)]
    root = create_filetree(files)
    out = '/'
    while len(root.dirs) == 1:
        key = list(root.dirs.keys())[0]
        if root.val != '~':
            out += f'{root.val}/'
        root = root.get(key)

    out += root.val if root.val != '~' else ''
    return out

if __name__ == "__main__":

    assert common_dir("/a/b/x.txt", "/a/c/z.txt") == "/a"
    assert common_dir("/a/b/x.txt", "/a/b/y.txt") == "/a/b"
    assert common_dir("/a/b/x.txt", "/d/w.txt") == "/"