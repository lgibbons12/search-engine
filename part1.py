from collections import defaultdict

class Node:
    def __init__(self, name) -> None:
        self.val = name
        self.dirs = dict()
        self.subtree_size = 0

    def add_dir(self, dir) -> None:
        self.dirs[dir.val] = dir

    def get(self, name: str):
        return self.dirs.get(name, None)

    def add_size(self, size: int) -> None:
        self.subtree_size += size

    def get_size(self) -> int:
        return self.subtree_size

    def move(self, src_dir: str, dst_parent: str) -> None:
        # find src dir
        splits = src_dir.split('/')
        src_node = self
        for i in range(1, len(splits) - 1):
            split = splits[i]
            if split not in src_node.dirs:
                raise ValueError("Invalid source directory")

            src_node = src_node.get(split)

        src_act_node = src_node.get(splits[-1])
        del src_node.dirs[splits[-1]]

        # add to dst_parent
        splits = src_dir.split('/')
        dst_node = self
        for i in range(1, len(splits) - 1):
            split = splits[i]
            if split not in dst_node.dirs:
                raise ValueError("Invalid source directory")

            dst_node = dst_node.get(split)

        dst_node.add_dir(src_act_node)


        

    def __str__(self) -> str:
        output = self.val
        for dir in self.dirs:
            output += '\n\t' + str(self.dirs[dir])
            print(f"{output=}")
        return output


def create_filetree(files: list[tuple[str, int]]):
    root = Node('~')
    # create a tree
    for filepath, size in files:
        splits = filepath.split('/')
        curr_node = root
        for i in range(1, len(splits)):
            split = splits[i]
            if split not in curr_node.dirs:
                split_node = Node(split)     
                curr_node.add_dir(split_node)

            curr_node.add_size(size)
            curr_node = curr_node.get(split)
    return root

def dir_size(files: list[tuple[str, int]], path: str) -> int:
    """
    Returns the size of all files in path's directory and subdirectories
    given the filesystem provided by files

    Input:
    - files: list of (abs_path, size) files
    - path: directory

    Output:
    - size of all files in that directory including its subdirectories
    """
    root = create_filetree(files)

    # walk the path
    splits = path.split('/')
    curr_node = root
    for i in range(1, len(splits)):
        if splits[i] == '':
            continue
        res = curr_node.get(splits[i])
        if res is None:
            return 0
        curr_node = res
    return curr_node.get_size()

if __name__ == "__main__":
    starter_files = [
        ("/a/b/x.txt", 100),
        ("/a/b/y.txt", 50),
        ("/a/c/z.txt", 30),
        ("/a/q.txt", 5),
        ("/ab/v.txt", 7),
        ("/d/w.txt", 10),
    ]
    print(dir_size(starter_files, '/a'))

