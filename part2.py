from part1 import Node, create_filetree
import heapq
def big_dirs(files: list[tuple[str, int]], threshold: int) -> list[tuple[str, int]]:
    """
    Create ordered list of directories whose sizes are greater than threshold
    Input
    - files: list of files
    - threshold
    """
    root = create_filetree(files)

    heap = [(-1 * root.get_size(), root.val, '', root)]
    out = []
    while heap:
        neg_size, val, so_far, curr = heapq.heappop(heap)
        for next_node in curr.dirs:
            node = curr.get(next_node)
            size = node.get_size()
            if size < threshold:
                continue
            heapq.heappush(heap, (-1 * size, node.val, f"{so_far}/{node.val}", node))

        out.append((so_far, neg_size * -1))

    if out:
        out[0] = ('/', out[0][1])
    return out

if __name__ == "__main__":
    starter_files = [
        ("/a/b/x.txt", 100),
        ("/a/b/y.txt", 50),
        ("/a/c/z.txt", 30),
        ("/a/q.txt", 5),
        ("/ab/v.txt", 7),
        ("/d/w.txt", 10),
    ]
    print(big_dirs(starter_files, 100))
