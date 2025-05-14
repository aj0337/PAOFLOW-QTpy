def divide_work(start: int, end: int, rank: int, size: int) -> tuple[int, int]:
    """Divide a 1-indexed range across MPI ranks."""
    total = end - start + 1
    chunk = total // size
    remainder = total % size
    i_start = start + rank * chunk + min(rank, remainder)
    i_end = i_start + chunk - 1
    if rank < remainder:
        i_end += 1
    return i_start, i_end
