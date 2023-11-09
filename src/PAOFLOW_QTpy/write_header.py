def write_header(msg):
    """
    Print out the given header message msg.

    Parameters:
        msg (str): Header message to be printed.
    """

    if len(msg) >= 66:
        raise ValueError(f"message longer than 66 characters: {msg}")

    separator = '=' * 70

    print(f"  {separator}")
    print(f"  =  {msg:^66s}=")
    print(f"  {separator}")
