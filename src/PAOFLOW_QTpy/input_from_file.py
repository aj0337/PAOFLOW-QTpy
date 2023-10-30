# TODO Ask Marco what how this recieves input file. CLI option? Might not be required?

import sys


def input_from_file(iunit):
    """
    This function checks program arguments and, if an input file is present,
    attaches input unit iunit to the specified file.

    Parameters:
        iunit (int): Input unit number.

    Returns:
        file: File object if the input file is successfully opened, None otherwise.
    """
    try:
        # Look for the '--input' or '-i' flag followed by the input file name
        input_flag_index = sys.argv.index(
            '--input') if '--input' in sys.argv else sys.argv.index('-i')
        input_file = sys.argv[input_flag_index + 1]

        return open(input_file, 'r')
    except (ValueError, FileNotFoundError, IndexError):
        print("Error: Input file not found or invalid command line arguments.")
        return None


# Example usage:
# iunit = 10  # Specify the input unit number
# input_file_handle = input_from_file(iunit)
# if input_file_handle is not None:
#     # Read data from input file using input_file_handle
#     # ...
#     input_file_handle.close()  # Close the file when done
