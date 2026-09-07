import os
import sys

def create_symlinks(file_list, target_folder, verbose=False):
    """
    Creates a folder with symbolic links to the given list of files.

    Args:
        file_list (list of str): List of file paths to create symlinks for.
        target_folder (str): Path to the folder where symlinks will be created.

    Returns:
        None
    """
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    for file_path in file_list:
        try:
            # Check if the file exists
            if not os.path.isfile(file_path):
                print(f"Skipping: {file_path} (not a valid file)")
                continue
            
            # Get the file name and create the symlink in the target folder
            file_name = os.path.basename(file_path)
            symlink_path = os.path.join(target_folder, file_name)

            if os.path.exists(symlink_path):
                if verbose:
                    print(f"Skipping: {file_name} (symlink already exists)")
                continue

            os.symlink(os.path.abspath(file_path), symlink_path)
            if verbose:
                print(f"Created symlink: {symlink_path} -> {file_path}")
        except Exception as e:
            print(f"Error creating symlink for {file_path}: {e}")
            
            
def test_loadable():
    print("loadable")
    print(os.listdir("."))
