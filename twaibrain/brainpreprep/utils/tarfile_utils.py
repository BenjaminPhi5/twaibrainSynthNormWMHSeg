import tarfile

def get_tar_file_structure(tar_path, verbose=False):
    files_list = []
    try:
        # Attempt to open the file as a gzip-compressed tar file
        with tarfile.open(tar_path, 'r:gz') as tar:
            print("Opened as gzip-compressed tar file")
            for member in tar.getmembers():
                if verbose:
                    print(member.name)
                files_list.append(member.name)
    except tarfile.ReadError:
        # If it fails, try to open it as a regular tar file
        try:
            with tarfile.open(tar_path, 'r') as tar:
                print("Opened as a regular tar file")
                for member in tar.getmembers():
                    if verbose:
                        print(member.name)
                    files_list.append(member.name)
        except Exception as e:
            print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
    return files_list
