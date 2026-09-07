import zipfile
import os

def get_zip_fileslist(zip_path, verbose=False):
    """
    return the list of files in a zip file
    """
    files_list = []
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        # List all files in the zip file
        file_list = zip_file.namelist()
        
        for file in file_list:
            if verbose:
                print(file)
            files_list.append(file)
    return files_list

def extract_specific_file(zip_path, file_name, output_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Check if the file exists in the zip archive
            if file_name in zip_file.namelist():
                # Extract the specific file
                with zip_file.open(file_name) as specific_file:
                    # Read the contents of the specific file
                    content = specific_file.read()
                    # Save the content to a new file
                    with open(output_path, 'wb') as output_file:
                        output_file.write(content)
                # print(f"File '{file_name}' extracted to '{output_path}'")
            else:
                print(f"File '{file_name}' not found in the zip archive")
    except Exception as e:
        print(f"An error occurred: {e}")

def extract_multiple_files(zip_path, file_paths, output_folder):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Check if the file exists in the zip archive
            for fp in file_paths:
                if fp in zip_file.namelist():
                    filename = fp.split("/")[-1]
                    # Extract the specific file
                    with zip_file.open(fp) as specific_file:
                        # Read the contents of the specific file
                        content = specific_file.read()
                        # Save the content to a new file
                        with open(os.path.join(output_folder, filename), 'wb') as output_file:
                            output_file.write(content)
                else:
                    print(f"File '{fp}' not found in the zip archive")
    except Exception as e:
        print(f"An error occurred: {e}")
