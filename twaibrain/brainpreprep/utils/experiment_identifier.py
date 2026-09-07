"""
just creates a file in the filesystem to tell any processing scripts that the given experiment has been run on this directory
"""
import os
import argparse

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)
        
        
def check_exp_id(experiment_id_path):
    """
    returns False if the id path exists
    returns True if it does not exist, and creates the file
    """
    if os.path.exists(experiment_id_path):
        print("experiment has been run before")
        return False
    if not experiment_id_path.endswith(".expid"):
        experiment_id_path += ".expid"
    if os.path.exists(experiment_id_path):
        print("experiment has been run before")
        return False
    
    basedir = os.path.dirname(experiment_id_path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
        
    touch(experiment_id_path)
    
    return experiment_id_path

def remove_experiment_ids(experiment_folder, exp_name):
    """
    removes all .expid files from a given folder and its nested subdirectories
    """
    print(f"REMOVING ALL EXPERIMENT IDS (.expid files) for folder: {experiment_folder}")
    for (root, folder, files) in os.walk(experiment_folder):
        for file in files:
            if file.startswith(exp_name) and file.endswith(".expid"):
                filepath = os.path.join(root, file)
                os.remove(filepath)

def construct_parser():
    parser = argparse.ArgumentParser(description = "remove all experiment ids from a given folder")
    
    # folder arguments
    parser.add_argument('--dir', required=True, type=str)
    parser.add_argument('--name', required=True, type=str, help="name that the experiment file must start with")
    return parser                

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()
    remove_experiment_ids(args.dir, args.name)
