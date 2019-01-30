import os


def list_all_folders(path):
    filenames = os.listdir(path)  # get all files' and folders' names in the current directory

    result = []
    for filename in filenames:  # loop through all the files and folders
        if filename in 'data':
            continue
        if filename in '.git':
            continue
        if filename in '__pycache__':
            continue
        if os.path.isdir(os.path.join(os.path.abspath(path), filename)):
            result.extend(list_all_folders(os.path.join(os.path.abspath(path), filename)))
            result.append(os.path.join(os.path.abspath(path), filename))

    return result


with open('.env', 'w') as f:
    # add working directories
    # absolute path of working directory
    wd = os.path.dirname(os.path.realpath(__file__))

    # list of every subdirectory
    subdirs = list_all_folders(wd)

    # build the PYTHONPATH string
    python_path = 'PYTHONPATH=' + wd

    for subdir in subdirs:
        python_path += ':' + subdir

    print('break')

    f.write(python_path)
    # TODO make config file ready
    # f.write('\n')
    # f.write('LUIGI_CONFIG_PATH=' + wd + '/luigi.cfg')

