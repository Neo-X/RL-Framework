

import subprocess

def get_git_revision_hash():
    out = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
    return out.decode('UTF-8')

def get_git_revision_short_hash():
    out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return out.decode('UTF-8')



if __name__ == '__main__':
    
    print ("get_git_revision_hash: ", str(get_git_revision_hash()))
    
    print ("get_git_revision_short_hash: ", get_git_revision_short_hash())