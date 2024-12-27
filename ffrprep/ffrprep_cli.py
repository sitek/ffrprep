import argparse
import os

# define parser to collect required inputs
def get_parser():

    __version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    '_version.py')).read()



    parser = argparse.ArgumentParser(description='A standardized and robust (pre-) processing pipeline for FFR data.')
    parser.add_argument('-v', '--version', action='version',
                        version='ffrprep version {}'.format(__version__))
    return parser


# define the CLI
def run_ffrprep():

    # get arguments from parser
    args = get_parser().parse_args()

    # special variable set in the container
    if os.getenv('IS_DOCKER'):
        exec_env = 'singularity'
        cgroup = Path('/proc/1/cgroup')
        if cgroup.exists() and 'docker' in cgroup.read_text():
            exec_env = 'docker'
    else:
        exec_env = 'local'


# run the CLI
if __name__ == "__main__":

    run_ffrprep()
