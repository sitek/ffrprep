import argparse


# define parser to collect required inputs
def get_parser():

    __version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    '_version.py')).read()



    parser = argparse.ArgumentParser(description='Add description of package here')
    parser.add_argument('-v', '--version', action='version',
                        version='package_name version {}'.format(__version__))
    return parser


# define the CLI
def run_package_name():

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

    run_package_name()
