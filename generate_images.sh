#Generate Dockerfile.

#!/bin/sh

 set -e

generate_docker() {
  docker run --rm kaczmarj/neurodocker:0.7.0 generate docker \
             --base ffrprep \
             --pkg-manager apt \
             --arg DEBIAN_FRONTEND=noninteractive \
             --miniconda \
               version=latest \
               conda_install="python=3.11 mne" \
               pip_install="pandas seaborn" \
               create_env='ffrprep' \
               activate=true \
            --copy . /home/ffrprep \
            --run-bash "source activate ffrprep && cd /home/package_name && pip install -e ." \
            --env IS_DOCKER=1 \
            --workdir '/tmp/' \
            --entrypoint "/neurodocker/startup.sh  ffrprep"
}

# generate files
generate_docker > Dockerfile

# check if images should be build locally or not
if [ $1 = local ]; then
    echo "docker image will be build locally"
    # build image using the saved files
    docker build -t package_name:local .
else
  echo "Image(s) won't be build locally."
fi            


