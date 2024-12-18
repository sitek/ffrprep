#Generate Dockerfile.

#!/bin/sh

 set -e

generate_docker() {
  docker run --rm kaczmarj/neurodocker:0.7.0 generate docker \
             --base BaseImage \
             --pkg-manager apt \
             --arg DEBIAN_FRONTEND=noninteractive \
             --miniconda \
               version=latest \
               conda_install="python=PythonVersion PythonPackages" \
               pip_install="PythonPackages" \
               create_env='package_name' \
               activate=true \
            --copy . /home/package_name \
            --run-bash "source activate package_name && cd /home/package_name && pip install -e ." \
            --env IS_DOCKER=1 \
            --workdir '/tmp/' \
            --entrypoint "/neurodocker/startup.sh  package_name"
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


