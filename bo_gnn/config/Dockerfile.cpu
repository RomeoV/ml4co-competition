FROM continuumio/miniconda3
MAINTAINER Romeo Valentin <rvalentin@student.ethz.ch>

COPY conda.yaml .
RUN \
  conda install -c conda-forge -y mamba \
  && mamba env update -n base -f conda.yaml \
  && mamba clean -a -y
