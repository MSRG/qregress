FROM  --platform=linux/amd64 debian:12
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
COPY ./ ./ 

# Install Conda via shell script. It is is not in any main repo :( 
RUN mkdir -p /opt/conda 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/conda/miniconda.sh 
RUN bash /opt/conda/miniconda.sh -b -p /opt/miniconda 
# Install your environment.yml deps into base env 
# Uncomment once you are ready to start productionizing the image 

RUN . /opt/miniconda/bin/activate && conda install -y python=3.9 && conda env update --name base --file qchem.yml 
# RUN . /opt/miniconda/bin/activate && conda install -y python=3.9
# RUN cat uniq.txt | sort | xargs -n 1 /opt/miniconda/bin/python -m pip install 
# Install your softwares 
# Uncomment once you have software to install 
# Run the software in conda base environment 
