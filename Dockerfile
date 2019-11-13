FROM python:3.7
WORKDIR /workspace

# copy contents of repo to container
COPY . /workspace

# install requirements
RUN pip install pillow joblib glob2 tqdm torch scikit-learn pandas

# define entrypoint
ENTRYPOINT ["/workspace/docker-entrypoint.sh"]
RUN chmod +x /workspace/docker-entrypoint.sh

# get model weights and input scaler
RUN mkdir /workspace/bin && wget https://github.com/HHNK/HHNK-dijkprofiel/releases/download/v0.1-test/models.tar.gz
RUN tar -zxvf models.tar.gz
