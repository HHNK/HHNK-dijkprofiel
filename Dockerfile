FROM python:3.7
WORKDIR /workspace
COPY . /workspace
RUN pip install pillow joblib glob2 tqdm torch scikit-learn pandas
ENTRYPOINT ["/workspace/docker-entrypoint.sh"]
RUN chmod +x /workspace/docker-entrypoint.sh
RUN mkdir /workspace/bin && wget https://github.com/HHNK/HHNK-dijkprofiel/releases/download/v0.1-test/models.tar.gz
RUN tar -zxvf models.tar.gz
RUN ls -lah /workspace
