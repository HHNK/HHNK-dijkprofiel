FROM pytorch/pytorch

RUN mkdir /workspace
WORKDIR /workspace
COPY . /workspace/.
RUN pip install pillow joblib glob tqdm operator
ENTRYPOINT ["/workspace/docker-entrypoint.sh"]

