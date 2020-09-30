FROM anyscale/ray:0.8.7-gpu

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip && ./aws/install && rm awscliv2.zip && rm -rf ./aws

RUN apt-get update && apt-get install -y vim htop && apt-get clean

RUN conda install -y faiss-cpu pytorch torchvision -c pytorch && conda clean -y --all

COPY requirements.txt .
COPY train_data.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install -U --no-cache-dir https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.9.0.dev0-cp37-cp37m-manylinux1_x86_64.whl
RUN pip install -U pip torch==1.4.0 torchvision==0.5.0 wandb google-api-python-client==1.7.8
RUN pip install git+https://github.com/NVIDIA/apex.git@4ef930c1c884fdca5f472ab2ce7cb9b505d26c1a
RUN pip install -U git+git://github.com/huggingface/transformers.git@3a7fdd3f5214d1ec494379e7c65b4eb08146ddb0

RUN mkdir -p /root/demo

# Copy workspace dir over
COPY . /root/demo

# Install the movie_recs package
RUN cd /root/demo && pip install .
