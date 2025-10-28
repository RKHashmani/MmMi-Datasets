FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

COPY requirements.txt /tmp/requirements.txt

RUN conda install git

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r /tmp/requirements.txt
