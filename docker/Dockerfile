FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

COPY . .

RUN pip install --upgrade pip && pip install -e .

CMD ["python", "src/train.py"]
