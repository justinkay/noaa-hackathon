FROM python:3.7

ADD . /

RUN apt update
RUN apt install --yes libgl1-mesa-glx

RUN python3.7 -m pip install --upgrade pip

RUN python3.7 -m pip install -r requirements.txt -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.6/index.html -f https://download.pytorch.org/whl/torch_stable.html

RUN python3.7 -m pip install -e .

CMD ["python3.7", "run_test.py"]