FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

RUN apt update && apt install -y curl

RUN ${DL_PYTHON_EXECUTABLE} -m pip install --upgrade pip
RUN ${DL_PYTHON_EXECUTABLE} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY /requirements.txt .

RUN ${DL_PYTHON_EXECUTABLE} -m pip install  -r requirements.txt


# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/sapiens:0.0.1 -f Dockerfile .
# docker run -it gcr.io/viewo-g/piper/agent/runner/gpu/sapiens:0.0.1 bash
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/sapiens:0.0.1
