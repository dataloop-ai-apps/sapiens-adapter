FROM hub.dataloop.ai/dtlpy-runner-images/gpu:python3.10_cuda11.8_pytorch2

RUN apt update && apt install -y curl wget

RUN mkdir -p /tmp/app/weights

RUN ${DL_PYTHON_EXECUTABLE} -m pip install --upgrade pip
RUN ${DL_PYTHON_EXECUTABLE} -m pip install torch --index-url https://download.pytorch.org/whl/cu118

# Download weights into image
RUN wget -q -O /tmp/app/weights/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2 \
    https://storage.googleapis.com/model-mgmt-snapshots/sapiens/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2
RUN wget -q -O /tmp/app/weights/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2 \
    https://storage.googleapis.com/model-mgmt-snapshots/sapiens/sapiens_0.6b_goliath_best_goliath_mIoU_7777_epoch_178_torchscript.pt2
RUN wget -q -O /tmp/app/weights/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2 \
    https://storage.googleapis.com/model-mgmt-snapshots/sapiens/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2

# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/apps/sapiens:0.0.1 -f Dockerfile .
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/sapiens:0.0.1 bash
# docker push gcr.io/viewo-g/piper/agent/runner/apps/sapiens:0.0.1
