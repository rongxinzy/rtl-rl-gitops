ARG BASE_IMAGE=rtl-l20-training:20260913-cu130
FROM ${BASE_IMAGE}
COPY requirements-runtime.txt /opt/llamafactory/requirements-runtime.txt
COPY wheels /opt/llamafactory/wheels
RUN python3 -m pip uninstall -y spin \
    && python3 -m pip install --no-index --find-links=/opt/llamafactory/wheels --no-deps -r /opt/llamafactory/requirements-runtime.txt /opt/llamafactory/wheels/llamafactory-0.9.6.dev0-py3-none-any.whl \
    && python3 -m pip check \
    && rm -rf /opt/llamafactory/wheels \
    && printf '100e9a42c6c09f8f7849b70d60f3da445fb2024b\n' > /opt/llamafactory/COMMIT
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
