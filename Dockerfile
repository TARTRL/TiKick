FROM reg.real-ai.cn/launchpad/launchpad-tensorflow:latest
MAINTAINER Sen Na <sen.na@realai.ai>

WORKDIR /tmarl

COPY . .

ARG pip_registry='https://mirrors.aliyun.com/pypi/simple/'
RUN pip install -e .

ARG pip_dependencies='\
    torch \
    wandb \
    setproctitle \
    gym \
    seaborn \
    tensorboardX \
    icecream'

RUN pip install -i $pip_registry $pip_dependencies
