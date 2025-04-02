PROJECT_NAME=decode-video-pytorch
VERSION=0.0.2

IMAGE_NAME=$(PROJECT_NAME):$(VERSION)
CONTAINER_NAME=--name=$(PROJECT_NAME)

NET=--net=host
IPC=--ipc=host
BUILD_NET=--network=host

# Specify which GPU the container can see, e.g.: all gpus if called without
# options, or a specific GPU if called with GPUS='"device=1"'.
GPUS?=all
ifeq ($(GPUS),none)
	GPUS_OPTION=--gpus=all
else
	GPUS_OPTION=--gpus=$(GPUS)
endif

.PHONY: all build stop run logs

all: build stop run logs

build:
	docker build $(BUILD_NET) -t $(IMAGE_NAME) -f Dockerfile .

stop:
	docker stop $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)

kill:
	docker kill $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)
	docker rm $(shell docker container ls -q --filter name=$(PROJECT_NAME)*)

run:
	docker run --rm -it $(GPUS_OPTION) \
		$(NET) $(IPC) \
		-v $(shell pwd):/workdir/ \
		$(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		bash

logs:
	docker logs -f $(PROJECT_NAME)
