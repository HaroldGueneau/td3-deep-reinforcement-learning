.PHONY: grid play train

grid:
	docker build -t td3-grid -f deploy/DockerfileGrid .
	docker run --name td3-grid-example -d td3-grid

play:
	docker build -t td3-play -f deploy/DockerfilePlaying .
	docker run --name td3-play-example -d td3-play

train:
	docker build -t td3-train -f deploy/DockerfileTraining .
	docker run --name td3-train-example -d td3-train
