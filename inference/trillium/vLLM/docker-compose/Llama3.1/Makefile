.PHONY: help up down logs shell clean setup benchmark

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  setup      Create .env file from the example"
	@echo "  up         Start the vLLM server in the background"
	@echo "  down       Stop and remove the vLLM server"
	@echo "  logs       Follow the server logs"
	@echo "  shell      Get a bash shell inside the running container"
	@echo "  benchmark  Run the benchmark test against the server"
	@echo "  clean      Remove the .env file"

# Creates the .env file if it doesn't exist
setup:
	@if [ ! -f .env ]; then \
		echo "Creating .env file..."; \
		cp .env.example .env; \
		sed -i'' -e "s/your-username/$(USER)/g" .env; \
		echo "Created .env and set USER to '$(USER)'. Please edit it to add your Hugging Face token."; \
	else \
		echo ".env file already exists."; \
	fi

# The -E flag preserves environment variables for the container
up: setup
	sudo -E docker compose up -d

down:
	sudo docker compose down

logs:
	sudo docker compose logs -f

# Get a bash shell inside the running container
shell:
	sudo docker exec -it $(shell sudo docker compose ps -q vllm) bash

benchmark:
	@echo "Running benchmark inside the container..."
	@sudo docker exec -it \
		-e MODEL_NAME=$(shell grep '^MODEL_NAME=' .env | cut -d= -f2) \
		-e SEED=$(shell grep '^SEED=' .env | cut -d= -f2) \
		$(shell sudo docker compose ps -q vllm) \
		bash -c ' \
			cd /workspace/vllm && \
			pip install -q datasets && \
			vllm bench serve \
				--model "$${MODEL_NAME}" \
				--dataset-name random \
				--num-prompts 1000 \
				--random-input-len=1800 \
				--random-output-len=128 \
				--seed=$${SEED} \
		'

clean:
	rm -f .env
