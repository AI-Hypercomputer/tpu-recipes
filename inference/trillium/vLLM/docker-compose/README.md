# vLLM Recipes with Docker Compose

This directory provides an alternative workflow for running the vLLM recipes using `docker-compose`.

Using `docker-compose` simplifies the process by abstracting away the long, complex `docker run` commands into declarative `docker-compose.yml` files. Configuration is managed via `.env` files, making it easier to switch between models and settings.

## Available Recipes

*   [Llama3.1](./Llama3.1/README.md)
*   [Qwen3](./Qwen3/README.md)
*   [Qwen2.5-32B](./Qwen2.5-32B/README.md)
*   [Qwen2.5-VL](./Qwen2.5-VL/README.md)
