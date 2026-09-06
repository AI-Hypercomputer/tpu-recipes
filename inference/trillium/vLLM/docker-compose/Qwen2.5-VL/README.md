# Serve Qwen2.5-VL with vLLM using Docker Compose and Make

This guide provides a streamlined workflow for serving the multi-modal Qwen2.5-VL-7B model using `docker-compose` and a `Makefile`.

## Step 1: Create and Access your TPU VM

Follow the instructions in the [original recipe](../../Qwen2.5-VL/README.md) to provision a TPU VM and SSH into it.

## Step 2: Setup and Configuration

On your TPU VM, clone this repository (if you haven't already) and navigate to this directory.

```bash
cd ~/tpu-recipes/inference/trillium/vLLM/docker-compose/Qwen2.5-VL/
```

Run the `setup` command to create your `.env` file.

```bash
make setup
```

After running, edit the new `.env` file to add your Hugging Face token.

```bash
nano .env
```

## Step 3: Run the Server

The `Makefile` provides simple commands to manage the server lifecycle.

*   **Start the server:**
    ```bash
    make up
    ```
*   **Follow the logs:**
    ```bash
    make logs
    ```
    The server is ready when you see the message: `Application startup complete.`

*   **Run the benchmark:**
    Once the server is ready, you can run the multi-modal benchmark with a single command:
    ```bash
    make benchmark
    ```

*   **Open a shell for manual testing:**
    If you want to debug inside the container, you can use:
    ```bash
    make shell
    ```

*   **Stop the server:**
    ```bash
    make down
    ```
