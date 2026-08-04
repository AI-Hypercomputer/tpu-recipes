# Serve Qwen2.5-32B with vLLM using Docker Compose and Make

This guide provides a streamlined workflow for serving the Qwen2.5-32B model using `docker-compose` and a `Makefile` to simplify all commands.

## Step 1: Create and Access your TPU VM

Follow the instructions in the [original recipe](../../Qwen2.5-32B/README.md) to provision a TPU VM and SSH into it.

## Step 2: Setup and Configuration

On your TPU VM, clone this repository (if you haven't already) and navigate to this directory.

```bash
cd ~/tpu-recipes/inference/trillium/vLLM/docker-compose/Qwen2.5-32B/
```

Now, run the `setup` command. This will create a `.env` file for your configuration.

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
    Once the server is ready, you can run the benchmark with a single command:
    ```bash
    make benchmark
    ```

*   **Open a shell for manual testing:**
    If you want to run `curl` commands manually or debug inside the container, you can use:
    ```bash
    make shell
    ```

*   **Stop the server:**
    ```bash
    make down
    ```
