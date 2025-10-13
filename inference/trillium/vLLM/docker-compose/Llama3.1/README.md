# Serve Llama3.1 with vLLM using Docker Compose and Make

This guide provides a streamlined workflow for serving Llama3.1 models using `docker-compose` and a `Makefile` to simplify all commands.

## Step 1: Create and Access your TPU VM

Follow the instructions in the [original recipe](../../Llama3.1/README.md) to provision a TPU VM and SSH into it.

## Step 2: Setup and Configuration

On your TPU VM, clone this repository (if you haven't already) and navigate to this directory.

```bash
cd ~/tpu-recipes/inference/trillium/vLLM/docker-compose/Llama3.1/
```

Now, run the `setup` command. This will create a `.env` file for your configuration.

```bash
make setup
```

After running, edit the new `.env` file to add your `HF_TOKEN`. You can also switch between the 70B and 8B models by commenting and uncommenting the relevant lines.

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
    This command will enter the container, install the necessary dependencies, and run the benchmark for you.

*   **Open a shell for manual testing:**
    If you want to run `curl` commands manually or debug inside the container, you can use:
    ```bash
    make shell
    ```

*   **Stop the server:**
    ```bash
    make down
    ```

---

## Advanced Usage

### Forcing a Rebuild

If you need to pull a newer version of the Docker image, you can use the `--build` flag with the `up` command:

```bash
sudo -E docker compose up -d --build
```

### Removing the Docker Image

To manually remove the Docker image from your TPU VM, first ensure the services are down, then use the `docker rmi` command.

```bash
# Stop and remove the containers
make down

# Remove the image
sudo docker rmi $(grep "^DOCKER_URI=" .env | cut -d '=' -f2)
```
