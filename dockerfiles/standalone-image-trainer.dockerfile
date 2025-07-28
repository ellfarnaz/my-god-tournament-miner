# This Dockerfile creates the custom training environment for image tasks (SDXL, Flux).
# It uses the official Kohya SS image as a base and adds our custom scripts.

FROM diagonalge/kohya_latest:latest

# Set the working directory
WORKDIR /workspace

# Copy the entire G.O.D project structure into the image.
# This is a simple way to ensure all necessary modules like 'core' are available.
COPY . /workspace

# Copy custom scripts and configs into the workspace
COPY tournament_miner_custom/scripts/ /workspace/scripts/
COPY tournament_miner_custom/configs/ /workspace/configs/

# Make the entrypoint script executable
RUN chmod +x /workspace/scripts/run_image_trainer.sh

# Set the entrypoint for the container
ENTRYPOINT ["/workspace/scripts/run_image_trainer.sh"] 