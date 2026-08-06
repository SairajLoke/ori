zstd -dc "$$LOADED_ARCHIVE" | docker load
docker image inspect "$IMAGE"