
BACKUP_DIR="/media/external_drive/docker_backups"

docker save "$IMAGE" \
  | zstd -T0 -3 -o "${BACKUP_DIR}/${ARCHIVE}.partial"

mv "${BACKUP_DIR}/${ARCHIVE}.partial" "${BACKUP_DIR}/${ARCHIVE}"
