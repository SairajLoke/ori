
BACKUP_DIR="/run/media/kaustubh/CRUZER_BLA/ori"

# docker save "$IMAGE" \
#   | zstd -T0 -3 -o "${BACKUP_DIR}/${ARCHIVE}.partial"

# mv "${BACKUP_DIR}/${ARCHIVE}.partial" "${BACKUP_DIR}/${ARCHIVE}"

NEWARCHIVE="${BACKUP_DIR}/${ARCHIVE}"

zstd -t "$NEWARCHIVE" || (echo "Error: Archive verification failed!" && exit 1)

sha256sum "$NEWARCHIVE" | tee "${NEWARCHIVE}.sha256"