FROM ghcr.io/speaches-ai/speaches:latest-cuda

COPY speaches/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
