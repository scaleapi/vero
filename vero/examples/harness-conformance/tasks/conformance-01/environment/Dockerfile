# Smallest thing that can run the verifier. The target harness needs no
# packages of its own -- it runs outside the task container and reaches this
# environment through harbor's exec interface.
FROM python:3.12-slim
WORKDIR /app
RUN mkdir -p /app /logs/verifier
