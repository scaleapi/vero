FROM python:3.12-slim

ARG INDEX_REVISION=b3f37f70c33829eb09d04784a54277a31871fd63

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# pyserini reaches Lucene via jnius, which needs JAVA_HOME (a headless JRE with
# JAVA_HOME unset leaves jnius' get_jdk_home() returning None and every search
# crashing). Resolve it from the installed JDK so it survives arch/version drift.
RUN ln -s "$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")" /opt/java-home
ENV JAVA_HOME=/opt/java-home

RUN pip install --no-cache-dir \
    huggingface-hub==0.34.4 \
    openai==2.46.0 \
    pyserini==1.2.0

# The immutable upstream BM25 index is about 2.2 GB. Identical Dockerfiles let
# Harbor reuse one image for every generated query task.
RUN hf download Tevatron/browsecomp-plus-indexes \
    --repo-type dataset \
    --revision "$INDEX_REVISION" \
    --include "bm25/*" \
    --local-dir /opt/browsecomp/indexes

COPY resources/search.py /opt/browsecomp/search.py
WORKDIR /app
