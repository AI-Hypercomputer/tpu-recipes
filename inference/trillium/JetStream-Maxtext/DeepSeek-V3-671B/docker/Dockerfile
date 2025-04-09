# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt -y update && apt install -y --no-install-recommends \
    apt-transport-https ca-certificates gnupg git wget \
    python3.10 python3-pip curl nano vim

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install google cloud sdk
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update -y \
    && apt-get install google-cloud-sdk -y

# Install pip
RUN python3 -m pip install --upgrade pip

# Set environment variables
ENV JAX_PLATFORMS=proxy
ENV JAX_BACKEND_TARGET=grpc://localhost:38681
ENV XCLOUD_ENVIRONMENT=GCP

# Install JetStream and MaxText

RUN git clone https://github.com/AI-Hypercomputer/JetStream.git && \
git clone https://github.com/AI-Hypercomputer/maxtext.git && \
git clone https://github.com/google/aqt.git

RUN cd /maxtext && bash setup.sh && pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install setuptools fastapi uvicorn rouge_score scikit-learn

RUN cd /JetStream && pip install -e .

RUN apt -y update && apt-get -y install python3-dev && apt-get -y install build-essential
RUN cp -r /aqt/aqt/* /usr/local/lib/python3.10/dist-packages/aqt/

ENTRYPOINT [ "/bin/bash" ]