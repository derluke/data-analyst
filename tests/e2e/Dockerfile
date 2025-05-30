FROM python:3.11-alpine

LABEL org.opencontainers.image.title="Application Template E2E Testing with Selenium and Chromium"
LABEL org.opencontainers.image.authors="Applications <apps@datarobot.com>"
LABEL org.opencontainers.image.source="https://github.com/datarobot/data-analyst"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.vendor="DataRobot"

# Update apk repo and install dependencies
RUN echo "http://dl-4.alpinelinux.org/alpine/v3.14/main" >> /etc/apk/repositories && \
    echo "http://dl-4.alpinelinux.org/alpine/v3.14/community" >> /etc/apk/repositories && \
    apk update --no-cache && \
    apk add --no-cache chromium chromium-chromedriver make && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir 'selenium<=4.33.0' 'pytest<=9.0'

# create a non-root user and switch to it
RUN addgroup -S testgroup && adduser -S testuser -G testgroup
USER testuser
