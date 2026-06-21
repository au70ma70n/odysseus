# Unofficial Proton CalDAV bridge (ferroxide). CalDAV-only mode for Odysseus.
# Build from rupert648/ferroxide (PR #6 calendar fixes) + post-2FA token refresh patch.
FROM golang:1.23-alpine AS build
RUN apk add --no-cache git ca-certificates python3
ARG FERROXIDE_REPO=https://github.com/rupert648/ferroxide.git
ARG FERROXIDE_REF=456f067742f49f6ded386548a7e731084c50c8a1
RUN git clone --depth 1 "${FERROXIDE_REPO}" /src/ferroxide \
    && cd /src/ferroxide && git fetch --depth 1 origin "${FERROXIDE_REF}" && git checkout "${FERROXIDE_REF}"
COPY patch-auth-2fa-refresh.sh /patch-auth-2fa-refresh.sh
COPY patch-caldav-calendar.sh /patch-caldav-calendar.sh
RUN chmod +x /patch-auth-2fa-refresh.sh /patch-caldav-calendar.sh \
    && /patch-auth-2fa-refresh.sh \
    && /patch-caldav-calendar.sh
RUN cd /src/ferroxide && go build -o /ferroxide ./cmd/ferroxide

FROM alpine:3.20
RUN apk add --no-cache ca-certificates wget
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY --from=build /ferroxide /usr/local/bin/ferroxide
ENV FERROXIDE_CONFIG_HOME=/config
EXPOSE 8081
ENTRYPOINT ["/entrypoint.sh"]
