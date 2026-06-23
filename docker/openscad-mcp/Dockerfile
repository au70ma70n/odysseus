# OpenSCAD text-to-STL sidecar for Odysseus (jhacksman/OpenSCAD-MCP-Server core libs).
FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends openscad git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG OPENSCAD_MCP_REPO=https://github.com/jhacksman/OpenSCAD-MCP-Server.git
ARG OPENSCAD_MCP_REF=6948c2dc5a385adc987bc3ff75ba435d26f48ba3

WORKDIR /app

RUN git clone "${OPENSCAD_MCP_REPO}" /tmp/openscad-mcp \
    && cd /tmp/openscad-mcp \
    && git fetch --depth 1 origin "${OPENSCAD_MCP_REF}" \
    && git checkout "${OPENSCAD_MCP_REF}" \
    && cp -r src /app/src \
    && cp -r scad /app/scad \
    && cp src/models/scad_templates/basic_shapes.scad /app/scad/basic_shapes.scad \
    && rm -rf /tmp/openscad-mcp

COPY requirements-min.txt mcp_legacy.py minimal_server.py /app/
RUN pip install --no-cache-dir -r requirements-min.txt

ENV PYTHONPATH=/app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "minimal_server:app", "--host", "0.0.0.0", "--port", "8000"]
