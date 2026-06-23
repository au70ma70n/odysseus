"""Minimal MCPServer shim matching the decorator API used by OpenSCAD-MCP-Server."""


class MCPServer:
    def __init__(self):
        self.tools = {}

    def tool(self, fn):
        self.tools[fn.__name__] = fn
        return fn
