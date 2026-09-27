"""Test helpers: an in-process toolbox exposing the MCP server functions with the same interface as
MCPToolbox (so graph tests don't need a subprocess), plus a graph factory using the offline LLM."""
from __future__ import annotations

from langchain_core.tools import StructuredTool

from mcp_server import server as mcp_srv


class LocalToolbox:
    def __init__(self) -> None:
        self.tools = {
            fn.__name__: StructuredTool.from_function(fn, name=fn.__name__, description=fn.__doc__ or fn.__name__)
            for fn in (mcp_srv.get_patient_record, mcp_srv.get_prior_admissions, mcp_srv.find_followup_slots)
        }

    def tool(self, name: str):
        return self.tools[name]

    async def read_resource(self, uri: str) -> str:
        assert uri == "resource://hospital/discharge_checklist"
        return mcp_srv.discharge_checklist()
