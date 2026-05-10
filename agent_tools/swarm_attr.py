"""Swarm attribution metrics for PMOVES cooperative AI execution lanes.

Exposes Agent Zero-callable tools wrapping swarm_potential.py metric collection.
Computes connection density, swarm potential, and Dirichlet-weighted lane attribution.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error
import json


_AGENT_ZERO_URL = os.environ.get("AGENT_ZERO_URL", "http://localhost:8080")
_HIRAG_URL = os.environ.get("HIRAG_URL", "http://localhost:8086")
_NEO4J_URL = os.environ.get("NEO4J_URL", "bolt://localhost:7687")
_SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
_PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")


def _http_get(url: str, timeout: int = 5) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def _fetch_agent_count() -> int:
    data = _http_get(f"{_AGENT_ZERO_URL}/api/v1/agents")
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return data.get("count", 0)
    return 0


def _fetch_indexed_chunks() -> int:
    data = _http_get(f"{_HIRAG_URL}/api/stats")
    if isinstance(data, dict):
        return data.get("total_documents", data.get("indexed_chunks", 0))
    return 0


def _fetch_supabase_count() -> int:
    if not _SUPABASE_URL:
        return 0
    data = _http_get(f"{_SUPABASE_URL}/rest/v1/rpc/count_rows")
    if isinstance(data, int):
        return data
    return 0


def get_swarm_metrics() -> Dict[str, Any]:
    """Collect current swarm metrics from PMOVES services.

    Returns:
        Dict with agent_count, indexed_chunks, supabase_rows,
        connection_density (agents * chunks^0.5 / 1000),
        swarm_potential (normalized 0.0-1.0 composite score).
    """
    agents = _fetch_agent_count()
    chunks = _fetch_indexed_chunks()
    rows = _fetch_supabase_count()

    # connection_density: proxy for knowledge-agent coupling
    connection_density = round((agents * (chunks ** 0.5)) / max(1, 1000), 4)

    # swarm_potential: composite normalized score
    agent_score = min(1.0, agents / 10.0)
    chunk_score = min(1.0, chunks / 100000.0)
    row_score = min(1.0, rows / 10000.0)
    swarm_potential = round((agent_score * 0.4 + chunk_score * 0.4 + row_score * 0.2), 4)

    return {
        "agent_count": agents,
        "indexed_chunks": chunks,
        "supabase_rows": rows,
        "connection_density": connection_density,
        "swarm_potential": swarm_potential,
    }


def score_attribution(
    agent_ids: List[str],
    weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute Dirichlet-weighted attribution scores across execution lanes.

    Args:
        agent_ids: List of agent identifiers participating in the swarm.
        weights: Optional Dirichlet alpha weights per agent (uniform if omitted).

    Returns:
        Dict with agents, normalized_scores (sum to 1.0), and dominant_agent.
    """
    if not agent_ids:
        return {"agents": [], "normalized_scores": [], "dominant_agent": None}

    n = len(agent_ids)
    if weights is None:
        weights = [1.0] * n
    elif len(weights) != n:
        weights = ([weights[i] if i < len(weights) else 1.0 for i in range(n)])

    total = sum(max(0.0001, w) for w in weights)
    normalized = [round(max(0.0001, w) / total, 6) for w in weights]
    dominant_idx = normalized.index(max(normalized))

    return {
        "agents": agent_ids,
        "normalized_scores": normalized,
        "dominant_agent": agent_ids[dominant_idx],
    }


def export_prometheus() -> str:
    """Export swarm metrics in Prometheus text format.

    Returns:
        Prometheus-format string with pmoves_swarm_* gauge metrics.
    """
    m = get_swarm_metrics()
    lines = [
        "# HELP pmoves_swarm_agent_count Number of active agents",
        "# TYPE pmoves_swarm_agent_count gauge",
        f"pmoves_swarm_agent_count {m['agent_count']}",
        "# HELP pmoves_swarm_indexed_chunks Total indexed chunks in HiRAG",
        "# TYPE pmoves_swarm_indexed_chunks gauge",
        f"pmoves_swarm_indexed_chunks {m['indexed_chunks']}",
        "# HELP pmoves_swarm_connection_density Swarm connection density",
        "# TYPE pmoves_swarm_connection_density gauge",
        f"pmoves_swarm_connection_density {m['connection_density']}",
        "# HELP pmoves_swarm_potential Normalized swarm potential score",
        "# TYPE pmoves_swarm_potential gauge",
        f"pmoves_swarm_potential {m['swarm_potential']}",
    ]
    return "\n".join(lines) + "\n"
