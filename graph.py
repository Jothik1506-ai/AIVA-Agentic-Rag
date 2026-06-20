"""
app/api/graph.py  ─  Knowledge Graph API
==========================================
Endpoints for the inter-document graph visualiser.
Returns D3.js-ready JSON for the frontend knowledge graph view.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional

from app.core.database import get_db, UserRecord
from app.core.security import get_current_user
from app.auth.rbac import get_allowed_doc_ids

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/visualise")
async def get_graph_visualisation(
    max_nodes: int = Query(50, ge=5, le=200),
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user),
):
    """
    Returns full graph data for D3.js force-directed visualisation.
    Nodes: Documents (blue) + Entities (colour-coded by type)
    Links: SHARES_ENTITY, CONTAINS_ENTITY, REFERENCES, RELATED_TO
    RBAC: only returns documents the user has access to.
    """
    _require_neo4j()
    allowed = get_allowed_doc_ids(current_user, db)
    # Convert doc_ids to doc_names for the graph querier
    from app.core.database import DocumentRecord
    name_map = {
        r.doc_id: r.filename
        for r in db.query(DocumentRecord).filter(
            DocumentRecord.doc_id.in_(allowed)
        ).all()
    }
    allowed_names = set(name_map.values())

    from app.graph.inter_document_graph import InterDocumentGraphQuerier
    data = InterDocumentGraphQuerier().get_graph_data_for_ui(
        allowed_doc_ids=allowed_names,
        max_nodes=max_nodes,
    )
    return data


@router.get("/related/{doc_name}")
async def get_related_documents(
    doc_name: str,
    max_results: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user),
):
    """Find documents most related to doc_name via shared entities."""
    _require_neo4j()
    allowed = get_allowed_doc_ids(current_user, db)
    from app.core.database import DocumentRecord
    allowed_names = {
        r.filename
        for r in db.query(DocumentRecord).filter(
            DocumentRecord.doc_id.in_(allowed)
        ).all()
    }
    from app.graph.inter_document_graph import InterDocumentGraphQuerier
    return InterDocumentGraphQuerier().get_related_documents(
        doc_name=doc_name,
        max_results=max_results,
        allowed_doc_ids=allowed_names,
    )


@router.get("/entities")
async def get_entity_map(
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user),
):
    """Get all entities shared across accessible documents."""
    _require_neo4j()
    allowed = get_allowed_doc_ids(current_user, db)
    from app.core.database import DocumentRecord
    allowed_names = {
        r.filename
        for r in db.query(DocumentRecord).filter(
            DocumentRecord.doc_id.in_(allowed)
        ).all()
    }
    from app.graph.inter_document_graph import InterDocumentGraphQuerier
    return InterDocumentGraphQuerier().get_entity_map(allowed_doc_ids=allowed_names)


@router.get("/path")
async def find_document_path(
    doc_a: str = Query(...),
    doc_b: str = Query(...),
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user),
):
    """Find relationship path between two documents through shared entities."""
    _require_neo4j()
    from app.graph.inter_document_graph import InterDocumentGraphQuerier
    return InterDocumentGraphQuerier().find_path_between_documents(doc_a, doc_b)


@router.get("/neighbourhood/{doc_name}")
async def get_neighbourhood(
    doc_name: str,
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user),
):
    """Get full neighbourhood of a document: shared entities and related docs."""
    _require_neo4j()
    allowed = get_allowed_doc_ids(current_user, db)
    from app.core.database import DocumentRecord
    allowed_names = {
        r.filename
        for r in db.query(DocumentRecord).filter(
            DocumentRecord.doc_id.in_(allowed)
        ).all()
    }
    from app.graph.inter_document_graph import InterDocumentGraphQuerier
    return InterDocumentGraphQuerier().get_document_neighbourhood(
        doc_name=doc_name, allowed_doc_ids=allowed_names
    )


@router.post("/rebuild")
async def rebuild_graph(
    db: Session = Depends(get_db),
    current_user: UserRecord = Depends(get_current_user),
):
    """Manually trigger a full inter-document graph rebuild (VP only)."""
    if current_user.role != "vice_president":
        raise HTTPException(403, "Only Vice Presidents can trigger a full graph rebuild")
    _require_neo4j()
    from app.graph.inter_document_graph import InterDocumentGraphBuilder
    from app_state import doc_manager
    if not doc_manager:
        raise HTTPException(503, "Document manager not initialised")
    result = InterDocumentGraphBuilder().build_from_documents(doc_manager, db)
    return {"message": "Graph rebuilt", **result}


def _require_neo4j():
    from app_state import neo4j_db
    if neo4j_db is None:
        raise HTTPException(
            503,
            "Neo4j is not running. Graph features require Neo4j. "
            "Start with: docker run -p 7474:7474 -p 7687:7687 "
            "-e NEO4J_AUTH=neo4j/password neo4j:latest"
        )
