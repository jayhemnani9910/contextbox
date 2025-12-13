
# Advanced Search API Example (FastAPI)
from fastapi import FastAPI, Query, HTTPException
from datetime import datetime
from typing import Optional, List
from search import SearchEngine, SearchCriteria

app = FastAPI(title="ContextBox Advanced Search API")

@app.post("/search")
async def advanced_search(
    query: str = Query(..., description="Search query"),
    content_types: Optional[List[str]] = Query(None, description="Content types to filter"),
    date_from: Optional[datetime] = Query(None, description="Start date"),
    date_to: Optional[datetime] = Query(None, description="End date"),
    fuzzy_threshold: Optional[int] = Query(None, description="Fuzzy matching threshold"),
    use_regex: bool = Query(False, description="Use regex matching"),
    highlight: bool = Query(True, description="Include highlighting"),
    limit: int = Query(100, ge=1, le=1000, description="Result limit"),
    offset: int = Query(0, ge=0, description="Results offset")
):
    """Advanced search endpoint with full feature support."""
    
    try:
        search_engine = SearchEngine()
        
        criteria = SearchCriteria(
            query=query,
            content_types=content_types,
            date_from=date_from,
            date_to=date_to,
            fuzzy_threshold=fuzzy_threshold,
            use_regex=use_regex,
            highlight=highlight,
            limit=limit,
            offset=offset
        )
        
        results, metadata = search_engine.search(criteria)
        
        return {
            "results": [
                {
                    "id": r.id,
                    "title": r.title,
                    "content": r.content,
                    "url": r.url,
                    "kind": r.kind,
                    "relevance_score": r.relevance_score,
                    "highlights": r.highlights,
                    "context_snippets": r.context_snippets,
                    "created_at": r.created_at.isoformat()
                }
                for r in results
            ],
            "metadata": metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/export")
async def export_search_results(
    results: List[dict],
    format: str = Query("csv", regex="^(csv|json|txt)$"),
    include_highlights: bool = Query(True)
):
    """Export search results to file."""
    
    try:
        search_engine = SearchEngine()
        
        # Convert dict results back to SearchResult objects
        search_results = []
        for r in results:
            from search import SearchResult
            search_results.append(SearchResult(**r))
        
        output_file = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        success = search_engine.export_results(search_results, output_file, format, include_highlights)
        
        if success:
            return {"success": True, "file": output_file}
        else:
            raise HTTPException(status_code=500, detail="Export failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/searches/saved")
async def get_saved_searches():
    """Get all saved searches."""
    search_engine = SearchEngine()
    return search_engine.get_saved_searches()

@app.post("/searches/save")
async def save_search(
    name: str,
    description: str = "",
    criteria: dict = {}
):
    """Save a search configuration."""
    search_engine = SearchEngine()
    
    from search import SearchCriteria
    search_criteria = SearchCriteria(**criteria)
    
    search_id = search_engine.save_search(name, search_criteria, description)
    return {"success": True, "search_id": search_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
