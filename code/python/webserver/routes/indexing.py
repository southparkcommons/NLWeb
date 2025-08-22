"""JSON indexing API routes"""

import json
import logging
import tempfile
import os
from aiohttp import web

from data_loading.db_load import loadJsonToDB

logger = logging.getLogger(__name__)


def setup_indexing_routes(app: web.Application):
    """Setup indexing API routes"""
    app.router.add_post('/api/index/json', index_json_handler)


async def index_json_handler(request: web.Request) -> web.Response:
    """
    Handle direct JSON data indexing.
    
    Expects JSON body with:
    - data (array or object): JSON data to index (single object or array of objects)
    - site (str): Site identifier (e.g., "user_qdrant", "company_qdrant", etc.)
    - batch_size (int, optional): Batch size for processing (default: 100)
    - delete_existing (bool, optional): Whether to delete existing entries for this site (default: false)
    - database (str, optional): Specific database endpoint to use
    """
    try:
        body = await request.json()
        
        data = body.get('data')
        site = body.get('site')
        
        if not data or not site:
            return web.json_response({
                "success": False,
                "error": "Missing required parameters: data and site"
            }, status=400)
        
        batch_size = body.get('batch_size', 100)
        delete_existing = body.get('delete_existing', False)
        database = body.get('database')
        
        # Create temporary JSONL file from the data
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as temp_file:
                if isinstance(data, list):
                    # Array of objects
                    for item in data:
                        temp_file.write(json.dumps(item) + '\n')
                else:
                    # Single object
                    temp_file.write(json.dumps(data) + '\n')
                
                temp_path = temp_file.name
            
            logger.info(f"Starting JSON indexing: site={site}, objects={len(data) if isinstance(data, list) else 1}")
            
            # Use loadJsonToDB with the temporary file
            total_documents = await loadJsonToDB(
                file_path=temp_path,
                site=site,
                batch_size=batch_size,
                delete_existing=delete_existing,
                force_recompute=False,
                database=database
            )
            
            logger.info(f"JSON indexing completed: {total_documents} documents indexed for site {site}")
            
            return web.json_response({
                "success": True,
                "message": f"Successfully indexed {total_documents} documents",
                "details": {
                    "total_documents": total_documents,
                    "site": site,
                    "input_objects": len(data) if isinstance(data, list) else 1,
                    "database": database or "default"
                }
            })
            
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
        
    except Exception as e:
        logger.error(f"Error in JSON indexing: {e}", exc_info=True)
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)