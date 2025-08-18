#!/usr/bin/env python3
"""
Multi-embedding generator for creating multiple focused embeddings per document.

Instead of one diluted embedding, create multiple specialized embeddings that
all point to the same document, allowing for precise search matching.
"""

import json
from typing import Dict, Any, List
import hashlib

def generate_document_embeddings(item: Dict[str, Any], base_url: str, site: str) -> List[Dict[str, Any]]:
    """
    Generate multiple specialized embeddings for a single document.
    
    For companies, creates specialized embeddings for:
    - Identity embedding: Company names, stage, exact identifiers
    - Investor embedding: Funding, investors, financing relationships  
    - Business embedding: Industries, descriptions, business context
    - Metadata embedding: Locations, employee size, other details
    
    Args:
        item: The document data
        base_url: Base URL for the document
        site: Site identifier
        
    Returns:
        List of embedding documents, each with focused text and shared metadata
    """
    
    item_type = detect_item_type(item)
    base_doc_id = generate_base_doc_id(base_url)
    
    if item_type == "company":
        embeddings = generate_company_embeddings(item, base_url, base_doc_id, site)
    else:
        embeddings = generate_generic_embeddings(item, base_url, base_doc_id, site)
    
    # Validate embeddings before returning
    validated_embeddings = []
    for emb in embeddings:
        if validate_embedding_document(emb):
            validated_embeddings.append(emb)
    
    return validated_embeddings if validated_embeddings else [create_fallback_embedding(item, base_url, base_doc_id, site)]

def detect_item_type(item: Dict[str, Any]) -> str:
    """Detect whether item is a company based on fields."""
    company_fields = {"investors", "founded_year", "stage", "industries", "short_description", "employee_size"}
    
    company_score = sum(1 for field in company_fields if field in item)
    
    # For this experiment, only generate company embeddings for items that look like companies
    return "company" if company_score > 0 else "generic"

def generate_base_doc_id(url: str) -> str:
    """Generate a consistent base document ID."""
    return hashlib.md5(url.encode()).hexdigest()

def create_embedding_document(base_doc_id: str, embedding_type: str, text: str, 
                            original_item: Dict[str, Any], url: str, site: str) -> Dict[str, Any]:
    """Create a single embedding document with shared metadata."""
    
    return {
        "id": f"{base_doc_id}_{embedding_type}",
        "base_doc_id": base_doc_id,  # Link back to original document
        "embedding_type": embedding_type,
        "embedding_text": text,
        "schema_json": json.dumps(original_item),  # Full original data
        "url": url,
        "name": original_item.get("name", ""),
        "site": site
    }

def generate_company_embeddings(item: Dict[str, Any], base_url: str, base_doc_id: str, site: str) -> List[Dict[str, Any]]:
    """Generate specialized embeddings for company documents."""
    
    embeddings = []
    name = (item.get("name") or "").strip()
    
    # 1. IDENTITY EMBEDDING - for exact name/title matching
    identity_parts = []
    if name:
        identity_parts.extend([name, name, name])  # Triple repetition
    
    stage = (item.get("stage") or "").strip()
    if stage:
        identity_parts.extend([stage, f"{stage} company"])
    
    if identity_parts:
        identity_text = " ".join(identity_parts)
        embeddings.append(create_embedding_document(
            base_doc_id, "identity", identity_text, item, base_url, site
        ))
    
    # 2. INVESTOR EMBEDDING - for investor/funding searches
    investors = item.get("investors", [])
    if investors:
        investor_parts = []
        investor_names = " ".join(str(inv) for inv in investors)
        
        investor_parts.extend([
            f"{name} funded by {investor_names}",
            f"{name} backed by {investor_names}",
            f"{name} invested in by {investor_names}",
            f"companies funded by {investor_names}",
            f"companies backed by {investor_names}",
            investor_names,  # Raw investor names
            investor_names,  # Repetition for strength
        ])
        
        investor_text = " ".join(investor_parts)
        embeddings.append(create_embedding_document(
            base_doc_id, "investor", investor_text, item, base_url, site
        ))
    
    # 3. INDUSTRY/BUSINESS EMBEDDING - for domain/industry searches
    industry_parts = []
    
    industries = item.get("industries", [])
    if industries:
        industry_text = " ".join(str(ind) for ind in industries)
        industry_parts.extend([
            f"{name} {industry_text}",
            f"{industry_text} company",
            industry_text
        ])
    
    description = (item.get("short_description") or "").strip()
    if description:
        industry_parts.extend([
            f"{name} {description}",
            description
        ])
    
    if industry_parts:
        business_text = " ".join(industry_parts)
        embeddings.append(create_embedding_document(
            base_doc_id, "business", business_text, item, base_url, site
        ))
    
    # 4. FOUNDER EMBEDDING - for people-company connections
    founders = item.get("founders", [])
    if founders:
        founder_parts = []
        founder_names = " ".join(str(founder) for founder in founders)
        
        founder_parts.extend([
            f"{name} founded by {founder_names}",
            f"{founder_names} founded {name}",
            f"{founder_names} founder of {name}",
            f"companies founded by {founder_names}",
            founder_names
        ])
        
        founder_text = " ".join(founder_parts)
        embeddings.append(create_embedding_document(
            base_doc_id, "founder", founder_text, item, base_url, site
        ))
    
    # 5. METADATA EMBEDDING - for location, size, year searches
    metadata_parts = []
    
    location = (item.get("location") or "").strip()
    if location:
        metadata_parts.extend([
            f"{name} located in {location}",
            f"companies in {location}",
            location
        ])
    
    founded_year = item.get("founded_year")
    if founded_year:
        metadata_parts.extend([
            f"{name} founded in {founded_year}",
            f"companies founded in {founded_year}"
        ])
    
    employee_size = (item.get("employee_size") or "").strip()
    if employee_size:
        metadata_parts.extend([
            f"{name} {employee_size} employees",
            f"{employee_size} company"
        ])
    
    if metadata_parts:
        metadata_text = " ".join(metadata_parts)
        embeddings.append(create_embedding_document(
            base_doc_id, "metadata", metadata_text, item, base_url, site
        ))
    
    return embeddings

def generate_generic_embeddings(item: Dict[str, Any], base_url: str, base_doc_id: str, site: str) -> List[Dict[str, Any]]:
    """Fallback for unknown item types."""
    
    # For unknown types, create a single embedding with all content
    all_text_parts = []
    
    # Extract all string values
    for key, value in item.items():
        if isinstance(value, str) and value.strip():
            all_text_parts.append(value.strip())
        elif isinstance(value, list):
            list_text = " ".join(str(v) for v in value if v)
            if list_text:
                all_text_parts.append(list_text)
    
    if all_text_parts:
        full_text = " ".join(all_text_parts)
        return [create_embedding_document(
            base_doc_id, "full", full_text, item, base_url, site
        )]
    
    return []

def validate_embedding_document(emb_doc: Dict[str, Any]) -> bool:
    """Validate that an embedding document has required fields and valid content."""
    required_fields = {"id", "base_doc_id", "embedding_type", "embedding_text"}
    
    # Check required fields exist
    if not all(field in emb_doc for field in required_fields):
        return False
    
    # Check non-empty values
    if not all(emb_doc[field] for field in required_fields):
        return False
    
    # Check embedding text is meaningful (more than just whitespace)
    if not emb_doc["embedding_text"].strip():
        return False
    
    return True

def create_fallback_embedding(item: Dict[str, Any], base_url: str, base_doc_id: str, site: str) -> Dict[str, Any]:
    """Create a basic single embedding as fallback when multi-embedding fails."""
    item_json = json.dumps(item)
    
    return {
        "id": f"{base_doc_id}_fallback",
        "base_doc_id": base_doc_id,
        "embedding_type": "fallback",
        "embedding_text": item_json,
        "schema_json": item_json,
        "url": base_url,
        "name": item.get("name", ""),
        "site": site
    }