#!/usr/bin/env python3
"""
Simple script to create a Qdrant collection for NLWeb.

This script provides a simple interface to create Qdrant collections
using the existing QdrantVectorClient class.
"""

import asyncio
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from retrieval.qdrant import QdrantVectorClient
from utils.logging_config_helper import get_configured_logger

logger = get_configured_logger("create_qdrant_collection")


async def create_qdrant_collection(
    collection_name: str = None,
    vector_size: int = 1536,
    endpoint_name: str = None,
    recreate: bool = False,
) -> bool:
    """
    Create or recreate a Qdrant collection.

    Args:
        collection_name: Name of the collection to create (defaults to configured name)
        vector_size: Size of the embedding vectors (default: 1536)
        endpoint_name: Name of the endpoint to use (defaults to preferred endpoint)
        recreate: If True, drop and recreate the collection (default: False)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize the Qdrant client
        client = QdrantVectorClient(endpoint_name=endpoint_name)

        print(f"Using endpoint: {client.endpoint_name}")
        print(f"Collection name: {collection_name or client.default_collection_name}")
        print(f"Vector size: {vector_size}")

        if recreate:
            print(
                f"Recreating collection '{collection_name or client.default_collection_name}'..."
            )
            success = await client.recreate_collection(collection_name, vector_size)
            if success:
                print(
                    f"Successfully recreated collection '{collection_name or client.default_collection_name}'"
                )
            else:
                print("Collection recreation failed")
        else:
            # Check if collection already exists
            exists = await client.collection_exists(collection_name)
            if exists:
                print(
                    f"Collection '{collection_name or client.default_collection_name}' already exists!"
                )
                return True

            # Create the collection
            print(
                f"Creating collection '{collection_name or client.default_collection_name}'..."
            )
            success = await client.create_collection(collection_name, vector_size)

            if success:
                print(
                    f"Successfully created collection '{collection_name or client.default_collection_name}'"
                )
            else:
                print("Collection creation failed or collection already exists")

        return success

    except Exception as e:
        print(f"Error creating collection: {str(e)}")
        logger.exception(f"Error creating collection: {str(e)}")
        return False


async def list_qdrant_collections(endpoint_name: str = None) -> bool:
    """
    List all collections in the Qdrant instance.

    Args:
        endpoint_name: Name of the endpoint to use (defaults to preferred endpoint)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize the Qdrant client
        client = QdrantVectorClient(endpoint_name=endpoint_name)

        print(f"Using endpoint: {client.endpoint_name}")

        # Get the client and list collections
        qdrant_client = await client._get_qdrant_client()
        collections = await qdrant_client.get_collections()

        print(f"Found {len(collections.collections)} collections:")
        for collection in collections.collections:
            print(f"  - {collection.name}")

        return True

    except Exception as e:
        print(f"Error listing collections: {str(e)}")
        logger.exception(f"Error listing collections: {str(e)}")
        return False


def main():
    """Main function for command line usage."""
    if len(sys.argv) < 2:
        print("""
Usage: python create_qdrant_collection.py [command] [options]

Commands:
  create [collection_name] [vector_size] [endpoint_name]  - Create a new collection
  recreate [collection_name] [vector_size] [endpoint_name] - Recreate (drop and create) a collection
  list [endpoint_name]                                    - List all collections

Examples:
  python create_qdrant_collection.py create
  python create_qdrant_collection.py create my_collection 1536
  python create_qdrant_collection.py create my_collection 1536 qdrant_local
  python create_qdrant_collection.py recreate my_collection
  python create_qdrant_collection.py list
  python create_qdrant_collection.py list qdrant_url
""")
        return

    command = sys.argv[1].lower()

    if command == "create":
        collection_name = sys.argv[2] if len(sys.argv) > 2 else None
        vector_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1536
        endpoint_name = sys.argv[4] if len(sys.argv) > 4 else None

        asyncio.run(
            create_qdrant_collection(
                collection_name, vector_size, endpoint_name, recreate=False
            )
        )

    elif command == "recreate":
        collection_name = sys.argv[2] if len(sys.argv) > 2 else None
        vector_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1536
        endpoint_name = sys.argv[4] if len(sys.argv) > 4 else None

        asyncio.run(
            create_qdrant_collection(
                collection_name, vector_size, endpoint_name, recreate=True
            )
        )

    elif command == "list":
        endpoint_name = sys.argv[2] if len(sys.argv) > 2 else None

        asyncio.run(list_qdrant_collections(endpoint_name))

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
