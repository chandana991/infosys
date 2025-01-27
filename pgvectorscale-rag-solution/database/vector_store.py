import logging
import time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import openai  # Correctly import openai library
from config.settings import get_settings
from timescale_vector import client


class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with settings, OpenAI client, and Timescale Vector client."""
        self.settings = get_settings()  # Load settings
        self.openai_client = openai  # Use openai module directly
        self.embedding_model = self.settings["openai"]["embedding_model"]
        self.vector_settings = self.settings["vector_store"]
        self.vec_client = client.Sync(
            self.settings["database"]["service_url"],
            self.vector_settings["table_name"],
            self.vector_settings["embedding_dimensions"],
            time_partition_interval=self.vector_settings["time_partition_interval"],
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding_response = self.openai_client.Embedding.create(
            input=[text],
            model=self.embedding_model,
        )
        embedding = embedding_response["data"][0]["embedding"]
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self) -> None:
        """Create the necessary tables in the database."""
        self.vec_client.create_tables()

    def create_index(self) -> None:
        """Create the StreamingDiskANN index to speed up similarity search."""
        self.vec_client.create_embedding_index(client.DiskAnnIndex())

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database."""
        self.vec_client.drop_embedding_index()

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logging.info(
            f"Inserted {len(df)} records into {self.vector_settings['table_name']}"
        )

    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.

        Args:
            query_text: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary or list of dictionaries for equality-based metadata filtering.
            predicates: A Predicates object for complex metadata filtering.
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.
        """
        query_embedding = self.get_embedding(query_text)

        start_time = time.time()

        search_args = {"limit": limit}
        if metadata_filter:
            search_args["filter"] = metadata_filter
        if predicates:
            search_args["predicates"] = predicates
        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = self.vec_client.search(query_embedding, **search_args)
        elapsed_time = time.time() - start_time
        logging.info(f"Vector search completed in {elapsed_time:.3f} seconds")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

    def _create_dataframe_from_results(
        self, results: List[Tuple[Any, ...]]
    ) -> pd.DataFrame:
        """Create a pandas DataFrame from the search results."""
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )
        df = pd.concat(
            [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
        )
        df["id"] = df["id"].astype(str)
        return df
