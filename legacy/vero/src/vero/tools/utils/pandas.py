from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def query_and_order_df(
    df: "pd.DataFrame",
    query: str | None = None,
    sort_values_by: str | None = None,
    ascending: bool = True,
) -> "pd.DataFrame":
    """
    Filter a dataframe using Pandas' query API and order it by a column.

    Args:
        df: The DataFrame to filter and order
        query: A query string to filter the dataframe
        sort_values_by: Column name to order by
        ascending: Whether to sort in ascending order (default True)

    Returns:
        Filtered and ordered DataFrame
    """
    df = df.copy()

    if query and query not in ["None", "null", ""]:
        df = df.query(query)

    if sort_values_by is not None:
        if sort_values_by not in df.columns:
            raise ValueError(f"Column '{sort_values_by}' not found in DataFrame for ordering")

        df = df.sort_values(by=sort_values_by, ascending=ascending)

    return df
