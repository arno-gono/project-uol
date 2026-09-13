import sqlite3
from app.data.sqlite_connector import connecting_to_sqlite
from app.config import KAGGLE_DATASET_NAME


# Views built on top of the tables of each dataset, the way a reporting layer reads the data in production.
# A view is calibrated on the clean tables. Its _new_data version reads the tables with the new batch appended, so it
# is only created in the agent's database, where the _new_data tables exist.
DATASET_VIEWS = {
    "airbnb/seattle": {
        "v_monthly_reviews": """
                CREATE VIEW v_monthly_reviews AS
                SELECT
                    SUBSTR(R.date, 1, 7) AS month,
                    COUNT(*) AS nb_reviews,
                    COUNT(DISTINCT R.listing_id) AS nb_listings_reviewed,
                    COUNT(DISTINCT L.host_id) AS nb_hosts,
                    ROUND(AVG(CAST(REPLACE(REPLACE(L.price, '$', ''), ',', '') AS REAL)), 2) AS avg_listing_price
                FROM
                    reviews R
                INNER JOIN
                    listings L
                ON
                    R.listing_id = L.id
                GROUP BY
                    SUBSTR(R.date, 1, 7)
            """,
        # Same view with the new batch appended. Only the columns the view needs are selected: an error injected can
        # add a column to a _new_data table, which would break the UNION ALL
        "v_monthly_reviews_new_data": """
                CREATE VIEW v_monthly_reviews_new_data AS
                SELECT
                    SUBSTR(R.date, 1, 7) AS month,
                    COUNT(*) AS nb_reviews,
                    COUNT(DISTINCT R.listing_id) AS nb_listings_reviewed,
                    COUNT(DISTINCT L.host_id) AS nb_hosts,
                    ROUND(AVG(CAST(REPLACE(REPLACE(L.price, '$', ''), ',', '') AS REAL)), 2) AS avg_listing_price
                FROM
                    (
                        SELECT listing_id, date FROM reviews
                        UNION ALL
                        SELECT listing_id, date FROM reviews_new_data
                    ) R
                INNER JOIN
                    (
                        SELECT id, host_id, price FROM listings
                        UNION ALL
                        SELECT id, host_id, price FROM listings_new_data
                    ) L
                ON
                    R.listing_id = L.id
                GROUP BY
                    SUBSTR(R.date, 1, 7)
            """,
    },
}


def _create_sqlite_view(conn: sqlite3.Connection, view_name: str, view_query: str) -> None:
    # Dropping the current view if it already exists
    conn.execute(f"DROP VIEW IF EXISTS {view_name}")

    # Now creating the view
    conn.execute(view_query)

    return None


def create_sqlite_views(kaggle_dataset: str = KAGGLE_DATASET_NAME, database_type: str = "clean") -> None:
    # A dataset without views declared above is left as it is
    if kaggle_dataset not in DATASET_VIEWS:
        return None

    # Views are not migrated with the tables: they are created every time tables are written to SQLite
    conn = connecting_to_sqlite(kaggle_dataset, database_type=database_type)

    for view_name, view_query in DATASET_VIEWS[kaggle_dataset].items():

        # The _new_data version of a view reads the _new_data tables, which only exist in the agent's database
        if "_new_data" in view_name and database_type != "agent":
            continue

        _create_sqlite_view(conn, view_name, view_query)

    conn.close()
    return None


if __name__ == "__main__":
    kaggle_dataset = KAGGLE_DATASET_NAME

    # create_sqlite_views(kaggle_dataset, database_type="clean")
