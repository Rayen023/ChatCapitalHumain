import os

import pandas as pd
from sqlalchemy import MetaData, create_engine


def export_db_to_csv(db_url, output_dir="fixed_outliers_exported_csvs"):
    """
    Exports all tables from the database specified by db_url into separate CSV files.

    Parameters:
      db_url (str): Database URL in the format
                    "postgresql://postgres.***:***c@aws-0-ca-central-1.pooler.supabase.com:6543/postgres"
      output_dir (str): Directory where the CSV files will be saved.

    Returns:
      list: A list of file paths to the generated CSV files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a SQLAlchemy engine
    engine = create_engine(db_url)

    # Reflect the database schema to get all table names
    metadata = MetaData()
    metadata.reflect(bind=engine)

    csv_files = []
    for table_name in metadata.tables.keys():
        try:
            # Load table into a pandas DataFrame
            df = pd.read_sql_table(table_name, engine)

            # Define CSV file path
            csv_path = os.path.join(output_dir, f"{table_name}.csv")

            # Write DataFrame to CSV without the index column
            df.to_csv(csv_path, index=False)
            csv_files.append(csv_path)

            print(f"Exported table '{table_name}' to {csv_path}")
        except Exception as e:
            print(f"Error exporting table '{table_name}': {e}")

    # Close the engine connection
    engine.dispose()

    return csv_files


# Example usage:
if __name__ == "__main__":
    import streamlit as st

    db_url = st.secrets["db_url"]
    csv_files_created = export_db_to_csv(db_url)
    print("CSV files created:", csv_files_created)
