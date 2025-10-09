import pandas as pd

# Combine all CSVs into a single DataFrame
df_combined = pd.read_csv("crawl\\ir_treaty_link.csv")

# Drop duplicate rows based on the "link" column
df_unique = df_combined.drop_duplicates(subset="link")

# Save to a new CSV file
df_unique.to_csv("ir_treaty_cleaned.csv", index=False)

# Combine all CSVs into a single DataFrame
df_combined = pd.read_csv("crawl\\wars_links.csv")

# Drop duplicate rows based on the "link" column
df_unique = df_combined.drop_duplicates(subset="link")

# Save to a new CSV file
df_unique.to_csv("wars_links_cleaned.csv", index=False)