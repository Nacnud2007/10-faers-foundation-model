"""Process quarterly FAERS ASCII downloads into combined drug and reaction data."""
import os
import re
import pandas as pd

ROOT_DIR = "./data/raw"


def process_quarterly_data(root_dir):
    all_quarter_dfs = []

    folder_pattern = re.compile(r"faers_ascii_\d{4}q\d", re.IGNORECASE)

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)

        # Check if it's a directory and matches expected quarter pattern
        if os.path.isdir(folder_path) and folder_pattern.match(folder_name):
            print(f"Processing folder: {folder_name}...")

            # Path to the internal 'ascii' folder where the .txt files are stored
            ascii_path = os.path.join(folder_path, "ascii")
            if not os.path.exists(ascii_path):
                # Fallback in case some folders don't have the nested 'ascii' subfolder
                ascii_path = folder_path

            # Locate the DRUG and REAC files
            drug_file = None
            reac_file = None

            for file in os.listdir(ascii_path):
                file_upper = file.upper()
                if file_upper.startswith("DRUG") and file_upper.endswith(".TXT"):
                    drug_file = os.path.join(ascii_path, file)
                elif file_upper.startswith("REAC") and file_upper.endswith(".TXT"):
                    reac_file = os.path.join(ascii_path, file)

            # Process if both files are found in the folder
            if drug_file and reac_file:
                try:
                    # 'primaryid' is the unique patient ID. 'drugname' for drugs, 'pt' for reaction terms.
                    df_drug = pd.read_csv(
                        drug_file,
                        sep="$",
                        usecols=["primaryid", "drugname"],
                        low_memory=False,
                    )
                    df_reac = pd.read_csv(
                        reac_file, sep="$", usecols=["primaryid", "pt"], low_memory=False
                    )

                    # Delete incomplete entries
                    df_drug.dropna(subset=["primaryid", "drugname"], inplace=True)
                    df_reac.dropna(subset=["primaryid", "pt"], inplace=True)

                    # Clean string spaces
                    df_drug["drugname"] = df_drug["drugname"].astype(str).str.strip()
                    df_reac["pt"] = df_reac["pt"].astype(str).str.strip()

                    df_drug_agg = (
                        df_drug.groupby("primaryid")["drugname"]
                        .apply(lambda x: "; ".join(sorted(set(x))))
                        .reset_index()
                    )
                    df_reac_agg = (
                        df_reac.groupby("primaryid")["pt"]
                        .apply(lambda x: "; ".join(sorted(set(x))))
                        .reset_index()
                    )

                    # Merge the aggregated drugs and reactions on the unique primaryid
                    df_merged = pd.merge(
                        df_drug_agg, df_reac_agg, on="primaryid", how="inner"
                    )

                    quarter_label = folder_name.split("_")[-1].upper()
                    df_merged["quarter"] = quarter_label

                    all_quarter_dfs.append(df_merged)
                    print(f"Successfully processed {len(df_merged)} valid patients.")

                except Exception as e:
                    print(f"Error processing {folder_name}: {e}")
            else:
                print(
                    f"Skipping {folder_name}: Could not find both DRUG and REAC .txt files."
                )

    # Combine all quarters into one df
    if all_quarter_dfs:
        print("Combining all quarters into a single master file...")
        final_df = pd.concat(all_quarter_dfs, ignore_index=True)

        # Rename columns
        final_df.rename(
            columns={"drugname": "drug_combination", "pt": "adrs"}, inplace=True
        )

        # Save
        output_path = "faers_combined_cleaned.csv"
        final_df.to_csv(output_path, index=False)
        print(f"Finished. Final dataset saved to {output_path}")
        print(f"Total master rows: {len(final_df)}")
    else:
        print("No data was processed.")


if __name__ == "__main__":
    process_quarterly_data(ROOT_DIR)
