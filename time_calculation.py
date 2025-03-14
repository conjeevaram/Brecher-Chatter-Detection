import csv
import os

# List of CSV files to process
csv_files = [
    "7500_1.csv", "7500_2.csv", "7500_3.csv",
    "8000_1.csv", "8000_2.csv", "8000_3.csv", "8000_4.csv",
    "8500_1.csv", "8500_2.csv", "8500_3.csv", "8500_4.csv",
    "9000_1.csv", "9000_2.csv",
    "9500_1.csv", "9500_2.csv"
]

# Define the input folder containing the CSV files
input_folder = "C:/LJM-Data-Collection/csv"

# Define and create the output folder
output_folder = "Timestamp Regularized Training Data"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def process_csv(input_file, output_file):
    """Processes a CSV file by normalizing its timestamps and saving the result."""
    with open(input_file, newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Read and write the header row unchanged
        header = next(reader)
        writer.writerow(header)

        # Initialize starting time and increment value
        time_val = 0.0
        increment = 0.000125

        # Process each row after the header
        for row in reader:
            # Replace the first column (T) with the calculated time value formatted to 6 decimals
            row[0] = f"{time_val:.6f}"
            writer.writerow(row)
            time_val += increment


if __name__ == "__main__":
    for csv_name in csv_files:
        input_csv = os.path.join(input_folder, csv_name)
        base, _ = os.path.splitext(csv_name)
        output_csv = os.path.join(output_folder, f"{base}_timecalc.csv")

        # Process each file and save the output
        process_csv(input_csv, output_csv)
        print(f"Processed file saved as {output_csv}")
