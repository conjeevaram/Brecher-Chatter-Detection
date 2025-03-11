import csv
import sys
import os


def process_csv(input_file):
    # Generate output filename in the same directory
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_timecalc.csv"

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
            # Replace the first column (T) with the calculated time value.
            row[0] = f"{time_val:.6f}"  # formatted to 6 decimal places
            writer.writerow(row)
            time_val += increment


if __name__ == "__main__":
        input_csv = "C:/LJM-Data-Collection/csv/8500_2.csv"
        process_csv(input_csv)
        print(f"Processed file saved as {os.path.splitext(input_csv)[0]}_timecalc.csv")
