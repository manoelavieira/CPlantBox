#!/usr/bin/env python3
"""
Extract specific data from phloem_**.txt files.

For each file:
- Extract the line that starts with "starting integration on time segment"
- Look for content between "[FINAL] Solver results" and "[FINAL] Done"
- For each "C_fluxes" block, extract: Q_ST[i], F_in, F_out, Delta_JS_ST
"""

import os
import re
import csv
import math
from pathlib import Path


def extract_phloem_data(file_path):
    """Extract data from a single phloem file."""
    results = []

    with open(file_path, 'r') as f:
        content = f.read()

    # Extract "starting integration on time segment" line and parse time values
    time_segment_match = re.search(r'starting integration on time segment.*?\[([^\]]+)\]', content)
    if time_segment_match:
        time_values = time_segment_match.group(1).split(',')
        initial_time = float(time_values[0].strip())
        final_time = float(time_values[1].strip())
        time_interval_days = final_time - initial_time
        time_interval_hours = time_interval_days * 24  # Convert days to hours
    else:
        initial_time = None
        final_time = None
        time_interval_days = None
        time_interval_hours = None

    # Find content between [FINAL] Solver results and [FINAL] Done
    final_match = re.search(r'\[FINAL\] Solver results(.*?)\[FINAL\] Done', content, re.DOTALL)

    if not final_match:
        print(f"Warning: Could not find [FINAL] section in {file_path}")
        return None

    final_content = final_match.group(1)

    # Find all C_fluxes blocks
    # Pattern: C_fluxes followed by node_id, then extract values from subsequent lines
    c_flux_blocks = re.finditer(r'C_fluxes (\d+).*?(?=C_fluxes|\Z)', final_content, re.DOTALL)

    for block_match in c_flux_blocks:
        node_id = block_match.group(1)
        block_text = block_match.group(0)

        # Extract Q_ST[i] - it appears on the line with " C_ST[i]" followed by Q_ST[i]
        q_st_match = re.search(r'C_ST\[i\]\s+([\d.\-e]+)\s+Q_ST\[i\]\s+([\d.\-e]+)', block_text)
        q_st_value = q_st_match.group(2) if q_st_match else "Not found"

        # Extract F_in and F_out from the line containing "Delta_JS_ST"
        flux_match = re.search(r'Delta_JS_ST\s+([\d.\-e]+)\s+F_in\s+([\d.\-e]+)\s+F_out\s+([\d.\-e]+)', block_text)

        if flux_match:
            delta_js_st = flux_match.group(1)
            f_in = flux_match.group(2)
            f_out = flux_match.group(3)
        else:
            delta_js_st = "Not found"
            f_in = "Not found"
            f_out = "Not found"

        results.append({
            'node_id': node_id,
            'Q_ST[i]': q_st_value,
            'F_in': f_in,
            'F_out': f_out,
            'Delta_JS_ST': delta_js_st
        })

    return {
        'initial_time': initial_time,
        'final_time': final_time,
        'time_interval_days': time_interval_days,
        'time_interval_hours': time_interval_hours,
        'c_flux_data': results
    }


def main():
    """Main function to process all phloem files."""
    data_dir = Path(__file__).parent
    output_file = data_dir / 'phloem_extracted_data.csv'
    analysis_file = data_dir / 'phloem_analysis.csv'
    statistics_file = data_dir / 'phloem_statistics.csv'

    # Find all phloem_**.txt files
    phloem_files = sorted(data_dir.glob('phloem_*.txt'))

    # Remove the .h5 file if it's in the list
    phloem_files = [f for f in phloem_files if f.suffix == '.txt']

    print(f"Found {len(phloem_files)} phloem files")

    # Dictionary to store previous Q_ST[i] values for each node
    # Key: node_id, Value: Q_ST[i] from previous time step
    previous_q_st = {}

    # List to store analysis data
    analysis_data = []

    # List to store statistics per time interval
    statistics_data = []

    # Prepare CSV output
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'initial_time', 'final_time', 'time_interval_days', 'time_interval_hours',
                      'node_id', 'Q_ST[i]', 'F_in', 'F_out', 'Delta_JS_ST']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each file
        for file_path in phloem_files:
            print(f"Processing {file_path.name}...")

            try:
                data = extract_phloem_data(file_path)

                if data is None:
                    continue

                time_interval_hours = data['time_interval_hours']

                # List to store ds_dt/delta_flux values for this time interval
                ds_dt_over_delta_flux_values = []

                # Write data for each C_fluxes block
                for flux_data in data['c_flux_data']:
                    node_id = flux_data['node_id']
                    current_q_st = float(flux_data['Q_ST[i]']) if flux_data['Q_ST[i]'] != "Not found" else 0.0
                    delta_js_st = float(flux_data['Delta_JS_ST']) if flux_data['Delta_JS_ST'] != "Not found" else 0.0

                    # Get previous Q_ST[i] value (default to 0 if not found)
                    prev_q_st = previous_q_st.get(node_id, 0.0)

                    # Calculate ds_dt
                    if time_interval_hours is not None and time_interval_hours != 0:
                        ds_dt = (current_q_st - prev_q_st) / time_interval_hours
                    else:
                        ds_dt = 0.0

                    # Calculate ds_dt / delta_flux
                    if delta_js_st != 0:
                        ds_dt_over_delta_flux = ds_dt / delta_js_st
                    else:
                        ds_dt_over_delta_flux = float('inf') if ds_dt != 0 else 0.0

                    # Collect valid (non-infinite) values for statistics
                    if not math.isinf(ds_dt_over_delta_flux):
                        ds_dt_over_delta_flux_values.append(ds_dt_over_delta_flux)

                    # Store analysis data
                    analysis_data.append({
                        'filename': file_path.name,
                        'initial_time': data['initial_time'],
                        'final_time': data['final_time'],
                        'time_interval_hours': time_interval_hours,
                        'node_id': node_id,
                        'Q_ST[i]': current_q_st,
                        'previous_Q_ST[i]': prev_q_st,
                        'ds_dt': ds_dt,
                        'delta_flux': delta_js_st,
                        'ds_dt/delta_flux': ds_dt_over_delta_flux
                    })

                    # Update previous Q_ST[i] for this node
                    previous_q_st[node_id] = current_q_st

                    writer.writerow({
                        'filename': file_path.name,
                        'initial_time': data['initial_time'],
                        'final_time': data['final_time'],
                        'time_interval_days': data['time_interval_days'],
                        'time_interval_hours': time_interval_hours,
                        'node_id': flux_data['node_id'],
                        'Q_ST[i]': flux_data['Q_ST[i]'],
                        'F_in': flux_data['F_in'],
                        'F_out': flux_data['F_out'],
                        'Delta_JS_ST': flux_data['Delta_JS_ST']
                    })

                # Calculate statistics for this time interval
                if ds_dt_over_delta_flux_values:
                    abs_values = [abs(v) for v in ds_dt_over_delta_flux_values]
                    mean_abs = sum(abs_values) / len(abs_values)
                    min_abs = min(abs_values)
                    max_abs = max(abs_values)
                else:
                    mean_abs = 0.0
                    min_abs = 0.0
                    max_abs = 0.0

                statistics_data.append({
                    'filename': file_path.name,
                    'initial_time': data['initial_time'],
                    'final_time': data['final_time'],
                    'time_interval_hours': time_interval_hours,
                    'mean_abs_ds_dt_over_delta_flux': mean_abs,
                    'min_abs_ds_dt_over_delta_flux': min_abs,
                    'max_abs_ds_dt_over_delta_flux': max_abs,
                    'num_nodes': len(ds_dt_over_delta_flux_values)
                })

            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                continue

    print(f"\nExtraction complete! Data saved to: {output_file}")
    print(f"Total files processed: {len(phloem_files)}")

    # Write analysis file
    print(f"\nWriting analysis data to: {analysis_file}")
    with open(analysis_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'initial_time', 'final_time', 'time_interval_hours', 'node_id',
                      'Q_ST[i]', 'previous_Q_ST[i]', 'ds_dt', 'delta_flux', 'ds_dt/delta_flux']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in analysis_data:
            writer.writerow(row)

    print(f"Analysis complete! Data saved to: {analysis_file}")
    print(f"Total analysis rows: {len(analysis_data)}")

    # Write statistics file
    print(f"\nWriting statistics data to: {statistics_file}")
    with open(statistics_file, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'initial_time', 'final_time', 'time_interval_hours',
                      'mean_abs_ds_dt_over_delta_flux', 'min_abs_ds_dt_over_delta_flux',
                      'max_abs_ds_dt_over_delta_flux', 'num_nodes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in statistics_data:
            writer.writerow(row)

    print(f"Statistics complete! Data saved to: {statistics_file}")
    print(f"Total statistics rows: {len(statistics_data)}")


if __name__ == '__main__':
    main()
