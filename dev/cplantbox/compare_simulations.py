#!/usr/bin/env python3
"""
Compare two CPlantBox phloem simulation HDF5 files.

This script compares plant structure and sucrose-related variables between
two simulation outputs to verify if they are identical or significantly different.

Usage:
    python compare_simulations.py <file1.h5> <file2.h5>

Example:
    python compare_simulations.py data/sim_02/phloem_simulation.h5 data/sim_03/phloem_simulation.h5
"""

import sys
import argparse
import h5py
import numpy as np
from pathlib import Path


def print_header(text, char='='):
    """Print a formatted header."""
    width = 80
    print('\n' + char * width)
    print(text.center(width))
    print(char * width)


def print_section(text):
    """Print a section header."""
    print(f'\n{text}')
    print('-' * 80)


def compare_structure(f1, f2, sim1_name, sim2_name):
    """Compare the plant structure between two simulations."""
    print_section('1. PLANT STRUCTURE COMPARISON')

    # Get first step
    step_key = 'step_000'

    if step_key not in f1 or step_key not in f2:
        print('  ⚠️  Warning: step_000 not found in one or both files')
        return

    # Compare number of nodes
    nodes_1 = f1[step_key]['nodes']
    nodes_2 = f2[step_key]['nodes']

    n_nodes_1 = len(list(nodes_1.keys()))
    n_nodes_2 = len(list(nodes_2.keys()))

    # Try to get actual node count from a data array
    for key in nodes_1.keys():
        try:
            data = np.array(nodes_1[key])
            if len(data.shape) == 1:
                n_nodes_1 = len(data)
                break
        except:
            pass

    for key in nodes_2.keys():
        try:
            data = np.array(nodes_2[key])
            if len(data.shape) == 1:
                n_nodes_2 = len(data)
                break
        except:
            pass

    print(f'\n  Number of nodes:')
    print(f'    {sim1_name}: {n_nodes_1}')
    print(f'    {sim2_name}: {n_nodes_2}')

    if n_nodes_1 == n_nodes_2:
        print(f'    ✓ Same number of nodes')
    else:
        print(f'    ⚠️  Different number of nodes (Δ = {abs(n_nodes_1 - n_nodes_2)})')

    # Compare segments
    if 'segments' in f1[step_key] and 'segments' in f2[step_key]:
        seg_1 = f1[step_key]['segments']
        seg_2 = f2[step_key]['segments']

        if 'connectivity' in seg_1 and 'connectivity' in seg_2:
            conn_1 = np.array(seg_1['connectivity'])
            conn_2 = np.array(seg_2['connectivity'])

            print(f'\n  Number of segments:')
            print(f'    {sim1_name}: {len(conn_1)}')
            print(f'    {sim2_name}: {len(conn_2)}')

            if len(conn_1) == len(conn_2):
                identical = np.array_equal(conn_1, conn_2)
                if identical:
                    print(f'    ✓ Same number and IDENTICAL connectivity')
                else:
                    print(f'    ✓ Same number but DIFFERENT connectivity')
            else:
                print(f'    ⚠️  Different number of segments (Δ = {abs(len(conn_1) - len(conn_2))})')

        # Compare organ types
        if 'organ_types' in seg_1 and 'organ_types' in seg_2:
            types_1 = np.array(seg_1['organ_types'])
            types_2 = np.array(seg_2['organ_types'])

            if np.array_equal(types_1, types_2):
                print(f'    ✓ Identical organ types')
            else:
                print(f'    ⚠️  Different organ types')


def compare_variable(data_1, data_2, var_name, sim1_name, sim2_name, verbose=False):
    """Compare a single variable between two simulations."""
    if not np.issubdtype(data_1.dtype, np.number):
        return None

    mean_1 = data_1.mean()
    mean_2 = data_2.mean()
    std_1 = data_1.std()
    std_2 = data_2.std()
    min_1 = data_1.min()
    min_2 = data_2.min()
    max_1 = data_1.max()
    max_2 = data_2.max()

    diff_max = np.abs(data_1 - data_2).max()
    diff_mean = np.abs(data_1 - data_2).mean()

    mean_val = (np.abs(mean_1) + np.abs(mean_2)) / 2
    rel_diff_max = (diff_max / mean_val * 100) if mean_val > 1e-20 else 0
    rel_diff_mean = (diff_mean / mean_val * 100) if mean_val > 1e-20 else 0

    # Determine status
    if diff_max < 1e-10:
        status = '⚠️  IDENTICAL'
        status_code = 0
    elif rel_diff_max < 2:
        status = '~ Nearly identical (< 2%)'
        status_code = 1
    elif rel_diff_max < 10:
        status = '≈ Small difference (< 10%)'
        status_code = 2
    else:
        status = '✓ Significant difference (≥ 10%)'
        status_code = 3

    print(f'\n  {var_name}:')
    if verbose:
        print(f'    {sim1_name}: mean={mean_1:.6e}, std={std_1:.6e}, min={min_1:.6e}, max={max_1:.6e}')
        print(f'    {sim2_name}: mean={mean_2:.6e}, std={std_2:.6e}, min={min_2:.6e}, max={max_2:.6e}')
        print(f'    Absolute diff: max={diff_max:.6e}, mean={diff_mean:.6e}')
        print(f'    Relative diff: max={rel_diff_max:.2f}%, mean={rel_diff_mean:.2f}%')
    else:
        print(f'    {sim1_name} mean: {mean_1:.6e}')
        print(f'    {sim2_name} mean: {mean_2:.6e}')
        print(f'    Max diff: {diff_max:.6e} ({rel_diff_max:.2f}% relative)')
    print(f'    Status: {status}')

    return {
        'name': var_name,
        'mean_1': mean_1,
        'mean_2': mean_2,
        'diff_max': diff_max,
        'rel_diff_max': rel_diff_max,
        'status_code': status_code,
        'status': status
    }


def compare_sucrose_variables(f1, f2, sim1_name, sim2_name, step_key='step_000'):
    """Compare sucrose-related variables."""
    print_section('2. SUCROSE VARIABLES COMPARISON (Initial Step)')

    if step_key not in f1 or step_key not in f2:
        print(f'  ⚠️  Warning: {step_key} not found in one or both files')
        return []

    nodes_1 = f1[step_key]['nodes']
    nodes_2 = f2[step_key]['nodes']

    # Priority sucrose variables
    priority_vars = ['Q_ST', 'C_ST_np', 'C_meso']
    other_sucrose_vars = [k for k in nodes_1.keys()
                          if ('ST' in k or 'C_' in k or 'sucrose' in k.lower())
                          and k not in priority_vars]

    results = []

    # Compare priority variables first
    print('\n  Key Sucrose Variables:')
    for var in priority_vars:
        if var in nodes_1 and var in nodes_2:
            data_1 = np.array(nodes_1[var])
            data_2 = np.array(nodes_2[var])

            if len(data_1) != len(data_2):
                print(f'\n  {var}:')
                print(f'    ⚠️  Different array lengths: {len(data_1)} vs {len(data_2)}')
                continue

            result = compare_variable(data_1, data_2, var, sim1_name, sim2_name, verbose=True)
            if result:
                results.append(result)

    # Compare other sucrose variables
    if other_sucrose_vars:
        print('\n  Other Sucrose-Related Variables:')
        for var in sorted(other_sucrose_vars):
            if var in nodes_2:
                data_1 = np.array(nodes_1[var])
                data_2 = np.array(nodes_2[var])

                if len(data_1) != len(data_2):
                    continue

                result = compare_variable(data_1, data_2, var, sim1_name, sim2_name, verbose=False)
                if result:
                    results.append(result)

    return results


def compare_time_evolution(f1, f2, sim1_name, sim2_name, variables=['C_ST_np', 'Q_ST', 'C_meso']):
    """Compare how variables evolve over time."""
    print_section('3. TIME EVOLUTION COMPARISON')

    # Find all time steps
    steps_1 = sorted([k for k in f1.keys() if k.startswith('step_')])
    steps_2 = sorted([k for k in f2.keys() if k.startswith('step_')])

    print(f'\n  Number of time steps:')
    print(f'    {sim1_name}: {len(steps_1)}')
    print(f'    {sim2_name}: {len(steps_2)}')

    common_steps = sorted(set(steps_1) & set(steps_2))

    if not common_steps:
        print('  ⚠️  No common time steps found')
        return

    # Sample steps for comparison (first, middle, last few)
    if len(common_steps) > 10:
        sample_steps = ([common_steps[0]] +
                       common_steps[len(common_steps)//4::len(common_steps)//4][:3] +
                       common_steps[-3:])
        sample_steps = sorted(set(sample_steps))
    else:
        sample_steps = common_steps

    for var in variables:
        print(f'\n  {var} evolution:')
        print(f'    {"Step":<12} {sim1_name+" mean":<18} {sim2_name+" mean":<18} {"Max Diff":<15} {"Rel %":<10}')
        print('    ' + '-' * 75)

        for step in sample_steps:
            try:
                data_1 = np.array(f1[step]['nodes'][var])
                data_2 = np.array(f2[step]['nodes'][var])

                if len(data_1) != len(data_2):
                    print(f'    {step:<12} ⚠️  Different lengths: {len(data_1)} vs {len(data_2)}')
                    continue

                mean_1 = data_1.mean()
                mean_2 = data_2.mean()
                diff_max = np.abs(data_1 - data_2).max()
                mean_val = (np.abs(mean_1) + np.abs(mean_2)) / 2
                rel_diff = (diff_max / mean_val * 100) if mean_val > 1e-20 else 0

                print(f'    {step:<12} {mean_1:<18.6e} {mean_2:<18.6e} {diff_max:<15.6e} {rel_diff:<10.2f}')

            except Exception as e:
                print(f'    {step:<12} ⚠️  Error: {str(e)[:40]}')


def generate_summary(structure_same, results):
    """Generate a summary of the comparison."""
    print_header('SUMMARY', '=')

    if structure_same:
        print('\n  Plant Structure: ✓ SAME (identical topology)')
    else:
        print('\n  Plant Structure: ⚠️  DIFFERENT (different number of nodes/segments)')

    if not results:
        print('\n  No sucrose variables compared.')
        return

    # Count by status
    identical = sum(1 for r in results if r['status_code'] == 0)
    nearly_identical = sum(1 for r in results if r['status_code'] == 1)
    small_diff = sum(1 for r in results if r['status_code'] == 2)
    significant_diff = sum(1 for r in results if r['status_code'] == 3)

    print(f'\n  Sucrose Variables Summary:')
    print(f'    Total compared: {len(results)}')
    print(f'    Identical (< 1e-10): {identical}')
    print(f'    Nearly identical (< 2%): {nearly_identical}')
    print(f'    Small difference (< 10%): {small_diff}')
    print(f'    Significant difference (≥ 10%): {significant_diff}')

    # Overall verdict
    print('\n  Overall Verdict:')
    if identical == len(results):
        print('    ⚠️⚠️⚠️  SIMULATIONS ARE IDENTICAL  ⚠️⚠️⚠️')
        print('    The same seed was likely used for both simulations.')
    elif (identical + nearly_identical) >= len(results) * 0.8:
        print('    ~ Simulations are VERY SIMILAR (< 2% difference in most variables)')
        print('    May indicate same seed or very similar plant structures.')
    elif (identical + nearly_identical + small_diff) >= len(results) * 0.8:
        print('    ≈ Simulations are SIMILAR (< 10% difference in most variables)')
        print('    Likely same topology with small geometric variations.')
    else:
        print('    ✓ Simulations are SIGNIFICANTLY DIFFERENT')
        print('    Different seeds produced different plant architectures.')

    print('\n' + '=' * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Compare two CPlantBox phloem simulation HDF5 files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_simulations.py data/sim_02/phloem_simulation.h5 data/sim_03/phloem_simulation.h5
  python compare_simulations.py sim1.h5 sim2.h5 --verbose
        """
    )

    parser.add_argument('file1', type=str, help='Path to first HDF5 simulation file')
    parser.add_argument('file2', type=str, help='Path to second HDF5 simulation file')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed statistics for all variables')

    args = parser.parse_args()

    # Validate files
    file1 = Path(args.file1)
    file2 = Path(args.file2)

    if not file1.exists():
        print(f'Error: File not found: {file1}')
        sys.exit(1)

    if not file2.exists():
        print(f'Error: File not found: {file2}')
        sys.exit(1)

    # Get simulation names from paths
    sim1_name = file1.parent.name if file1.parent.name.startswith('sim_') else file1.stem
    sim2_name = file2.parent.name if file2.parent.name.startswith('sim_') else file2.stem

    # Open and compare files
    print_header(f'COMPARING: {sim1_name} vs {sim2_name}')
    print(f'\n  File 1: {file1}')
    print(f'  File 2: {file2}')

    try:
        with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:

            # Compare structure
            compare_structure(f1, f2, sim1_name, sim2_name)

            # Determine if structure is the same
            step_key = 'step_000'
            structure_same = False
            if step_key in f1 and step_key in f2:
                if 'segments' in f1[step_key] and 'segments' in f2[step_key]:
                    if 'connectivity' in f1[step_key]['segments'] and 'connectivity' in f2[step_key]['segments']:
                        conn_1 = np.array(f1[step_key]['segments']['connectivity'])
                        conn_2 = np.array(f2[step_key]['segments']['connectivity'])
                        structure_same = np.array_equal(conn_1, conn_2)

            # Compare sucrose variables
            results = compare_sucrose_variables(f1, f2, sim1_name, sim2_name)

            # Compare time evolution
            compare_time_evolution(f1, f2, sim1_name, sim2_name)

            # Generate summary
            generate_summary(structure_same, results)

    except Exception as e:
        print(f'\nError reading files: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
