import os
import json
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from collections import defaultdict
from pathlib import Path
from pdb_contact_args import ContactsAnalyzerArgsParser


class AvgModelsPDB:
    """
    A class to analyze contacts between protein chains in PDB files.
    """

    def __init__(self, distance_cutoff=8.0, pdb_prefix="ranked_", pdb_indices=None):
        """
        Initialize the ProteinContactsAnalyzer.

        Args:
            distance_cutoff (float): Distance cutoff in Angstroms to define a contact
            pdb_prefix (str): Prefix for PDB files
            pdb_indices (list): Indices for PDB files to analyze
        """
        self.distance_cutoff = distance_cutoff
        self.pdb_prefix = pdb_prefix
        self.pdb_indices = pdb_indices if pdb_indices is not None else [0, 1, 2, 3, 4]

    def extract_contacts(self, pdb_file):
        """
        Extract contacts between chains B and C in a PDB file.

        Args:
            pdb_file (str): Path to the PDB file

        Returns:
            set: Set of tuples (residue1, residue2) representing contacts
        """
        # Parse PDB file
        parser = PDBParser(QUIET=True)
        structure_id = os.path.basename(pdb_file).split('.')[0]
        try:
            structure = parser.get_structure(structure_id, pdb_file)
        except Exception as e:
            print(f"Error parsing {pdb_file}: {e}")
            return set()

        # Get atoms from chains B and C
        atoms_chain_b = []
        atoms_chain_c = []

        for model in structure:
            if 'B' in model and 'C' in model:
                for residue in model['B']:
                    if residue.id[0] == ' ':  # Filter out hetero-atoms, waters
                        for atom in residue:
                            atoms_chain_b.append(atom)

                for residue in model['C']:
                    if residue.id[0] == ' ':  # Filter out hetero-atoms, waters
                        for atom in residue:
                            atoms_chain_c.append(atom)

        # Find contacts using NeighborSearch
        contacts = set()
        ns = NeighborSearch(atoms_chain_c)

        for atom1 in atoms_chain_b:
            residue1 = atom1.get_parent()
            res1_id = ('B', residue1.id[1], residue1.resname)

            neighbors = ns.search(atom1.coord, self.distance_cutoff)
            for atom2 in neighbors:
                residue2 = atom2.get_parent()
                res2_id = ('C', residue2.id[1], residue2.resname)

                # Add the contact as a tuple of residue identifiers
                contacts.add((res1_id, res2_id))

        return contacts

    def analyze_contacts_across_structures(self, pdb_files):
        """
        Compare contacts across PDB files, calculate frequencies,
        normalize by the number of files, sum the normalized values,
        and divide by number of unique contacts.

        Args:
            pdb_files (list): List of paths to PDB files

        Returns:
            dict: Dictionary with contacts as keys and normalized frequencies as values
            float: Sum of all normalized frequencies
            int: Total number of unique contacts
            float: Final metric - sum of normalized frequencies divided by number of unique contacts
        """
        # Extract and compare contacts from each PDB file
        all_unique_contacts = set()
        contact_counts = defaultdict(int)

        for pdb_file in pdb_files:
            contacts = self.extract_contacts(pdb_file)

            # Store all unique contacts and count occurrences
            for contact in contacts:
                all_unique_contacts.add(contact)
                contact_counts[contact] += 1

        # Calculate frequencies and normalize by dividing by number of files
        normalized_frequencies = {}
        for contact in all_unique_contacts:
            normalized_frequencies[contact] = contact_counts[contact] / len(pdb_files)

        # Sum all normalized frequencies
        sum_normalized = sum(normalized_frequencies.values())

        # Divide the sum by the number of unique contacts
        num_unique_contacts = len(all_unique_contacts)
        final_metric = sum_normalized / num_unique_contacts if num_unique_contacts > 0 else 0

        return normalized_frequencies, sum_normalized, num_unique_contacts, final_metric

    @staticmethod
    def format_contact(contact):
        """
        Format a contact tuple for pretty printing.

        Args:
            contact (tuple): A contact tuple ((chain1, resid1, resname1), (chain2, resid2, resname2))

        Returns:
            str: Formatted contact string
        """
        chain1, resid1, resname1 = contact[0]
        chain2, resid2, resname2 = contact[1]
        return f"{resname1}_{resid1}_{chain1} - {resname2}_{resid2}_{chain2}"

    def analyze_folder(self, folder_path, output_format="csv", detailed=False):
        """
        Analyze contacts for a specific folder containing PDB files.

        Args:
            folder_path (str): Path to folder containing PDB files
            output_format (str): Output format (csv or text)
            detailed (bool): Whether to save detailed contact information

        Returns:
            tuple: (folder_name, num_unique_contacts, sum_normalized_frequencies, final_metric)
        """
        # Find the PDB files in the specified folder
        pdb_files = [
            os.path.join(folder_path, f"{self.pdb_prefix}{idx}.pdb") for idx in self.pdb_indices
        ]

        # Check if all files exist
        missing_files = [f for f in pdb_files if not os.path.exists(f)]
        if missing_files:
            print(f"Warning: Missing files in {folder_path}: {', '.join([os.path.basename(f) for f in missing_files])}")
            return None

        # Analyze contacts between chains B and C
        normalized_contacts, sum_normalized, num_unique_contacts, final_metric = self.analyze_contacts_across_structures(
            pdb_files)

        folder_name = os.path.basename(os.path.normpath(folder_path))

        # Output summary line
        if output_format == "csv":
            result = f"{folder_name},{num_unique_contacts},{sum_normalized:.4f},{final_metric:.4f}"
        else:
            result = (
                f"Folder: {folder_name}\n"
                f"Total unique contacts: {num_unique_contacts}\n"
                f"Sum of normalized frequencies: {sum_normalized:.4f}\n"
                f"Final metric (sum/unique contacts): {final_metric:.4f}"
            )

        # Optionally, write detailed contacts to a file
        if detailed:
            output_file = os.path.join(folder_path, "contacts_analysis.txt")
            with open(output_file, 'w') as f:
                f.write(f"Folder: {folder_path}\n")
                f.write(f"Total unique contacts between chains B and C: {num_unique_contacts}\n")
                f.write(f"Sum of normalized contact frequencies: {sum_normalized:.4f}\n")
                f.write(f"Final metric (sum/unique contacts): {final_metric:.4f}\n\n")
                f.write("Contact frequency analysis:\n")
                f.write("=========================\n")
                f.write("Contact | Occurrence Count | Normalized Frequency\n")

                for contact, frequency in sorted(normalized_contacts.items(), key=lambda x: x[1], reverse=True):
                    count = int(frequency * len(pdb_files))  # Convert back to occurrence count
                    f.write(f"{self.format_contact(contact)} | {count}/{len(pdb_files)} | {frequency:.2f}\n")

        return folder_name, num_unique_contacts, sum_normalized, final_metric

    def process_batch(self, parent_dir, output_file='contacts_summary.csv', detailed=False):
        """
        Process multiple folders according to the configuration.

        Args:
            parent_dir (str): Parent directory containing folders with PDB files
            output_file (str): Output file path for summary CSV
            detailed (bool): Whether to save detailed contact information for each folder

        Returns:
            int: Number of folders processed
        """
        # Write header to the CSV file
        with open(output_file, 'w') as f:
            f.write("protein1,protein2,final_metric\n")

        # Process all subdirectories
        processed = 0
        for dir_name in os.listdir(parent_dir):
            dir_path = os.path.join(parent_dir, dir_name)
            if os.path.isdir(dir_path):
                print(f"Processing {dir_path}...")
                result = self.analyze_folder(
                    dir_path,
                    "csv",
                    detailed
                )
                if result:
                    # Extract protein1 and protein2 from the folder name
                    # Assumes folder name format is like "protein1_protein2"
                    folder_name = result[0]
                    proteins = folder_name.split('_', 1)  # Split at first underscore
                    protein1 = proteins[0] if len(proteins) > 0 else folder_name
                    protein2 = proteins[1] if len(proteins) > 1 else ""

                    # Write only protein1, protein2, and final_metric to CSV
                    with open(output_file, 'a') as f:
                        f.write(f"{protein1},{protein2},{result[3]:.4f}\n")

                    processed += 1
                    print(f"Completed processing {dir_path}")

        print(f"Analysis complete. Processed {processed} folders. Results saved to {output_file}")
        return processed

    def run_from_config(self, config_file):
        """
        Run the analysis based on a JSON configuration file.

        Args:
            config_file (str): Path to JSON configuration file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Parse the config file using the ContactsAnalyzerArgsParser
            config_parser = ContactsAnalyzerArgsParser(config_file)
            config = config_parser.parse()

            # Update instance variables with config values
            self.distance_cutoff = config.cutoff
            self.pdb_prefix = config.pdb_prefix
            self.pdb_indices = config.pdb_indices

            # Process based on mode
            if config.mode == 'folder':
                result = self.analyze_folder(
                    config.folder_path,
                    config.output_format,
                    config.detailed
                )

                if result:
                    print(result[0] if config.output_format == 'csv' else result)
                    return True

            elif config.mode == 'batch':
                processed = self.process_batch(
                    config.parent_dir,
                    config.output_file,
                    config.detailed
                )
                return processed > 0

            else:
                print(f"Unknown mode: {config.mode}. Supported modes are 'folder' and 'batch'.")
                return False

        except Exception as e:
            print(f"Error processing config file: {e}")
            return False

        return False



