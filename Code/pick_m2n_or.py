import os
import pickle
import networkx as nx
import re

def get_available_file_numbers(datasetdir):
    dirpath = os.path.join(datasetdir)
    
    if not os.path.exists(dirpath):
        print(f"Directory not found: {dirpath}")
        return []
    
    file_numbers = []
    for filename in os.listdir(dirpath):
        match = re.search(r'data-set-(\d+)\.pkl', filename)
        if match:
            file_numbers.append(int(match.group(1)))
    
    return sorted(file_numbers)

def process_and_save_datasets(input_dir, output_dir, min_file_number):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get available file numbers
    file_numbers = get_available_file_numbers(input_dir)

    # Process files with numbers greater than min_file_number
    for file_number in file_numbers:
        if file_number > min_file_number:
            filename = f'data-set-{file_number}.pkl'
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read the pickle file
            with open(input_path, 'rb') as f:
                data_set = pickle.load(f)

            # Remove 'graph_type' attribute from the graph if it exists
            if 'G' in data_set and isinstance(data_set['G'], nx.Graph):
                G = data_set['G']
                
                # Before removal
                # sample_node = list(G.nodes())[0]  # Get the first node as a sample
                # print(f"Sample node attributes before removal: {G.nodes[sample_node]}")

                # Remove 'graph_type' attribute
                for node in G.nodes():
                    if 'graph_type' in G.nodes[node]:
                        del G.nodes[node]['graph_type']

                # After removal
                # print(f"Sample node attributes after removal: {G.nodes[sample_node]}")

                # Update the graph in the data_set
                data_set['G'] = G

                # Save the modified data set to the new location
                with open(output_path, 'wb') as f:
                    pickle.dump(data_set, f)

                print(f"Processed and saved file number {file_number}: {output_path}")
            else:
                print(f"Skipped file number {file_number} - {input_path}: 'G' not found or not a NetworkX graph")

# Usage
input_directory = "/home/emu/Documents/surrogate/dataset/pickle/m2n/cs8-BU/"
output_directory = "/home/emu/Documents/surrogate/dataset/pickle/m2n/cs8/"
minimum_file_number = 0

process_and_save_datasets(input_directory, output_directory, minimum_file_number)
