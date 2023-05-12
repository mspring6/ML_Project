import pickle
import csv

def sort_dict(unsorted_dict):
    """
    Function to sort a dictionary by value
    """

    unsorted_dict_keys = list(unsorted_dict.keys())
    unsorted_dict_keys.sort()
    sorted_dict = {i: unsorted_dict[i] for i in unsorted_dict_keys}

    return sorted_dict

def export_dict_to_txt(export_dict, filename=None):
    """
    Function to convert a dictionary into a text file
    """

    if filename is None:
        raise Exception('Please enter a proper "filename" parameter...')
    else:
        export_file = open(filename, 'wb')
        pickle.dump(export_dict, export_file)
        export_file.close()

def pickle_load_dict_from_txt(filename=None):
    """
    Function to load a dictionary from a text file
    """

    if filename is None:
        raise Exception('Please enter a proper "filename" parameter...')
    else:
        with open(filename, 'rb') as f:
            pickle_dict = pickle.load(f)
        
        return pickle_dict
    
def export_dict_to_csv(export_dict, filename=None, csv_header=None):
    """
    Function to convert a dictionary into a csv file
    """

    if filename is None: 
        raise Exception('Please enter a proper "filename" parameter...')
    else:
        csv_entries = []

        for key, value in export_dict.items():
            dict_entry = {csv_header[0]:f'({str(key[0][0])},{str(key[0][1])})', csv_header[1]:f'{key[1]}', csv_header[2]:value}
            csv_entries.append(dict_entry)

        with open(filename, 'w', newline='') as csvfile:

            if csv_header is not None:
                writer = csv.DictWriter(csvfile, fieldnames=csv_header)
                writer.writeheader()
            else:
                writer = csv.DictWriter(csvfile)
            writer.writerows(csv_entries)
