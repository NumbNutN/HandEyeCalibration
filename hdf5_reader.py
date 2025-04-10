## list all group in hdf5 file
import h5py

def list_groups(file_path):
    with h5py.File(file_path, 'r') as f:
        def printname(name):
            print(name)
        f.visit(printname)

# Example usage
list_groups('./data/new/episode_0.hdf5')  # Replace with your HDF5 file path