import os
import shutil
from Validation_class import prep_data_cooling_surrogate, surrogate_testing



# === USER INPUTS ===
base_path = "/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Training_data_group/"  # Replace with the actual folder path
selected_folders = ["Cup", "Long", "Medium", "Short"]
train_group_count = 3  # Number of groups used for training in each split

# Get the full paths of selected folders
group_folders = surrogate_testing.get_group_folders(base_path, selected_folders)

# Generate all possible train-validation splits
splits = surrogate_testing.generate_train_test_splits(group_folders, train_group_count)

# Copy folders
train_folder = "/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Train_data_del/"
validation_folder = "/Users/tristanhirs/Downloads/Thesis/Measurements_cup/Measurements/Val_data_del/"

# === CROSS-VALIDATION LOOP ===
for i, (train_folders, validation_folders) in enumerate(splits):
    print(f"\n--- Fold {i+1} ---")
    print(f"Train Folders: {train_folders}")
    print(f"Validation Folders: {validation_folders}")

    # Copy training data (all at once)
    surrogate_testing.copy_files(train_folders, train_folder)
    
    # Obtain training and validation data
    X_train, y_train, X_val_train, y_val_train = prep_data_cooling_surrogate(train_folder, scaled=False, n_outputs=1, n_samples=50, start_time=None, end_time=None).train_val_split()
    
    print(X_train.shape)

    # Get validation file paths (from all validation folders)
    validation_files = surrogate_testing().get_file_paths(validation_folders)

    for val_file in validation_files:
        # Copy **one** validation file at a time
        os.makedirs(validation_folder, exist_ok=True)
        shutil.copy(val_file, os.path.join(validation_folder, os.path.basename(val_file)))

        # Obtain experimental data
        X_test, T_true, _, _, _, _ = prep_data_cooling_surrogate(validation_folder, scaled=False, n_outputs=1, n_samples=1e6, start_time=None, end_time=None).compare_with_data()

        # Remove the file after validation
        surrogate_testing.delete_files(validation_folder)

    # Clean up training data after the fold
    surrogate_testing.delete_files(train_folder)
