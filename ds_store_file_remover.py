import os

folder_path = "/Users/tristanhirs/Downloads/Thesis/Motor test data/Training_data_group/6"

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file == ".DS_Store":
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"🗑 Removed: {file_path}")
            print("✅ All .DS_Store files deleted!")
            
print("Program completed")
