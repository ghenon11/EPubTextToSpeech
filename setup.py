
import os
import sys


def replace_strings_in_files(directory, replacements):

    for root, dirs, files in os.walk(directory):

        for file in files:

            if file.endswith(('.yml', '.ini')):

                file_path = os.path.join(root, file)
                print(f"Checking {file_path}")

                with open(file_path, 'r', encoding='utf-8') as f:

                    content = f.read()

                for search_string, replace_string in replacements.items():

                    if search_string in content:

                        content = content.replace(
                            search_string, replace_string)

                        print(
                            f'Replaced "{search_string}" with "{replace_string}" in {file_path}')

                with open(file_path, 'w', encoding='utf-8') as f:

                    f.write(content)

                with open(file_path, 'r', encoding='utf-8') as f:

                    content = f.read()

                for search_string, replace_string in replacements.items():
                    search_string = search_string.replace("\\", "\\\\")
                    replace_string = replace_string.replace("\\", "\\\\")

                    if search_string in content:

                        content = content.replace(
                            search_string, replace_string)

                        print(
                            f'Replaced "{search_string}" with "{replace_string}" in {file_path}')

                with open(file_path, 'w', encoding='utf-8') as f:

                    f.write(content)

# Define the directory to search and the strings to replace


directory_to_search = '.'
INSTALL_DIR = os.path.dirname(sys.executable)

replacements = {

    "D:\\Python\\Epubtextspeech\\Models": os.path.join(INSTALL_DIR, "Models"),
    "D:\Python\\Epubtextspeech\\Epub": os.path.join(INSTALL_DIR, "Epub"),
    "D:\\Python\\Epubtextspeech\\StyleTTS2": os.path.join(INSTALL_DIR, "StyleTTS2")

}

# Call the function
os.makedirs("logs", exist_ok=True)
replace_strings_in_files(directory_to_search, replacements)

print("\n\nPress any key to end...")
input()
