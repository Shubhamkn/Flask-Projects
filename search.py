import os

# Directory to search for packages
site_packages_dir = os.path.join('venv', 'Lib', 'site-packages')

# Function to search for 'getargspec' in all .py files
def search_getargspec(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        content = f.read()
                        if 'inspect.getargspec' in content:
                            print(f"Found in {file_path}")
                    except:
                        continue

search_getargspec(site_packages_dir)
