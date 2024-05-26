import subprocess
import sys

def install_dependencies():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install dependencies.")

# def virtual_env_install():
#     try:
#         subprocess.run([sys.executable, '-m', 'virtualenv', 'venv'],check=True)
#         print("virtual environment is ready to be activated!")
        
#     except subprocess.CalledProcessError:
#         print("Problem occured in installing virtual environment")

# def activate_env():
#     subprocess.run([sys.executable, 'venv \ Scripts \ activate'])


if __name__ == '__main__':
    install_dependencies()
    # virtual_env_install()
    # activate_env()
    