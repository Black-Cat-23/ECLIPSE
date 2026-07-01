import os
import sys
import subprocess
import time
import threading

def run_backend():
    print("🚀 Installing/Verifying Backend Dependencies...")
    env = os.environ.copy()
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=os.path.dirname(__file__))
    
    print("🚀 Starting FastAPI Backend...")
    # Ensure database is initialized before serving
    subprocess.run([sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"], env=env)

def run_frontend():
    print("🚀 Starting Vite Frontend...")
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    
    print("📦 Installing frontend dependencies...")
    subprocess.run(["npm", "install"], cwd=frontend_dir, shell=True)
    
    subprocess.run(["npm", "run", "dev"], cwd=frontend_dir, shell=True)

if __name__ == "__main__":
    print("==================================================")
    print("      ECLIPSE-PRIME PLATFORM STARTUP SCRIPT       ")
    print("==================================================")
    
    # Start both in parallel threads so we see output from both
    backend_thread = threading.Thread(target=run_backend)
    frontend_thread = threading.Thread(target=run_frontend)
    
    backend_thread.daemon = True
    frontend_thread.daemon = True
    
    backend_thread.start()
    time.sleep(2)  # Give backend a slight head start
    frontend_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ECLIPSE Platform...")
        sys.exit(0)
