import subprocess, sys, os

python = sys.executable
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genesis_watcher.py")
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "watcher.log")

# Start watcher as detached process on Windows
if sys.platform == "win32":
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    DETACHED_PROCESS = 0x00000008
    proc = subprocess.Popen(
        [python, script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=open(log_file, "w", buffering=1),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    )
else:
    proc = subprocess.Popen(
        [python, script],
        stdout=open(log_file, "w", buffering=1),
        stderr=subprocess.STDOUT,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        start_new_session=True,
    )

print(f"Watcher started: PID {proc.pid}")
print(f"Log: {log_file}")
