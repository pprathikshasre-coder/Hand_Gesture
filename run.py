import sys
import os
import subprocess

SCRIPT_MAP = {
    "1": ("Interactive Gesture Menu", "MenuScreen.py"),
    "2": ("Hand Landmark Tracking", "Sample.py"),
    "3": ("Finger Counter", "FingerCounter.py"),
    "4": ("Volume Control", "VolumeControl.py"),
    "5": ("Rock Paper Scissors Game", "RockPaperScissorsGame.py"),
    "6": ("Snake Game", "SnakeGame.py"),
    "7": ("Virtual Keyboard", "VirtualKeyboard.py"),
    "8": ("Presentation Control", "PresentationControl.py"),
    "9": ("Basic MediaPipe Demo", "Basic.py"),
    "10": ("Desktop GUI Dashboard", "GUI.py"),
}

ALIAS_MAP = {
    "web": "web_app.py",
    "browser": "web_app.py",
    "ui": "SuperUI.py",
    "superui": "SuperUI.py",
    "menu": "MenuScreen.py",
    "track": "Sample.py",
    "tracking": "Sample.py",
    "sample": "Sample.py",
    "fingers": "FingerCounter.py",
    "counter": "FingerCounter.py",
    "volume": "VolumeControl.py",
    "vol": "VolumeControl.py",
    "rps": "RockPaperScissorsGame.py",
    "game": "RockPaperScissorsGame.py",
    "snake": "SnakeGame.py",
    "keyboard": "VirtualKeyboard.py",
    "presentation": "PresentationControl.py",
    "slides": "PresentationControl.py",
    "basic": "Basic.py",
    "gui": "GUI.py",
}


def print_banner():
    print("=" * 65)
    print("        HAND GESTURE RECOGNITION - MAIN LAUNCHER")
    print("=" * 65)
    print("Select a module to run (press 'q' inside any window to close it):")
    for key, (name, filename) in SCRIPT_MAP.items():
        print(f"  [{key:>2}] {name:<30} ({filename})")
    print("  [ q] Exit")
    print("-" * 65)


def run_script(script_path):
    print(f"\n>>> Running: {script_path} ...")
    try:
        subprocess.run([sys.executable, script_path])
    except KeyboardInterrupt:
        print("\n[!] Execution stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] Failed to run {script_path}: {e}")


def cli_loop():
    while True:
        print_banner()
        choice = input("Enter option [1-10 or q]: ").strip().lower()
        if choice in ['q', 'exit', 'quit']:
            print("Exiting. Goodbye!")
            break
        if choice in SCRIPT_MAP:
            run_script(SCRIPT_MAP[choice][1])
        elif choice in ALIAS_MAP:
            run_script(ALIAS_MAP[choice])
        else:
            print("[!] Invalid option. Please choose a number between 1 and 10 or 'q'.\n")


def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower().strip("-")
        if arg in ALIAS_MAP:
            run_script(ALIAS_MAP[arg])
            return
        if arg in SCRIPT_MAP:
            run_script(SCRIPT_MAP[arg][1])
            return
        if arg in ["cli", "console", "terminal"]:
            cli_loop()
            return
        if arg in ["help", "h"]:
            print("Usage:")
            print("  python run.py              # Opens the GUI Dashboard")
            print("  python run.py cli          # Opens interactive CLI menu")
            print("  python run.py <module>     # Runs specific module directly")
            print("Available module aliases:", ", ".join(ALIAS_MAP.keys()))
            return

    # By default, open SuperUI Dashboard if available, else fallback to CLI
    try:
        run_script("SuperUI.py")
    except Exception:
        cli_loop()


if __name__ == "__main__":
    main()
