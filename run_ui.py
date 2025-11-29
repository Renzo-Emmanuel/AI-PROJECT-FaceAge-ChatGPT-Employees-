#!/usr/bin/env python3
"""
Gender & Age Detection - Main Launcher
Run this file to start the desktop UI application
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Change to src directory for relative paths to work
os.chdir(os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the UI
from ui import GenderAgeDetectorUI
import tkinter as tk

if __name__ == "__main__":
    print("Starting Gender & Age Detection UI...")
    root = tk.Tk()
    app = GenderAgeDetectorUI(root)
    root.mainloop()