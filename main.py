"""
Main Entry Point for Hand Gesture Recognition System.
Runs the complete All-In-One Hand Gesture Recognition Engine.

Modes Included in this Single File:
  [0] Unified All-In-One HUD (Hand tracking, gesture classifier, finger counter, volume slider)
  [1] Hand Landmark Tracking & Mesh Inspector
  [2] Dedicated Finger Counter HUD (0-5)
  [3] Master Volume Controller (Thumb-Index pinch with pinky lock)
  [4] Rock-Paper-Scissors AI Battle (Live countdown, AI opponent, score tracker)
  [5] Snake Arcade Game (Index head control, donut food eating, tail collision)
  [6] Virtual Air Keyboard (Air hover & pinch typing)
  [7] Presentation Controller (Thumb/Pinky slide navigation & in-air whiteboard drawing)

Controls:
  - Press keys '0' through '7' on keyboard to switch modes.
  - Or click the top tab buttons directly with your mouse.
  - Or hover your index finger over a tab and raise your middle finger (Air Touch).
  - Press 'q' or ESC to exit.
"""

from HandTrackingModule import main

if __name__ == "__main__":
    main()
