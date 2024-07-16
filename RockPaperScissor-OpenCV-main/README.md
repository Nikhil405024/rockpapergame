# RockPaperScissor-OpenCV

This is a simple rock paper scissor game, made using python, OpenCV and mediapipe for hand-tracking.

This contains a HandTrackingModule which makes working with hand-tracking easy, it has methods - findHands to track hands, findPosition to get coordinates i.e. hand landmarks for all 21 points that mediapipe offers for a hand, and a fingersUp method that returns how many fingers are currently open. 

The fingersUp method is used to know whether the player has played rock, paper or scissor - rock for 0 fingers open, scissor for 2 fingers open and paper for all five fingers open.

For accurately showcasing your move, don't wait for the timer to hit 0 to play your move, be ready with your move in the red area before the timer hits 0.

There is no end condition for the game, you can play it for as long as you want.

