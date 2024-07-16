import random
import cv2
import HandTrackingModule as htm
import numpy as np
import time
MOVES = ['rock', 'paper', 'scissor']
NUM_ACTIONS = len(MOVES)
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Epsilon for epsilon-greedy policy

# Initialize Q-table
q_table = np.zeros((NUM_ACTIONS, NUM_ACTIONS))

def choose_action(state):
    if random.uniform(0, 1) < EPSILON:
        return random.randint(0, NUM_ACTIONS - 1)  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

def update_q_table(state, action, reward, next_state):
    best_next_action = np.argmax(q_table[next_state])
    q_table[state, action] += ALPHA * (reward + GAMMA * q_table[next_state, best_next_action] - q_table[state, action])

def get_state(player_move, comp_move):
    return MOVES.index(player_move), MOVES.index(comp_move)
def checkWinner(player, comp):
    if player == comp:
        return None
    elif player == 'rock' and comp == 'scissor':
        return 0
    elif player == 'paper' and comp == 'rock':
        return 0
    elif player == 'scissor' and comp == 'paper':
        return 0
    else:
        return 1

# VARIABLES
waitTime = 4
moves = ['rock', 'paper', 'scissor']
scores = [0, 0]  # [player, comp]
comp, player = None, None
wCam, hCam = 1280, 720

# Get feed
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Get detector
detector = htm.handDetector(detectionCon=0.8, maxHands=1)

# Time variables for time limit
prevTime = time.time()
newTime = time.time()

# Colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)

# Start screen interface
start_screen = True
while start_screen:
    success, start_img = cap.read()
    start_img = cv2.flip(start_img, 1)
    cv2.putText(start_img, 'Press "S" to Start the Game', (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Start Screen', start_img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        start_screen = False

cv2.destroyWindow('Start Screen')

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Drawing centre line, area for input, and scores
    cv2.line(img, (wCam // 2, 0), (wCam // 2, hCam), GREEN, 5)
    cv2.rectangle(img, (780, 260), (1180, 660), RED, 2)
    cv2.putText(img, f'Your Score: {scores[0]}', (wCam - 400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN, 2)
    cv2.putText(img, f'Computer Score: {scores[1]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

    img = detector.findHands(img)

    lmList = detector.findPosition(img, draw=False)

    # Handling time limit
    if waitTime - int(newTime) + int(prevTime) < 0:
        cv2.putText(img, '0', (960, 160), cv2.FONT_HERSHEY_PLAIN, 7, RED, 3)
    else:
        cv2.putText(img, f'{waitTime - int(newTime) + int(prevTime)}', (960, 160), cv2.FONT_HERSHEY_PLAIN, 7, RED, 3)

    # Hand landmarks obtained, next
    if len(lmList) != 0:

        if newTime - prevTime >= waitTime:

            x, y = lmList[0][1:]

            if 780 < x < 1180 and 260 < y < 660:
                fingers = detector.fingersUp()
                totalFingers = fingers.count(1)

                # Game logic
                if totalFingers == 0:
                    player = 'rock'
                elif totalFingers == 2:
                    player = 'scissor'
                elif totalFingers == 5:
                    player = 'paper'

                comp = moves[random.randint(0, 2)]

                winner = checkWinner(player, comp)
                if winner is not None:
                    scores[winner] = scores[winner] + 1
                    if scores[1] >= 3:
                        cv2.putText(img, 'Computer Win!', (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, RED, 5)
                        cv2.putText(img, 'Better Luck Next Time', (350, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, RED, 2)
                        cv2.imshow('Image', img)
                        cv2.waitKey(3000)
                        break
                    elif scores[0] >= 3:
                        cv2.putText(img, 'You Wins!', (400, 300), cv2.FONT_HERSHEY_SIMPLEX, 3, GREEN, 5)
                        cv2.putText(img, 'Congratulations!!', (300, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, GREEN, 2)
                        cv2.imshow('Image', img)
                        cv2.waitKey(3000)
                        break

                prevTime = time.time()

    # Show computer move
    if comp:
        comp_img = cv2.imread(f'Fingers/{comp}.jpg')
        comp_img = cv2.resize(comp_img, (400, 400))
        img[160:560, 20:420] = comp_img

    newTime = time.time()

    cv2.imshow('Image', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()
cap.release()