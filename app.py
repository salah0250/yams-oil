# app.py

from flask import Flask, request, jsonify
import numpy as np
from stable_baselines3 import PPO

# Charger les modèles
model_dice = PPO.load("models/A2C_yahtzee_dice_v13")
model_category = PPO.load("models/ppo_yahtzee_categories_v_dice_A_10")

app = Flask(__name__)

categories = [
    "Ones", "Twos", "Threes", "Fours", "Fives", "Sixes",
    "Three of a Kind", "Four of a Kind", "Full House",
    "Small Straight", "Large Straight", "Yahtzee", "Chance"
]

def roll_dice(current_dice, keep):
    return [current_dice[i] if keep[i] == 1 else np.random.randint(1, 7) for i in range(5)]

def calculate_score(dice, category_index):
    counts = np.bincount(dice, minlength=7)[1:]
    if category_index == 0:
        return counts[0] * 1
    elif category_index == 1:
        return counts[1] * 2
    elif category_index == 2:
        return counts[2] * 3
    elif category_index == 3:
        return counts[3] * 4
    elif category_index == 4:
        return counts[4] * 5
    elif category_index == 5:
        return counts[5] * 6
    elif category_index == 6:
        return sum(dice) if np.any(counts >= 3) else 0
    elif category_index == 7:
        return sum(dice) if np.any(counts >= 4) else 0
    elif category_index == 8:
        return 25 if np.any(counts == 3) and np.any(counts == 2) else 0
    elif category_index == 9:
        return 30 if is_small_straight(dice) else 0
    elif category_index == 10:
        return 40 if is_large_straight(dice) else 0
    elif category_index == 11:
        return 50 if np.any(counts == 5) else 0
    elif category_index == 12:
        return sum(dice)
    return 0

def is_small_straight(dice):
    unique_dice = np.unique(dice)
    straights = [set([1, 2, 3, 4]), set([2, 3, 4, 5]), set([3, 4, 5, 6])]
    return any(straight.issubset(unique_dice) for straight in straights)

def is_large_straight(dice):
    unique_dice = np.unique(dice)
    straights = [set([1, 2, 3, 4, 5]), set([2, 3, 4, 5, 6])]
    return any(straight.issubset(unique_dice) for straight in straights)

# Définir vos routes Flask ici
@app.route('/api/roll_dice', methods=['POST'])
def api_roll_dice():
    data = request.json
    current_dice = data['current_dice']
    keep = data['keep']
    new_dice = roll_dice(current_dice, keep)
    return jsonify(new_dice)

@app.route('/api/calculate_score', methods=['POST'])
def api_calculate_score():
    data = request.json
    dice = data['dice']
    category_index = data['category_index']
    score = calculate_score(dice, category_index)
    return jsonify({'score': score})

@app.route('/api/ai_turn', methods=['POST'])
def api_ai_turn():
    data = request.json
    used_categories = data['used_categories']
    current_score = data['current_score']
    
    dice = roll_dice([0]*5, [0]*5)
    roll_count = 2
    for _ in range(2):
        obs = {'dice': np.array(dice) - 1, 'roll_count': np.array([roll_count])}
        action, _ = model_dice.predict(obs)
        action = np.ravel(action).tolist()
        action = [int(x) for x in action]
        dice = roll_dice(dice, action)
        roll_count -= 1

    final_dice = dice
    obs = {'dice': np.array(final_dice) - 1, 'used_categories': used_categories , 'current_score': np.array([current_score])}
    action, _ = model_category.predict(obs)
    action = np.ravel(action).tolist()
    action = action[0] if isinstance(action, list) else action
    
    while used_categories[action]:
        action = np.random.randint(0, 13)
        
    score = calculate_score(final_dice, action)
    used_categories[action] = 1
    return jsonify({'action': action, 'score': score, 'dice': final_dice.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
