# **Bluff-ToM: Exploring Theory of Mind in the Game of Bluff**

## **Project Overview**
This project is part of a Hybrid Intelligence course focused on implementing and analyzing theory of mind (ToM) agents. The objective is to study how different levels of ToM reasoning affect gameplay strategies and outcomes in the card game *Bluff*. By simulating agents with varying levels of ToM (zero-order, first-order, and second-order), we aim to evaluate their performance under different experimental setups.

The project involves:
- Implementing the game of *Bluff*.
- Creating agents with varying levels of ToM reasoning.
- Running simulations to compare agent performances in different experimental setups.

---

## **Rules of the Game: Bluff**

Bluff is a card-based bluffing game played with the face cards of a standard deck. The rules are as follows:

### **Setup**
- The game is played with four copies of each face card: Aces, Jacks, Queens, and Kings.
- Cards are distributed evenly among players, and excess cards are placed face-down on a central pile.

### **Gameplay**
1. The game starts with the rank *Aces*. The first player selects any number of their cards, plays them face-down on the pile, and implicitly claims that all are *Aces*.
2. Subsequent players take turns playing cards on the pile, with the claim that all cards match the current rank (e.g., *Aces* initially, then *Jacks*, etc.).

### **Challenges**
- A player may challenge the most recent play by revealing the cards. If the claim is incorrect (e.g., not all cards are the claimed rank), the previous player takes all cards in the pile. If the claim is correct, the challenger takes the pile.
- The game then continues with the next rank in sequence (e.g., from *Aces* to *Jacks*).

### **Winning Condition**
- The first player to run out of cards wins.

### **Variants**
- **Wild Bluff**: Adds Joker cards as wildcards, which can represent any face card.
- **Doubt It**: Players can declare any rank for the round and may choose to pass their turn. The game ends immediately after a successful or failed challenge, with the winner determined based on the challenge outcome.

---

## **Experimental Setups**

To study the role of theory of mind in gameplay, the following experimental setups will be explored:

1. **Explicit Memory Zero-Order Theory of Mind Agents**  
   Zero-order agents are extended with explicit memory to track actions and outcomes from previous rounds. This additional memory enhances decision-making by incorporating historical data into their strategies.

2. **Three-Player Variant**  
   The game is modified to include three players instead of two. Simulations will explore the interaction between different combinations of zero-order and first-order agents in a multi-agent environment.

3. **Second-Order Theory of Mind Agents**  
   Second-order agents are implemented to reason about what other agents believe about them. This adds an additional layer of strategic depth compared to first-order agents.

4. **Human-Agent Interaction**  
   Human players compete against zero-order and first-order agents. This setup collects data on how ToM agents perform against real-world strategies.

---
```plaintext
bluff_game/
├── agents/
│   ├── __init__.py
│   ├── agent.py                  # Generic Agent class
├── strategies/
│   ├── __init__.py
│   ├── strategy.py               # Abstract Strategy interface
│   ├── zero_order_strategy.py    # Zero-Order Strategy implementation
│   ├── first_order_strategy.py   # First-Order Strategy implementation
│   ├── second_order_strategy.py  # Second-Order Strategy implementation
│   ├── human_strategy.py         # Human Strategy implementation
├── core/
│   ├── __init__.py
│   ├── game_manager.py           # GameManager class and game logic
│   ├── deck.py                   # Deck class for managing cards
│   ├── hand.py                   # Hand class for managing player hands
│   ├── card.py                   # Card class for individual cards
├── main.py                       # Entry point to run the game
├── README.md                     # Project overview and instructions
├── requirements.txt              # Python dependencies


