#  Personalized App Notification Scheduler using Multi-Armed Bandits

This project demonstrates how **Multi-Armed Bandit (MAB)** algorithms can optimize notification timing in applications. By simulating user behavior and dynamically learning from feedback, the system identifies the best time slots to send notifications for maximum engagement. The entire system is implemented as an interactive **Streamlit web app**.

---

## 🚀 Features

- ✔️ Epsilon-Greedy and UCB bandit algorithms  
- ✔️ Simulated users with behavioral patterns  
- ✔️ Real-time learning & arm value updates  
- ✔️ Reward, regret, and optimal-action visualizations  
- ✔️ Time-slot performance analytics  
- ✔️ Downloadable simulation data (CSV)  

---

## 🧠 How It Works

Each notification time window is treated as an **arm** in a multi-armed bandit:

| Time Slot                  | Arm |
|---------------------------|-----|
| Morning (8–11 AM)         | 0   |
| Afternoon (2–5 PM)        | 1   |
| Evening (7–10 PM)         | 2   |

During simulation:

1. Users receive time-based notifications.  
2. The algorithm selects a slot using either **Epsilon-Greedy** or **UCB**.  
3. If the user opens the notification → reward = 1, else 0.  
4. The bandit updates its reward estimates.  
5. Metrics (reward, regret, value estimates) are plotted live.

This replicates real-world personalization systems such as push notification optimizers.

---

## 🛠 Tech Stack

- **Streamlit** – Web interface  
- **Python** – Core logic  
- **NumPy** – Bandit computations  
- **Pandas** – Simulation data storage  
- **Matplotlib** – Visualizations  

---

## 📂 Project Structure
├── app.py 
├── requirements.txt 
└── README.md 

link : https://personalized--app--notification--scheduling-czgjaajapsv8ntdut7.streamlit.app/

