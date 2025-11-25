#Personalized App Notification Scheduler using Multi-Armed Bandits

This project demonstrates how Multi-Armed Bandit (MAB) algorithms can optimize notification timing in applications. By simulating user behavior and dynamically learning from feedback, the system identifies the best time slots to send notifications for maximum engagement. The entire system is implemented as an interactive Streamlit web app.

🚀 Features

✔️ Epsilon-Greedy and UCB bandit algorithms

✔️ Simulated users with behavioral patterns

✔️ Real-time learning & arm value updates

✔️ Reward, regret, and optimal-action visualizations

✔️ Time-slot performance analytics

✔️ Downloadable simulation data (CSV)

🧠 How It Works

Each notification time window is treated as an arm in a multi-armed bandit:

Time Slot	Arm
Morning (8–11 AM)	0
Afternoon (2–5 PM)	1
Evening (7–10 PM)	2

During the simulation:

Users receive time-slot–based notifications.

The algorithm selects a slot using either Epsilon-Greedy or UCB.

If the user opens the notification → reward = 1, else 0.

The bandit algorithm updates its estimates.

Metrics (reward, regret, estimated values) are visualized live.

This mimics real-world personalization systems such as push notification schedulers in mobile apps.

🛠 Tech Stack

Streamlit – Web UI

Python – Core logic

NumPy – Bandit computations

Pandas – Simulation data

Matplotlib – Plots and learning curves

📂 Project Structure
├── app.py               # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md            # Documentation

▶️ Run Locally
1️⃣ Clone the Repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

2️⃣ Create a Virtual Environment
python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Launch the App
streamlit run app.py

🌐 Deployment Notes (Important)

Streamlit Cloud uses Python 3.13, so you must use a compatible NumPy version.

✔️ Correct:
numpy>=1.26

❌ Wrong:
numpy==1.24.3  # This will fail because it requires distutils (removed in Python 3.12+)


If you see a distutils or NumPy build error during deploy, ensure your requirements.txt is updated.

📊 Key Visualizations

The app generates:

📈 Average Reward Over Time

📉 Cumulative Regret

🎯 Optimal Action Rate

📊 Notification Distribution

👥 User Pattern Performance

These provide insight into how effectively the bandit algorithm learns over time.

📥 Downloadable Results

Users can export the entire simulation log as a CSV file for external analysis.
