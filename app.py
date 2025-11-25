import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import random
from typing import Dict, Tuple
import io

# ------------------------------
# Bandit classes
# ------------------------------
class EpsilonGreedyBandit:
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        # step-size for incremental update
        self.alpha = 0.1

    def select_arm(self) -> int:
        # explore
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        # exploit
        return int(np.argmax(self.values))

    def update(self, chosen_arm: int, reward: float):
        self.counts[chosen_arm] += 1
        current_value = self.values[chosen_arm]
        self.values[chosen_arm] = current_value + self.alpha * (reward - current_value)


class UCBBandit:
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select_arm(self) -> int:
        # play each arm once at the beginning
        if self.total_counts < self.n_arms:
            return self.total_counts

        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            bonus = self.c * np.sqrt(
                np.log(self.total_counts) / (self.counts[arm] + 1e-5)
            )
            ucb_values[arm] = self.values[arm] + bonus

        return int(np.argmax(ucb_values))

    def update(self, chosen_arm: int, reward: float):
        self.counts[chosen_arm] += 1
        self.total_counts += 1
        n = self.counts[chosen_arm]
        current_value = self.values[chosen_arm]
        # standard sample-average update
        self.values[chosen_arm] = ((n - 1) / n) * current_value + (1 / n) * reward


# ------------------------------
# Environment / User
# ------------------------------
class User:
    def __init__(self, user_id: int, behavior_pattern: str):
        self.user_id = user_id
        self.behavior_pattern = behavior_pattern

        # base probabilities per pattern
        if behavior_pattern == "morning":
            self.true_probabilities = [0.8, 0.3, 0.2]
        elif behavior_pattern == "afternoon":
            self.true_probabilities = [0.3, 0.8, 0.4]
        elif behavior_pattern == "evening":
            self.true_probabilities = [0.2, 0.4, 0.8]
        else:  # random
            self.true_probabilities = [0.5, 0.5, 0.5]

        # add a little noise and clamp to [0.1, 0.9]
        self.true_probabilities = [
            max(0.1, min(0.9, p + random.uniform(-0.1, 0.1)))
            for p in self.true_probabilities
        ]

    def will_open_notification(self, time_slot: int) -> bool:
        probability = self.true_probabilities[time_slot]
        return np.random.random() < probability


class NotificationScheduler:
    def __init__(self, algorithm: str = "epsilon_greedy", **kwargs):
        self.time_slots = [
            "Morning (8-11 AM)",
            "Afternoon (2-5 PM)",
            "Evening (7-10 PM)",
        ]
        self.n_arms = len(self.time_slots)

        if algorithm == "epsilon_greedy":
            self.bandit = EpsilonGreedyBandit(
                self.n_arms, kwargs.get("epsilon", 0.1)
            )
        elif algorithm == "ucb":
            self.bandit = UCBBandit(self.n_arms, kwargs.get("c", 2.0))
        else:
            raise ValueError("Algorithm must be 'epsilon_greedy' or 'ucb'")

        self.history = []
        self.algorithm_name = algorithm

    def send_notification(self, user: User) -> Tuple[int, bool]:
        time_slot = self.bandit.select_arm()
        opened = user.will_open_notification(time_slot)
        reward = 1.0 if opened else 0.0

        self.bandit.update(time_slot, reward)

        self.history.append(
            {
                "user_id": user.user_id,
                "time_slot": time_slot,
                "time_slot_name": self.time_slots[time_slot],
                "opened": opened,
                "reward": reward,
                "user_pattern": user.behavior_pattern,
                "timestamp": datetime.now(),
            }
        )

        return time_slot, opened

    def get_statistics(self) -> Dict:
        if not self.history:
            return {}

        df = pd.DataFrame(self.history)
        stats = {
            "total_notifications": len(self.history),
            "total_opens": int(df["opened"].sum()),
            "overall_open_rate": float(df["opened"].mean()),
            "algorithm": self.algorithm_name,
        }

        for i, slot_name in enumerate(self.time_slots):
            slot_data = df[df["time_slot"] == i]
            if len(slot_data) > 0:
                stats[f"{slot_name}_notifications"] = len(slot_data)
                stats[f"{slot_name}_opens"] = int(slot_data["opened"].sum())
                stats[f"{slot_name}_open_rate"] = float(slot_data["opened"].mean())
                stats[f"{slot_name}_estimated_value"] = float(self.bandit.values[i])

        return stats


# ------------------------------
# Simulation helper
# ------------------------------
def run_simulation(num_users, notifications_per_user, algorithm, **kwargs):
    patterns = ["morning", "afternoon", "evening", "random"]
    user_patterns = [random.choice(patterns) for _ in range(num_users)]
    users = [User(i, pattern) for i, pattern in enumerate(user_patterns)]

    scheduler = NotificationScheduler(algorithm=algorithm, **kwargs)

    cumulative_rewards = []
    cumulative_regrets = []
    optimal_actions = []

    for user in users:
        user_true_probs = user.true_probabilities
        optimal_arm = int(np.argmax(user_true_probs))
        optimal_prob = user_true_probs[optimal_arm]

        for _ in range(notifications_per_user):
            chosen_arm, opened = scheduler.send_notification(user)
            chosen_prob = user_true_probs[chosen_arm]
            regret = optimal_prob - chosen_prob

            cumulative_rewards.append(scheduler.bandit.values.mean())
            cumulative_regrets.append(regret)
            optimal_actions.append(1 if chosen_arm == optimal_arm else 0)

    return scheduler, cumulative_rewards, cumulative_regrets, optimal_actions


# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(
    page_title="Notification Scheduler",
    page_icon="🔔",
    layout="wide",
)


def main():
    st.title("🔔 Personalized App Notification Scheduler")
    st.markdown(
        """
    This web application demonstrates how **Multi-Armed Bandit** algorithms can optimize 
    notification timing to maximize user engagement.
    """
    )

    # Sidebar configuration
    st.sidebar.header("Configuration")

    algorithm = st.sidebar.selectbox(
        "Select Bandit Algorithm",
        ["epsilon_greedy", "ucb"],
        format_func=lambda x: "Epsilon-Greedy" if x == "epsilon_greedy" else "UCB",
    )

    if algorithm == "epsilon_greedy":
        epsilon = st.sidebar.slider(
            "Epsilon (Exploration rate)", 0.01, 0.5, 0.1, 0.01
        )
        params = {"epsilon": epsilon}
    else:
        c = st.sidebar.slider("C (Confidence parameter)", 0.5, 5.0, 2.0, 0.1)
        params = {"c": c}

    num_users = st.sidebar.slider("Number of Users", 10, 200, 50)
    notifications_per_user = st.sidebar.slider(
        "Notifications per User", 10, 100, 20
    )

    # Two main columns: left = simulation, right = About
    main_col, about_col = st.columns([2.5, 1])

    # ---------- RIGHT: About panel, always visible ----------
    with about_col:
        st.subheader("ℹ️ About This App")
        st.markdown(
            """
        **How it works (RL view):**
        - Each time slot is an **arm** of a multi-armed bandit  
        - The agent learns which time maximizes the open probability  
        - It balances **exploration** (trying all slots) and **exploitation** (using the best slot)

        **Time Slots:**
        - 🕗 Morning (8–11 AM)  
        - 🕑 Afternoon (2–5 PM)  
        - 🕗 Evening (7–10 PM)  

        **User Patterns (environment types):**
        - 🌅 Morning-active  
        - ☀️ Afternoon-active  
        - 🌙 Evening-active  
        - 🎲 Random  
        """
        )

        st.subheader("⚙️ Current Settings")
        st.write(
            f"Algorithm: {'Epsilon-Greedy' if algorithm == 'epsilon_greedy' else 'UCB'}"
        )
        if algorithm == "epsilon_greedy":
            st.write(f"Epsilon: {epsilon}")
        else:
            st.write(f"C parameter: {c}")
        st.write(f"Users: {num_users}")
        st.write(f"Notifications per user: {notifications_per_user}")

    # ---------- LEFT: main simulation area ----------
    with main_col:
        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                scheduler, rewards, regrets, optimal_actions = run_simulation(
                    num_users, notifications_per_user, algorithm, **params
                )

                st.subheader("📊 Simulation Results")

                stats = scheduler.get_statistics()
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Total Notifications", stats["total_notifications"])
                with m2:
                    st.metric("Total Opens", stats["total_opens"])
                with m3:
                    st.metric(
                        "Overall Open Rate",
                        f"{stats['overall_open_rate']:.3f}",
                    )

                # Time slot performance table
                st.subheader("⏰ Time Slot Performance")
                slot_data = []
                for slot in scheduler.time_slots:
                    key = f"{slot}_notifications"
                    if key in stats:
                        slot_data.append(
                            {
                                "Time Slot": slot,
                                "Notifications": stats[key],
                                "Open Rate": stats[f"{slot}_open_rate"],
                                "Estimated Value": stats[f"{slot}_estimated_value"],
                            }
                        )

                if slot_data:
                    slot_df = pd.DataFrame(slot_data)
                    st.dataframe(
                        slot_df.style.format(
                            {"Open Rate": "{:.3f}", "Estimated Value": "{:.3f}"}
                        )
                    )

                # Learning curves
                st.subheader("📈 Learning Progress")

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Average reward over time
                axes[0, 0].plot(rewards)
                axes[0, 0].set_title("Average Reward Over Time")
                axes[0, 0].set_xlabel("Notification Number")
                axes[0, 0].set_ylabel("Average Reward")
                axes[0, 0].grid(True, alpha=0.3)

                # Cumulative regret
                cumulative_regret = np.cumsum(regrets)
                axes[0, 1].plot(cumulative_regret)
                axes[0, 1].set_title("Cumulative Regret Over Time")
                axes[0, 1].set_xlabel("Notification Number")
                axes[0, 1].set_ylabel("Cumulative Regret")
                axes[0, 1].grid(True, alpha=0.3)

                # Optimal action rate
                optimal_rate = np.cumsum(optimal_actions) / (
                    np.arange(len(optimal_actions)) + 1
                )
                axes[1, 0].plot(optimal_rate)
                axes[1, 0].set_title("Optimal Action Rate Over Time")
                axes[1, 0].set_xlabel("Notification Number")
                axes[1, 0].set_ylabel("Optimal Action Rate")
                axes[1, 0].grid(True, alpha=0.3)

                # Time slot distribution
                df = pd.DataFrame(scheduler.history)
                slot_dist = df["time_slot_name"].value_counts().sort_index()
                axes[1, 1].bar(slot_dist.index, slot_dist.values)
                axes[1, 1].set_title("Notification Distribution by Time Slot")
                axes[1, 1].set_ylabel("Number of Notifications")
                axes[1, 1].tick_params(axis="x", rotation=45)

                plt.tight_layout()
                st.pyplot(fig)

                # User pattern analysis
                st.subheader("👥 User Pattern Analysis")
                pattern_performance = (
                    df.groupby("user_pattern")["opened"]
                    .mean()
                    .sort_values(ascending=False)
                )
                st.bar_chart(pattern_performance)

                # Download results
                st.subheader("💾 Download Results")
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Simulation Data as CSV",
                    data=csv,
                    file_name="notification_simulation.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
