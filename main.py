import tennisim as ts
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Define the Markov chain model for a tennis match
def markov_chain_model(p1_first_serve, p1_second_serve, p1_break_point, p2_first_serve, p2_second_serve, p2_break_point, num_iterations=1000):
    # States: 0 - Player 1 wins, 1 - Player 2 wins
    transition_matrix = np.array([
        [p1_first_serve * p1_break_point, 1 - p1_first_serve * p1_break_point],
        [1 - p2_first_serve * p2_break_point, p2_first_serve * p2_break_point]
    ])
    
    # Initial state probabilities
    initial_state = np.array([0.5, 0.5])
    
    # Simulate the Markov chain
    state = initial_state
    for _ in range(num_iterations):
        state = np.dot(state, transition_matrix)
    
    return state

# Function to run multiple simulations
def run_simulations(p1_first_serve, p1_second_serve, p1_break_point, p2_first_serve, p2_second_serve, p2_break_point, num_simulations=1500):
    results = []
    for _ in range(num_simulations):
        probabilities = markov_chain_model(p1_first_serve, p1_second_serve, p1_break_point, p2_first_serve, p2_second_serve, p2_break_point)
        results.append(probabilities)
    results = np.array(results)
    mean_probabilities = np.mean(results, axis=0)
    wins = np.sum(results, axis=0)
    return mean_probabilities, wins

# Function to plot the results and save as an image
def plot_results(player1, player2, mean_probabilities, wins, num_simulations):
    labels = [player1, player2]
    fig, ax = plt.subplots()
    bars = ax.bar(labels, mean_probabilities, color=['#007bff', '#ff5733'])
    ax.get_yaxis().set_visible(False)  # Hide the Y-axis
    
    # Label the bars with the number of wins and percentage
    for bar, win, prob in zip(bars, wins, mean_probabilities):
        height = bar.get_height()
        ax.annotate(f'{win:.0f} wins\n({prob:.2%})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    st.pyplot(fig)
    
    # Save the plot as an image
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Main function
def main():
    load_css("styles.css")
    st.title("mSticky Tennis Sim")
    
    st.write("""
    This app simulates the probability of winning a tennis match between two players using a Markov chain model. 
    Input the players' statistics to get the probability of each player winning the match.
    """)
    
    # User input for player names and detailed statistics
    player1 = st.text_input("Enter the name of Player 1:", "Player 1")
    player2 = st.text_input("Enter the name of Player 2:", "Player 2")
    
    # Select match type
    match_type = st.selectbox("Select Match Type", ["Men's", "Women's"])
    
    # Set default probabilities based on match type
    if match_type == "Men's":
        p1_first_serve_default = 0.65
        p1_second_serve_default = 0.50
        p1_break_point_default = 0.40
        p2_first_serve_default = 0.60
        p2_second_serve_default = 0.55
        p2_break_point_default = 0.45
    else:
        p1_first_serve_default = 0.60
        p1_second_serve_default = 0.45
        p1_break_point_default = 0.35
        p2_first_serve_default = 0.55
        p2_second_serve_default = 0.50
        p2_break_point_default = 0.40
    
    p1_first_serve = st.number_input(f"Enter {player1}'s first serve win percentage:", 0.0, 1.0, p1_first_serve_default)
    p1_second_serve = st.number_input(f"Enter {player1}'s second serve win percentage:", 0.0, 1.0, p1_second_serve_default)
    p1_break_point = st.number_input(f"Enter {player1}'s break point conversion rate:", 0.0, 1.0, p1_break_point_default)
    p2_first_serve = st.number_input(f"Enter {player2}'s first serve win percentage:", 0.0, 1.0, p2_first_serve_default)
    p2_second_serve = st.number_input(f"Enter {player2}'s second serve win percentage:", 0.0, 1.0, p2_second_serve_default)
    p2_break_point = st.number_input(f"Enter {player2}'s break point conversion rate:", 0.0, 1.0, p2_break_point_default)
    
    # Button for running simulations
    if st.button("Run Simulation"):
        num_simulations = 1500
        
        # Run simulations
        mean_probabilities, wins = run_simulations(p1_first_serve, p1_second_serve, p1_break_point, p2_first_serve, p2_second_serve, p2_break_point, num_simulations)
        
        # Output the probabilities
        st.write(f"Probability of {player1} winning: {mean_probabilities[0]:.2%}")
        st.write(f"Probability of {player2} winning: {mean_probabilities[1]:.2%}")
        
        # Plot the results and get the image buffer
        buf = plot_results(player1, player2, mean_probabilities, wins, num_simulations)
        
        # Provide a download button for the plot image
        st.download_button(
            label="Download Plot",
            data=buf,
            file_name="tennis_match_probability.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()