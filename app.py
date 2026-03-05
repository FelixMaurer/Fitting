import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- SECTION 1: WHAT IS FITTING? ---
st.header("1. What is Fitting?")
st.write("Fitting is essentially an algorithm trying to find the 'bottom of a valley'. "
         "Click the button below to watch how the line searches for the best slope by minimizing the errors (residues).")

# 1. Setup Data
true_slope = 2.2
x = np.arange(1, 11)
noise = np.array([1.2, -0.5, 2.1, -1.8, 0.5, 1.3, -0.9, 0.4, -1.1, 0.8])
y = true_slope * x + noise

# 2. Calculate the Error Surface
slopes_to_test = np.linspace(0, 4.5, 100)
sse_curve = np.array([np.sum((y - m * x)**2) for m in slopes_to_test])
best_slope = slopes_to_test[np.argmin(sse_curve)]

# 3. Create the "Overshoot" Path
t = np.linspace(0, 1, 60)
damp = np.exp(-4 * t)
oscillation = np.cos(2 * np.pi * t)
start_point = 0.5
path = best_slope - (best_slope - start_point) * damp * oscillation

# 4. Animation Trigger
if st.button("Run Fit Animation"):
    # Create an empty container to hold the frames
    plot_placeholder = st.empty() 
    
    # 5. Animation Loop
    for k in range(len(path)):
        current_m = path[k]
        current_y = current_m * x
        current_sse = np.sum((y - current_y)**2)
        
        # Status Logic
        if current_m > best_slope + 0.1:
            status_str = 'Status: OVERSHOOTING!'
            status_col = 'orange'
        elif k > 40:
            status_str = 'Status: SETTLING / CONVERGED'
            status_col = 'green'
        else:
            status_str = 'Status: SEARCHING...'
            status_col = 'black'

        # Initialize Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left Subplot: The Fit
        ax1.scatter(x, y, s=60, color='#3399CC', label='Data')
        ax1.plot(x, current_y, 'r', linewidth=2.5, label='Fitted Line')
        
        # Plot residual lines (vertical lines from data to fit)
        ax1.vlines(x, ymin=np.minimum(y, current_y), ymax=np.maximum(y, current_y), 
                   colors='#CC3333', linewidth=1, alpha=0.5)
        
        ax1.set_title('Visualizing the Fit')
        ax1.set_xlim(0, 11)
        ax1.set_ylim(min(y)-5, max(y)+5)
        ax1.grid(True)
        
        # Right Subplot: The Error Ball
        ax2.plot(slopes_to_test, sse_curve, 'k', linewidth=1.5)
        ax2.plot(current_m, current_sse, 'ro', markersize=10)
        ax2.text(0.2, max(sse_curve)*0.9, status_str, color=status_col, fontsize=12, fontweight='bold')
        
        ax2.set_title('Total Residue (The "Valley")')
        ax2.set_xlabel('Slope Parameter (m)')
        ax2.set_ylabel('Sum of Squared Residues')
        ax2.grid(True)
        
        # Overwrite the placeholder with the new frame
        plot_placeholder.pyplot(fig)
        
        # Close the figure to prevent memory leaks in the web app
        plt.close(fig) 
        
        # Pause to control animation speed
        time.sleep(0.08)
