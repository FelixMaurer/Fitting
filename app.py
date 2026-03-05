import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- SECTION 1: WHAT IS FITTING? ---
st.header("1. What is Fitting?")

# The new explanation paragraph
st.write("""
Why do we use fitting? It generally serves two essential purposes:
1. **Data Reduction:** It allows us to break down a large amount of correlating data into as few parameters as necessary, providing a clean, empirical summary of a system's behavior.
2. **Theoretical Modeling:** It acts as a bridge between observation and theory. By fitting data to a specific model, we can test underlying physical explanations and extract meaningful parameters for further theoretical investigation. For example, a careful fit can reveal that the fundamental physics of a system relies strictly on the ratio between parameters like M0 and M2 rather than their individual values, or it can help us isolate how changing a single physical force, such as repulsion, drives the entire observed system.
""")

st.write("Click the button below to watch a simplified simulation of an algorithm searching for the best parameter by minimizing the errors (residues).")

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
        
        # Plot residual lines
        ax1.vlines(x, ymin=np.minimum(y, current_y), ymax=np.maximum(y, current_y), 
                   colors='#CC3333', linewidth=1, alpha=0.5)
        
        ax1.set_title('Visualizing the Fit')
        ax1.set_xlim(0, 11)
        ax1.set_ylim(min(y)-5, max(y)+5)
        # Grid removed here
        
        # Right Subplot: The Error Ball
        ax2.plot(slopes_to_test, sse_curve, 'k', linewidth=1.5)
        ax2.plot(current_m, current_sse, 'ro', markersize=10)
        ax2.text(0.2, max(sse_curve)*0.9, status_str, color=status_col, fontsize=12, fontweight='bold')
        
        ax2.set_title('Total Residue (The "Valley")')
        ax2.set_xlabel('Slope Parameter (m)')
        ax2.set_ylabel('Sum of Squared Residues')
        # Grid removed here
        
        # Render and close
        plot_placeholder.pyplot(fig)
        plt.close(fig) 
        time.sleep(0.03)
# --- EXPLANATION OF THE ANIMATION ---
st.markdown("### What is happening here?")

st.write("""
**The Left Plot: The Residues** On the left, you see our data points (blue dots) and our model (the red line). The faint vertical lines connecting the dots to the red line are called **residues** (or residuals). A residue is simply the "mistake" the model makes for a specific data point—it is the vertical distance between the actual observed data and the theoretical line.

**The Right Plot: The Valley of Errors** If we take all those individual residues, square them (so negative and positive mistakes don't cancel each other out), and add them up, we get a single number representing the "Total Error." 

If you calculate this Total Error for every possible slope of the line, it forms a shape like a bowl or a valley, as seen in the right plot. 
""")



st.write("""
**The Algorithm and Converging** A fitting algorithm is essentially a blind explorer trying to find the lowest point in that valley. It starts with a guess (the red ball dropping in), checks the slope of the valley, and takes a step downwards. 

Sometimes, if it steps too far, it **overshoots** the lowest point and has to swing back. As it takes smaller and smaller steps, settling at the exact bottom where the total error is at its absolute minimum, we say the algorithm has **converged**. Finding this minimum is the mathematical goal of every standard fitting process.
""")
