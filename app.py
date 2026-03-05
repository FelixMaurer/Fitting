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
# --- SECTION 2: THE TWO MINIMUM PROBLEM ---
st.markdown("---")
st.header("2. The 'Two Minimum' Problem")

st.write("When we fit more than one parameter, the 'valley' of errors becomes a 3D landscape. "
         "Depending on where your algorithm starts, it might roll into the wrong valley. "
         "Try adjusting the starting values below and run the fit to see if you get trapped!")

# User Inputs for Starting Values
col1, col2 = st.columns(2)
with col1:
    user_p1 = st.slider("Start position for Narrow Peak (p1)", 1.0, 9.0, 7.5, step=0.5)
with col2:
    user_p2 = st.slider("Start position for Wide Peak (p2)", 1.0, 9.0, 2.0, step=0.5)

if st.button("Run 2D Fit Animation"):
    plot_placeholder_2 = st.empty()
    
    # 1. Setup Synthetic Data
    x = np.linspace(0, 10, 200)
    y_data = 1.0 * np.exp(-((x - 3) / 0.4)**2) + 0.8 * np.exp(-((x - 7) / 1.5)**2)
    
    # 2. Define Model and Error Function
    def model_func(p):
        return 1.0 * np.exp(-((x - p[0]) / 0.4)**2) + 0.8 * np.exp(-((x - p[1]) / 1.5)**2)
        
    def sse_func(p):
        return np.sum((y_data - model_func(p))**2)
        
    # 3. Pre-calculate Surface for Landscape
    mu_range = np.linspace(1, 9, 40)
    M1, M2 = np.meshgrid(mu_range, mu_range)
    Z = np.zeros_like(M1)
    for i in range(M1.shape[0]):
        for j in range(M1.shape[1]):
            Z[i, j] = np.log10(sse_func([M1[i, j], M2[i, j]]))
            
    # 4. Optimization Parameters
    lr = 0.002
    n_steps = 80
    eps = 1e-4
    
    curr_user = np.array([user_p1, user_p2])
    curr_good = np.array([2.0, 8.0]) # The known good starting point
    
    path_user = [curr_user.copy()]
    path_good = [curr_good.copy()]
    
    # 5. Animation Loop
    for t in range(n_steps):
        # Calculate gradients (Numeric derivative)
        v_user = sse_func(curr_user)
        g_user = np.array([
            (sse_func(curr_user + [eps, 0]) - v_user) / eps,
            (sse_func(curr_user + [0, eps]) - v_user) / eps
        ])
        curr_user = curr_user - lr * g_user
        path_user.append(curr_user.copy())
        
        v_good = sse_func(curr_good)
        g_good = np.array([
            (sse_func(curr_good + [eps, 0]) - v_good) / eps,
            (sse_func(curr_good + [0, eps]) - v_good) / eps
        ])
        curr_good = curr_good - lr * g_good
        path_good.append(curr_good.copy())
        
        # Draw every 3rd frame to speed up web rendering
        if t % 3 == 0 or t == n_steps - 1:
            fig = plt.figure(figsize=(14, 6))
            
            # LEFT: The Model Fit
            ax1 = fig.add_subplot(121)
            ax1.plot(x, y_data, 'k.', markersize=8, label='Experimental Data')
            ax1.plot(x, model_func(curr_user), 'r-', linewidth=2.5, label='User Path Fit (Red)')
            ax1.plot(x, model_func(curr_good), 'g-', linewidth=2.5, alpha=0.6, label='Global Path Fit (Green)')
            ax1.set_title('Current Fit to Data')
            ax1.set_ylim(-0.2, 1.5)
            ax1.legend(loc='upper right')
            
            # RIGHT: The Parameter Space (3D)
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_surface(M1, M2, Z, cmap='viridis', alpha=0.6, edgecolor='none')
            
            # Plot the paths
            p_user_arr = np.array(path_user)
            p_good_arr = np.array(path_good)
            
            z_user = [np.log10(sse_func(p)) for p in p_user_arr]
            z_good = [np.log10(sse_func(p)) for p in p_good_arr]
            
            ax2.plot(p_user_arr[:,0], p_user_arr[:,1], z_user, 'r.-', linewidth=2)
            ax2.plot(p_good_arr[:,0], p_good_arr[:,1], z_good, 'g.-', linewidth=2)
            
            # Plot current position markers
            ax2.scatter(*curr_user, np.log10(sse_func(curr_user)), color='red', s=100)
            ax2.scatter(*curr_good, np.log10(sse_func(curr_good)), color='green', s=100)
            
            ax2.set_title('Optimizer Path on Error Surface')
            ax2.set_xlabel('Narrow Peak Pos')
            ax2.set_ylabel('Wide Peak Pos')
            ax2.set_zlabel('log10(SSE)')
            ax2.view_init(elev=30, azim=45)
            
            plot_placeholder_2.pyplot(fig)
            plt.close(fig)
            time.sleep(0.01)

# --- EXPLANATION OF THE TWO MINIMUM PROBLEM ---
st.markdown("### Understanding Local vs. Global Minima")
st.write("""
In the animation above, you are watching two separate fits happening at the same time. 

The **Green Path** starts in a lucky spot and rolls smoothly down into the deepest part of the valley. This is the **Global Minimum** where the theoretical model perfectly matches the experimental data.

The **Red Path** controlled by your sliders might get stuck in a shallower crater along the way. Because fitting algorithms are blind and only feel the slope directly under their feet, they think they are done once they hit the bottom of any valley. This trap is called a **Local Minimum**.
""")



st.markdown("### Why We Simplify Parameters")
st.write("""
When fitting complex physical systems, letting every parameter float freely creates incredibly rugged error landscapes filled with these local minimum traps. 

This is exactly why it is often more robust to constrain parameters mathematically rather than treating them as entirely independent variables. By locking in specific relationships and ensuring that only a single physical driver is allowed to change, you effectively slice through this chaotic 3D landscape. You collapse a tricky and cratered map into a simpler and smoother curve. This forces the algorithm to bypass the traps and land exactly on the true global minimum.
""")
