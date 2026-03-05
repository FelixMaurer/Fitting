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

# --- SECTION 3: THE LIFETIME FIT ---
import os
from scipy.special import erfc
from scipy.optimize import curve_fit

st.markdown("***")
st.header("3. The Straightforward Lifetime Fit")

st.write("""
Now we apply the fitting algorithm to a real physical scenario. In lifetime spectroscopy, we measure how long particles survive before they decay or annihilate. 

The theoretical model is not a simple line. It is a sum of exponential decays. However, our detectors are not perfectly precise. They have a timing resolution that smears the data. Mathematically, this means we must convolve our exponential decay model with a Gaussian distribution representing the detector resolution.
""")

st.latex(r"F(t) = B + \sum_{i=1}^{3} A_i \exp\left(-\frac{t-t_0}{\tau_i}\right) \text{erfc}\left( \frac{1}{\sqrt{2}} \left( \frac{\sigma}{\tau_i} - \frac{t-t_0}{\sigma} \right) \right)")

st.write("""
In this equation:
* **A** represents the amplitude or intensity of a specific decay channel.
* **tau** represents the lifetime of that state.
* **t0** is the exact time zero point when the particles arrive.
* **sigma** is the detector resolution.
* **B** is the constant background noise.
* **erfc** is the complementary error function which handles the Gaussian smearing.
""")

# Load Data or generate dummy data if file is missing
file_name = 'positronlifetime.txt'
if os.path.exists(file_name):
    data = np.loadtxt(file_name)
    x_raw = data[:, 0]
    y_raw = data[:, 1]
    
    # Safe indexing for ROI
    i_start = 2300
    i_end = len(x_raw) - 1200
    x_data = x_raw[i_start:i_end]
    y_data = y_raw[i_start:i_end]
else:
    st.warning("Data file not found. Using synthetic data for demonstration.")
    x_data = np.linspace(10, 30, 1000)
    sigma_demo = 0.297 / 2.35
    def mock_model(x):
        comp1 = 25000 * np.exp(-(x-13.5)/0.25) * erfc(1/np.sqrt(2) * (sigma_demo/0.25 - (x-13.5)/sigma_demo))
        comp2 = 1700 * np.exp(-(x-13.5)/0.61) * erfc(1/np.sqrt(2) * (sigma_demo/0.61 - (x-13.5)/sigma_demo))
        comp3 = 350 * np.exp(-(x-13.5)/1.68) * erfc(1/np.sqrt(2) * (sigma_demo/1.68 - (x-13.5)/sigma_demo))
        return comp1 + comp2 + comp3 + 2.0
    
    y_true = mock_model(x_data)
    # Add Poisson-like noise
    y_data = np.random.poisson(np.maximum(y_true, 0))

# The Fit Function
def pals_fit_func(x, A1, t0, tau1, A2, tau2, B, A3, tau3):
    DeltaT = 0.297 / 2.35 # sigma
    
    comp1 = A1 * np.exp(-(x-t0)/tau1) * erfc(1/np.sqrt(2) * (DeltaT/tau1 - (x-t0)/DeltaT))
    comp2 = A2 * np.exp(-(x-t0)/tau2) * erfc(1/np.sqrt(2) * (DeltaT/tau2 - (x-t0)/DeltaT))
    comp3 = A3 * np.exp(-(x-t0)/tau3) * erfc(1/np.sqrt(2) * (DeltaT/tau3 - (x-t0)/DeltaT))
    
    return comp1 + comp2 + comp3 + B

st.subheader("Adjust Starting Parameters")
st.write("A fitting algorithm needs a starting point. The coarse default values provided below are positioned well enough to ensure the algorithm successfully converges to the true global minimum.")

st.write("**The Challenge:** As we saw in the previous section, a complex landscape is full of traps. I challenge you to experiment with these inputs before moving on. Can you find a combination of starting values that traps the algorithm in a false local minimum where the final fit clearly misses the data, or causes it to fail completely?")

col1, col2, col3, col4 = st.columns(4)
with col1:
    g_t0 = st.number_input("t0", value=13.0)
    g_B = st.number_input("Background B", value=2.0)
with col2:
    g_A1 = st.number_input("A1", value=25000.0)
    g_tau1 = st.number_input("tau1", value=0.3)
with col3:
    g_A2 = st.number_input("A2", value=2000.0)
    g_tau2 = st.number_input("tau2", value=1.0)
with col4:
    g_A3 = st.number_input("A3", value=500.0)
    g_tau3 = st.number_input("tau3", value=2.0)

initial_guess = [g_A1, g_t0, g_tau1, g_A2, g_tau2, g_B, g_A3, g_tau3]

initial_guess = [g_A1, g_t0, g_tau1, g_A2, g_tau2, g_B, g_A3, g_tau3]

if st.button("Perform Lifetime Fit"):
    # Calculate Weights (1 / sqrt(N))
    weights = 1.0 / np.sqrt(np.maximum(y_data, 1))
    
    # Define bounds: [A1, t0, tau1, A2, tau2, B, A3, tau3]
    # Lower bounds: everything must be strictly positive
    lower_bounds = [0, 0, 0.01, 0, 0.01, 0, 0, 0.01]
    # Upper bounds: set arbitrarily high for amplitudes, but reasonable for times
    upper_bounds = [np.inf, 50.0, 5.0, np.inf, 10.0, np.inf, np.inf, 50.0]
    
    try:
        popt, pcov = curve_fit(
            pals_fit_func, 
            x_data, 
            y_data, 
            p0=initial_guess, 
            sigma=weights, 
            absolute_sigma=True,
            bounds=(lower_bounds, upper_bounds), 
            method='trf',                        
            max_nfev=10000                       
        )
        
        y_fit = pals_fit_func(x_data, *popt)
        residuals = (y_data - y_fit) * weights 
        
        # Plotting
        fig, (ax_fit, ax_res) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Fit Plot
        ax_fit.semilogy(x_data, y_data, 'k.', markersize=2, label="Data")
        ax_fit.semilogy(x_data, y_fit, 'r-', linewidth=2, label="Final Fit")
        ax_fit.set_ylabel("Counts (log scale)")
        ax_fit.set_title("Lifetime Spectra and Final Fit")
        ax_fit.legend()
        
        # Residual Plot
        ax_res.plot(x_data, residuals, color='gray', linewidth=0.5)
        ax_res.axhline(0, color='red', linewidth=1)
        ax_res.axhline(2, color='red', linestyle=':')
        ax_res.axhline(-2, color='red', linestyle=':')
        ax_res.set_xlabel("Time (ns)")
        ax_res.set_ylabel("Weighted Residuals")
        ax_res.set_ylim(-5, 5)
        
        st.pyplot(fig)
        
        st.success("Fit converged successfully. See the final parameters below.")
        
        # Calculate Errors and Intensities
        perr = np.sqrt(np.diag(pcov))
        I_tot = popt[0] + popt[3] + popt[6]
        I1 = (popt[0] / I_tot) * 100
        I2 = (popt[3] / I_tot) * 100
        I3 = (popt[6] / I_tot) * 100

        # Construct Results Table
        results_data = {
            "Parameter": [
                "Amplitude A1", "Intensity I1 (%)", "Tau 1 (ns)",
                "Amplitude A2", "Intensity I2 (%)", "Tau 2 (ns)",
                "Amplitude A3", "Intensity I3 (%)", "Tau 3 (ns)",
                "Offset t0 (ns)", "Background B"
            ],
            "Estimate": [
                f"{popt[0]:.2f}", f"{I1:.2f}", f"{popt[2]:.4f}",
                f"{popt[3]:.2f}", f"{I2:.2f}", f"{popt[4]:.4f}",
                f"{popt[6]:.2f}", f"{I3:.2f}", f"{popt[7]:.4f}",
                f"{popt[1]:.4f}", f"{popt[5]:.2f}"
            ],
            "Std. Error (±)": [
                f"{perr[0]:.2f}", "", f"{perr[2]:.4f}",
                f"{perr[3]:.2f}", "", f"{perr[4]:.4f}",
                f"{perr[6]:.2f}", "", f"{perr[7]:.4f}",
                f"{perr[1]:.4f}", f"{perr[5]:.2f}"
            ]
        }
        
        st.table(results_data)

    except Exception as e:
        st.error(f"Fit failed to converge. Try adjusting the starting parameters. Error: {e}")

# Explanation of Residuals
st.markdown("### Reading the Residuals Plot")
st.write("""
The bottom panel in the plot above shows the weighted residuals. This is the absolute best tool to evaluate the quality of your fit. 

When an algorithm minimizes the sum of squared errors, it assumes that any remaining mismatch between the data and the model is purely random statistical noise. 

If your model perfectly describes the physics of the system, the residuals will look like TV static. They should be scattered entirely randomly around the zero line, mostly staying within the limits of plus 2 and minus 2. 
""")



st.write("""
If you see clear wavy patterns, slopes, or large spikes in the residuals, it means the model is failing to capture a physical process. The algorithm might have successfully minimized the math, but the physics is still incomplete. You might need to add another lifetime component or adjust your detector resolution parameter.
""")
# --- SECTION 4: THE WRONG MINIMUM ---
st.divider()
st.header("4. Trapped in the Wrong Minimum")

st.write("""
Let us look at a concrete example of a fit gone wrong. 

We will feed the algorithm a specific set of poor starting values where the amplitudes and lifetimes are severely mixed up. Component three is given an extremely short lifetime guess, and the main amplitude is severely underestimated. 

Click the button below to see what happens when the algorithm tries to optimize from this bad starting location.
""")

if st.button("Run Trapped Fit"):
    # The bad starting guess provided by the user
    # Order: [A1, t0, tau1, A2, tau2, B, A3, tau3]
    bad_guess = [25000.0, 13.47, 0.3, 600.0, 1.5, 1.85, 1.0, 0.01]
    
    weights = 1.0 / np.sqrt(np.maximum(y_data, 1))
    lower_bounds = [-np.inf,-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
    
    try:
        popt_bad, pcov_bad = curve_fit(
            pals_fit_func, 
            x_data, 
            y_data, 
            p0=bad_guess, 
            sigma=weights, 
            absolute_sigma=True,
            bounds=(lower_bounds, upper_bounds),
            method='trf',
            max_nfev=50000
        )
        
        y_fit_bad = pals_fit_func(x_data, *popt_bad)
        residuals_bad = (y_data - y_fit_bad) * weights
        
        # Plotting the bad fit
        fig_bad, (ax_fit_bad, ax_res_bad) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        ax_fit_bad.semilogy(x_data, y_data, 'k.', markersize=2, label="Data")
        ax_fit_bad.semilogy(x_data, y_fit_bad, 'r-', linewidth=2, label="Trapped Fit")
        ax_fit_bad.set_ylabel("Counts (log scale)")
        ax_fit_bad.set_title("Lifetime Spectra with Local Minimum Fit")
        ax_fit_bad.legend()
        
        ax_res_bad.plot(x_data, residuals_bad, color='gray', linewidth=0.5)
        ax_res_bad.axhline(0, color='red', linewidth=1)
        ax_res_bad.axhline(2, color='red', linestyle=':')
        ax_res_bad.axhline(-2, color='red', linestyle=':')
        ax_res_bad.set_xlabel("Time (ns)")
        ax_res_bad.set_ylabel("Weighted Residues")
        ax_res_bad.set_ylim(-15, 15) 
        
        st.pyplot(fig_bad)
        st.error("The algorithm stopped, but the physics is wrong.")

    except Exception as e:
        st.error(f"Fit failed completely. Error: {e}")

st.markdown("### Why this is a Bad Fit")
st.write("""
Look closely at the bottom panel showing the weighted residues. 

In a good fit, these dots should look like static noise on an old television. They should scatter randomly around the zero line. 

However, in the plot above, the residues form massive waves and obvious structures. They shoot far beyond the acceptable boundaries. This systematic deviation is the absolute clearest sign that the algorithm found a local crater rather than the true global minimum. It tells us that the mathematical model currently displayed does not reflect the actual physics of the measured data.
""")
