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
    # The new bad starting guess provided by the user
    # Order: [A1, t0, tau1, A2, tau2, B, A3, tau3]
    bad_guess = [25000.0, 13.0, 0.25, 2000.0, 1.5, 2.0, 1.0, 0.01]
    
    # CORRECTED WEIGHTING: 
    # curve_fit expects standard deviation (sigma), which is sqrt(N)
    true_sigma = np.sqrt(np.maximum(y_data, 1))
    
    try:
        # Use Levenberg Marquardt method to match MATLAB default
        popt_bad, pcov_bad = curve_fit(
            pals_fit_func, 
            x_data, 
            y_data, 
            p0=bad_guess, 
            sigma=true_sigma, 
            absolute_sigma=True,
            method='lm', 
            maxfev=10000
        )
        
        y_fit_bad = pals_fit_func(x_data, *popt_bad)
        
        # Weighted residuals are (Data - Fit) / Sigma
        residuals_bad = (y_data - y_fit_bad) / true_sigma
        
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
        
        st.error("The algorithm converged, but it is trapped in a false minimum.")
        
        # Check for infinity or NaN in the covariance matrix (which MATLAB outputs as 0.00 error)
        if np.isinf(pcov_bad).any() or np.isnan(pcov_bad).any():
            perr_bad = np.zeros_like(popt_bad)
        else:
            perr_bad = np.sqrt(np.diag(pcov_bad))
        
        # Display the trapped parameters
        col_trap1, col_trap2, col_trap3 = st.columns(3)
        col_trap1.metric("tau 1", f"{popt_bad[2]:.4f} ns", f"Error: {perr_bad[2]:.4f}")
        col_trap2.metric("tau 2", f"{popt_bad[4]:.4f} ns", f"Error: {perr_bad[4]:.4f}")
        col_trap3.metric("tau 3", f"{popt_bad[7]:.4f} ns", f"Error: {perr_bad[7]:.4f}")

    except Exception as e:
        st.error(f"Fit failed completely. Error: {e}")

st.markdown("### Why this is a Bad Fit")
st.write("""
At first glance, the numerical results might trick you. The algorithm reports very small mathematical errors for the parameters, which could lead you to believe the fit is highly accurate. 

However, look closely at the bottom panel showing the weighted residuals. If you observe the residual progression, you will notice a clear oscillation in the beginning. Instead of scattering like pure random noise, the dots form a distinct wavy pattern. 
""")



st.write("""
This oscillation is a clear systematic deviation. It proves that the mismatch has much more structure than just random statistical noise. Even though the standard errors are tiny, this structural deviation is the ultimate visual proof that the algorithm is trapped in a false minimum and the current model does not reflect the actual physics.
""")

# --- SECTION 5: THE AUTOCORRELATION FUNCTION ---
st.divider()
st.header("5. The Autocorrelation Function")

st.write("""
Visual inspection is a great start, but we need a strict mathematical tool to prove whether our residues are purely random noise or if they hide an unresolved systematic error. 

This is exactly what the Autocorrelation Function does. It takes the residual signal and compares it with a shifted version of itself. 

* **Random Noise:** If the fit is perfect, the errors are purely statistical. Knowing that one data point is too high tells you absolutely nothing about the next data point. The autocorrelation should immediately drop to zero and stay within the statistical confidence limits.
* **Structured Error:** If the fit is trapped in a false minimum, the residues form a wave. A high point is naturally followed by another high point. The autocorrelation function detects this memory in the data and will clearly spike outside the confidence boundaries.
""")



if st.button("Calculate Autocorrelation for Both Fits"):
    # We quickly recalculate both fits under the hood to guarantee we have the residuals
    true_sigma = np.sqrt(np.maximum(y_data, 1))
    
    # 1. The Good Fit
    good_guess = [25000.0, 13.0, 0.3, 2000.0, 1.0, 2.0, 500.0, 2.0]
    popt_good, _ = curve_fit(pals_fit_func, x_data, y_data, p0=good_guess, sigma=true_sigma, absolute_sigma=True, method='trf', bounds=([0, 0, 0.01, 0, 0.01, 0, 0, 0.01], [np.inf, 50.0, 5.0, np.inf, 10.0, np.inf, np.inf, 50.0]), max_nfev=10000)
    res_good = (y_data - pals_fit_func(x_data, *popt_good)) / true_sigma
    
    # 2. The Bad Fit (Trapped)
    bad_guess = [25000.0, 13.0, 0.25, 2000.0, 1.5, 2.0, 1.0, 0.01]
    popt_bad, _ = curve_fit(pals_fit_func, x_data, y_data, p0=bad_guess, sigma=true_sigma, absolute_sigma=True, method='lm', maxfev=10000)
    res_bad = (y_data - pals_fit_func(x_data, *popt_bad)) / true_sigma

    # Function to calculate ACF
    def calc_acf(res, max_lags=100):
        n = len(res)
        res_mean = np.mean(res)
        res_centered = res - res_mean
        sum_sq = np.sum(res_centered**2)
        acf = np.zeros(max_lags + 1)
        for k in range(max_lags + 1):
            if k == 0:
                acf[k] = 1.0
            else:
                acf[k] = np.sum(res_centered[:-k] * res_centered[k:]) / sum_sq
        conf_limit = 1.96 / np.sqrt(n)
        return acf, conf_limit

    acf_good, conf_good = calc_acf(res_good)
    acf_bad, conf_bad = calc_acf(res_bad)
    lags = np.arange(101)

    # Plotting both ACFs side by side
    fig_acf, (ax_acf1, ax_acf2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Good Fit ACF Plot
    ax_acf1.stem(lags, acf_good, linefmt='k-', markerfmt='k.', basefmt='k-')
    ax_acf1.axhline(conf_good, color='red', linestyle=':')
    ax_acf1.axhline(-conf_good, color='red', linestyle=':')
    ax_acf1.set_title("Autocorrelation: Global Minimum")
    ax_acf1.set_xlabel("Lag (Bins)")
    ax_acf1.set_ylabel("Autocorrelation")
    ax_acf1.set_ylim(-0.5, 1)
    
    # Bad Fit ACF Plot
    ax_acf2.stem(lags, acf_bad, linefmt='k-', markerfmt='k.', basefmt='k-')
    ax_acf2.axhline(conf_bad, color='red', linestyle=':')
    ax_acf2.axhline(-conf_bad, color='red', linestyle=':')
    ax_acf2.set_title("Autocorrelation: Trapped in False Minimum")
    ax_acf2.set_xlabel("Lag (Bins)")
    ax_acf2.set_ylim(-0.5, 1)
    
    st.pyplot(fig_acf)

    st.write("""
    Observe the difference between the two plots above. 
    
    The plot on the left represents our successful global minimum fit. Almost immediately after the first bin, the correlation drops to zero and stays securely within the red dotted boundaries. This confirms the remaining errors are just natural counting statistics.
    
    The plot on the right represents the trapped fit. The strong wave pattern violently breaches the confidence limits for dozens of bins. The math proves what our eyes suspected: the model is fundamentally incorrect.
    """)
# --- SECTION 6: THE MELT METHOD ---
from scipy.signal import find_peaks
from scipy.optimize import minimize

st.divider()
st.header("6. The MELT Method: Flooding the Energy Landscape")

st.write("""
In the previous sections, we visualized the fitting process as dropping a single marble into a complex valley. The marble rolls downhill and often gets stuck in a local minimum. We can think of this residual valley as a vast energy landscape. Standard algorithms struggle because they only explore one single path at a time.

The MELT method takes a completely different approach. Instead of guessing a few discrete lifetimes and hoping they land in the right craters, MELT evaluates a continuous spectrum of lifetimes all at once. 

Imagine flooding the entire energy landscape with water. The water naturally fills all the valleys simultaneously. By observing where the water pools, we can identify all the minima at the exact same time. The deepest pools correspond to the most probable lifetimes existing in our data.
""")



st.write("""
Mathematically, MELT achieves this by setting up a grid of dozens of possible lifetimes. It then uses an optimization technique that balances minimizing the statistical errors against maximizing the entropy of the system. This entropy term acts like a physical constraint that prevents the water from splashing everywhere, forcing the algorithm to return smooth and localized peaks representing the true physical decay channels.
""")

if st.button("Run MELT Estimation"):
    st.write("Constructing continuous lifetime spectrum...")
    
    # 1. Prepare the Grid and Kernel to exactly match MATLAB
    fitted_offset = 13.47 
    fitted_B = 1.85
    weights = 1.0 / np.maximum(y_data, 1)
    
    n_taus = 50 
    tau_grid = np.logspace(np.log10(0.01), np.log10(5.0), n_taus)
    K = np.zeros((len(x_data), n_taus))
    
    DeltaT = 0.297 / 2.35
    for j in range(n_taus):
        tau = tau_grid[j]
        K[:, j] = np.exp(-(x_data - fitted_offset)/tau) * erfc(1/np.sqrt(2) * (DeltaT/tau - (x_data - fitted_offset)/DeltaT))
        
    # 2. Optimization (Chi Squared + Entropy)
    lambda_reg = 0.001 
    def melt_objective(alpha):
        y_pred = K @ alpha + fitted_B
        chi_sq = np.sum(weights * (y_data - y_pred)**2)
        # Entropy regularization term
        entropy = lambda_reg * np.sum(alpha * np.log(alpha + 1e-12))
        return chi_sq + entropy

    alpha_guess = np.ones(n_taus) * (np.max(y_data) / n_taus)
    bounds = [(0, None) for _ in range(n_taus)]
    
    # Switch to SLSQP to match the MATLAB sqp algorithm
    res = minimize(
        melt_objective, 
        alpha_guess, 
        method='SLSQP', 
        bounds=bounds, 
        options={'maxiter': 50000, 'ftol': 1e-6}
    )
    alpha_dist = res.x
    
    # 3. Peak Finding (Extracting the top 3 peaks like MATLAB)
    peaks, _ = find_peaks(alpha_dist)
    
    if len(peaks) > 0:
        peak_amps = alpha_dist[peaks]
        # Sort by amplitude descending
        sorted_indices = np.argsort(peak_amps)[::-1]
        # Take the top 3
        top_peaks = peaks[sorted_indices[:min(3, len(peaks))]]
        # Sort back by tau (time) from left to right
        top_peaks = np.sort(top_peaks)
        
        peak_taus = tau_grid[top_peaks]
        peak_amps_display = alpha_dist[top_peaks]
    else:
        peak_taus = []
        peak_amps_display = []
    
    # 4. Visualization
    fig_melt, ax_melt = plt.subplots(figsize=(10, 5))
    ax_melt.semilogx(tau_grid, alpha_dist, 'b', linewidth=2, label="MELT Spectrum")
    
    # Mark the peaks
    for p_tau, p_amp in zip(peak_taus, peak_amps_display):
        ax_melt.semilogx(p_tau, p_amp, 'ro', markersize=8)
        ax_melt.text(p_tau, p_amp * 1.1, f"{p_tau:.3f} ns", ha='center', fontweight='bold', color='red')
        
    ax_melt.set_xlabel("Lifetime tau (ns)")
    ax_melt.set_ylabel("Amplitude alpha(tau)")
    ax_melt.set_title("Continuous Lifetime Spectrum (MELT)")
    ax_melt.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_melt.legend()
    
    st.pyplot(fig_melt)

    

    st.success("MELT successfully mapped the energy landscape.")
    
    st.write("""
    The plot above shows the resulting probability distribution. Every peak represents a distinct lifetime component found directly from the data without requiring a rigid initial guess. 
    
    By simply reading the positions of these peaks on the horizontal axis, we instantly obtain highly accurate estimates for our lifetimes. These values can either serve as the perfect starting point for our standard fitting algorithms or be used directly for theoretical analysis.
    """)
