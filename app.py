import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Quadratic Variation of Brownian Motion",
    layout="wide"
)

# CSS for responsive iframe scaling and Bootstrap-like theme
st.markdown("""
<style>
/* Bootstrap-like color scheme */
:root {
    --primary-color: #0d6efd;
    --secondary-color: #6c757d;
    --success-color: #198754;
    --info-color: #0dcaf0;
    --warning-color: #ffc107;
    --danger-color: #dc3545;
    --light-color: #f8f9fa;
    --dark-color: #212529;
}

/* Make the entire app responsive */
.main .block-container {
    max-width: 100% !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Scale content to fit iframe */
.stApp {
    transform-origin: top left;
    width: 100%;
    background-color: #ffffff;
    color: #212529;
}

/* Ensure plots scale properly */
.js-plotly-plot {
    width: 100% !important;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: var(--light-color);
}

/* Button styling */
.stButton > button {
    background-color: var(--primary-color);
    color: white;
    border: 1px solid var(--primary-color);
    border-radius: 0.375rem;
}

.stButton > button:hover {
    background-color: #0b5ed7;
    border-color: #0a58ca;
}

/* Input styling */
.stTextInput > div > div > input,
.stSelectbox > div > div > select,
.stSlider > div > div > div > div {
    border-color: #ced4da;
    border-radius: 0.375rem;
}

/* Metric styling */
.metric-container {
    background-color: var(--light-color);
    border: 1px solid #dee2e6;
    border-radius: 0.375rem;
    padding: 1rem;
}

/* Slider styling */
.stSlider > div > div > div > div {
    background-color: var(--primary-color);
}
</style>
""", unsafe_allow_html=True)

def b_motion_np(n, m, T, seeded=True):
    """Generate Brownian motion paths"""
    dt = T/n  # partition the total time interval into equally spaced intervals of length dt
    sig = 1
    vol = sig*np.sqrt(dt)
    if seeded:
        seed = 1234
        rg = np.random.RandomState(seed) 
        incs = rg.standard_normal(size=(n, m))
    else:
        incs = np.random.standard_normal(size=(n, m))
                                          
    Bt = np.concatenate((np.zeros((1, m)), incs), axis=0).cumsum(axis=0) 
    Bt *= vol
    tline = np.linspace(0, T, n+1)                            
    t = np.repeat(tline.reshape((n+1, 1)), m, axis=1)   
    return Bt, t

def create_quad_var_plots(n, m, T):
    """Create quadratic variation plots"""
    max_m = 10  # simulate max pre-set paths, user is just slicing them
    Bt, t = b_motion_np(n=n, m=max_m, T=T, seeded=True) 
    QV = (Bt[1:n + 1] - Bt[0:n])**2
    QV = np.concatenate((np.zeros((1, max_m)), QV), axis=0).cumsum(axis=0)

    # Brownian motion paths plot
    fig1 = go.Figure()
    
    # Add each selected path
    for i in range(min(m, max_m)):
        fig1.add_trace(go.Scatter(
            x=t[:, 0], 
            y=Bt[:, i], 
            mode='lines',
            name=f'Path {i+1}' if m <= 5 else None,
            showlegend=False
        ))
    
    fig1.update_layout(
        xaxis_title="Time",
        yaxis_title="Brownian Motion",
        template="plotly_white",
        height=350,
        width=400
    )

    # Quadratic variation plot
    fig2 = go.Figure()
    
    # Add each selected path's quadratic variation
    for i in range(min(m, max_m)):
        fig2.add_trace(go.Scatter(
            x=t[:, 0], 
            y=QV[:, i], 
            mode='lines',
            showlegend=False
        ))
    
    # Add theoretical line [B]_t = t
    fig2.add_trace(go.Scatter(
        x=t[:, 0],
        y=t[:, 0],
        mode='lines',
        line=dict(color='black', width=3, dash='dash'),
        showlegend=False
    ))
    
    fig2.update_layout(
        xaxis_title="Time",
        yaxis_title="Quadratic Variation",
        template="plotly_white",
        height=350,
        width=400,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text="- - - - Theoretical",
                showarrow=False,
                font=dict(size=16, weight="bold"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        ]
    )

    return fig1, fig2

# Top control area
with st.container():
    [col1,col2, col3] = st.columns(3)
     
    with col2:
        n = st.slider(
            "Number of subdivisions of time interval:",
            min_value=2,
            max_value=10000,
            value=10,
            step=1
        )
    
# Generate and display plots
fig1, fig2 = create_quad_var_plots(n, 5, 1)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)

