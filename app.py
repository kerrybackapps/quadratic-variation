import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Quadratic Variation of Brownian Motion",
    layout="wide"
)

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
            showlegend=(m <= 5)
        ))
    
    fig1.update_layout(
        xaxis_title="Time (t)",
        yaxis_title="Brownian Motion B(t)",
        template="plotly_white",
        height=400
    )

    # Quadratic variation plot
    fig2 = go.Figure()
    
    # Add each selected path's quadratic variation
    for i in range(min(m, max_m)):
        fig2.add_trace(go.Scatter(
            x=t[:, 0], 
            y=QV[:, i], 
            mode='lines',
            name=f'Path {i+1}' if m <= 5 else None,
            showlegend=(m <= 5)
        ))
    
    # Add theoretical line [B]_t = t
    fig2.add_trace(go.Scatter(
        x=t[:, 0],
        y=t[:, 0],
        mode='lines',
        name='Theoretical: [B]_t = t',
        line=dict(color='red', width=3, dash='dash'),
        showlegend=True
    ))
    
    fig2.update_layout(
        xaxis_title="Time",
        yaxis_title="Quadratic Variation",
        template="plotly_white",
        height=400
    )

    return fig1, fig2

# Top control area
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_horizon = st.slider(
            "Time horizon:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    with col2:
        n = st.selectbox(
            "Number of subdivisions:",
            options=[10, 100, 1000, 10000, 100000],
            index=1
        )
    
    with col3:
        m = st.slider(
            "Number of paths:",
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
    

# Generate and display plots
fig1, fig2 = create_quad_var_plots(n, m, time_horizon)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)

