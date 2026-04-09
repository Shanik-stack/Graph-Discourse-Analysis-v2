import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(layout="wide")
st.title("Dynamic Argument Graph (No NetworkX / No PyVis)")

# Matrix size
size = st.number_input("Number of claims (nodes)", min_value=2, max_value=50, value=6)

# Initialize A_T with random -1, 0, 1
if "A_T" not in st.session_state or st.session_state.A_T.shape[0] != size:
    st.session_state.A_T = np.random.choice([-1, 0, 1], size=(size, size))

placeholder = st.empty()

# random node positions (fixed)
if "pos" not in st.session_state or len(st.session_state.pos) != size:
    angles = np.linspace(0, 2 * np.pi, size, endpoint=False)
    st.session_state.pos = np.c_[np.cos(angles), np.sin(angles)]


# dynamic loop
for _ in range(1000000):

    A_T = st.session_state.A_T
    pos = st.session_state.pos

    fig = go.Figure()

    # Draw edges
    for i in range(size):
        for j in range(size):
            if A_T[i, j] != 0:
                x0, y0 = pos[i]
                x1, y1 = pos[j]

                color = "green" if A_T[i, j] == 1 else "red"

                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=2, color=color),
                    hoverinfo="none",
                    showlegend=False
                ))

    # Draw nodes
    fig.add_trace(go.Scatter(
        x=pos[:, 0],
        y=pos[:, 1],
        mode="markers+text",
        text=[f"Claim {i}" for i in range(size)],
        textposition="top center",
        marker=dict(size=20, color="lightblue"),
        hoverinfo="text",
        showlegend=False
    ))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=700,
        height=700,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    placeholder.plotly_chart(fig, use_container_width=True)

    time.sleep(0.5)
