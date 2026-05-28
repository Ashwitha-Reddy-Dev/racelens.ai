import os
from dotenv import load_dotenv
import streamlit as st  # type: ignore
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
model_id = "gemini_2.5_flash"
from racelens_gemini import RaceLensOrchestrator

# Page setup
st.set_page_config(
    page_title="🏁 RaceLens AI",
    page_icon="🏎️",
    layout="wide"
)

# Title
st.title("🏁 RaceLens AI")
st.markdown("**Explainable F1 Race Data Interpreter**")
st.markdown("Powered by IBM Granite + LangFlow + Docling")
st.markdown("---")

# Initialize
if "racelens" not in st.session_state:
    st.session_state.racelens = RaceLensOrchestrator(api_key=api_key, model_id=model_id)

racelens = st.session_state.racelens

# Sidebar - Mode Selection
with st.sidebar:
    st.title("🏁 RaceLens AI")
    mode = st.radio(
        "Choose Demo Mode:",
        [
            "🗣️ Ask the Race (Live)",
            "👶 Rookie Mode",
            "📺 Broadcaster Copilot",
            "📊 All 3 Demos"
        ]
    )
    
    st.markdown("---")
    st.info("""
    ### About RaceLens
    
    Uses IBM Granite 3.1 to explain F1 racing tactics in plain English.
    
    Every answer includes:
    - Sources & citations
    - Confidence scores
    - What-if scenarios
    """)

# Current Race State (sample data)
race_data = {
    "circuit": "Bahrain",
    "lap": 35,
    "total_laps": 57,
    "positions": [
        {"position": 1, "driver": "Lando Norris", "gap": "—", "tire": "HARD"},
        {"position": 2, "driver": "Max Verstappen", "gap": "+2.3s", "tire": "HARD"},
    ]
}

st.subheader("📊 Current Race State")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Circuit", race_data["circuit"])
with col2:
    st.metric("Lap", f"{race_data['lap']}/{race_data['total_laps']}")
with col3:
    st.metric("Leader", race_data["positions"][0]["driver"])
with col4:
    st.metric("Gap P2", race_data["positions"][1]["gap"])

# Positions table
st.subheader("Top 3 Positions")
positions_data = []
for p in race_data["positions"][:3]:
    positions_data.append({
        "Pos": p["position"],
        "Driver": p["driver"],
        "Gap": p["gap"],
        "Tire": p["tire"],
        "Speed": f"{p.get('speed', 'N/A')} km/h"
        if p.get('speed') else "N/A"
    })
st.dataframe(positions_data, use_container_width=True, hide_index=True)

st.markdown("---")

# ============================================================================
# MODE 1: Ask the Race
# ============================================================================

if mode == "🗣️ Ask the Race (Live)":
    st.subheader("🗣️ Ask the Race")
    st.markdown("Ask any question about what's happening on track right now.")
    
    question = st.text_input(
        "Your question:",
        placeholder="e.g., 'Why did Norris lose the lead?' or 'What's the pit stop strategy?'"
    )
    
    if st.button("🔍 Ask RaceLens", key="ask_btn"):
        if question:
            with st.spinner("RaceLens thinking..."):
                answer = racelens.ask_the_race(question)
            
            st.markdown("### Your Question")
            st.info(question)
            
            st.markdown("### RaceLens Answer")
            st.success(answer.answer)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Confidence", f"{answer.confidence}%")
            with col2:
                st.write(f"**Sources:**  \n{chr(10).join([f'• {s}' for s in answer.sources])}")

# ============================================================================
# MODE 2: Rookie Mode
# ============================================================================

elif mode == "👶 Rookie Mode":
    st.subheader("👶 Rookie Mode")
    st.markdown("New to F1? Learn the basics explained in simple terms.")
    
    topics = [
        "Why do they use different colored tires?",
        "What's a pit stop?",
        "Why is track position important?",
        "What does DRS mean?",
        "Why do pit stops take so long?",
        "How fast do F1 cars go?"
    ]
    
    selected = st.selectbox("Choose a topic:", topics)
    
    if st.button("📖 Explain This", key="rookie_btn"):
        with st.spinner("Explaining..."):
            answer = racelens.ask_the_race(selected)
        
        st.markdown("### Your Question")
        st.info(selected)
        
        st.markdown("### Simple Explanation")
        st.success(answer.answer)
        
        st.metric("Confidence", f"{answer.confidence}%")

# ============================================================================
# MODE 3: Broadcaster Copilot
# ============================================================================

elif mode == "📺 Broadcaster Copilot":
    st.subheader("📺 Broadcaster Copilot")
    st.markdown("Live research assistant for on-air commentators.")
    
    queries = [
        "Last 5 wet races at Spa - pole sitter vs race winner correlation",
        "Average pit stop time this season",
        "Verstappen vs Norris head-to-head stats 2024",
        "Most common pit stop mistakes",
        "Tire strategy at this circuit historically"
    ]
    
    selected = st.selectbox("Quick research query:", queries)
    
    if st.button("📊 Research Now", key="broadcast_btn"):
        with st.spinner("Researching..."):
            answer = racelens.ask_the_race(selected)
        
        st.markdown("### Query")
        st.info(selected)
        
        st.markdown("### Answer for On-Air")
        st.success(answer.answer)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{answer.confidence}%")
        with col2:
            st.metric("Response Time", "~2 seconds")
        
        st.markdown("**Source References:**")
        for source in answer.sources[:2]:
            st.write(f"• {source}")

# ============================================================================
# MODE 4: All 3 Demos
# ============================================================================

else:  # All 3 Demos
    st.subheader("📊 Complete Demo")
    st.markdown("Watch all three RaceLens capabilities in action.")
    
    # Demo 1
    with st.expander("📊 Demo 1: The Bahrain GP Undercut (Strategy)", expanded=True):
        st.info("**Situation:** Lap 35 at Bahrain. Lando Norris is leading but just pitted. After exiting the pits, he's BEHIND Max Verstappen. What happened?")
        
        q1 = "Why did Norris just lose the lead after the pit stop?"
        a1 = st.session_state.racelens.ask_the_race(q1)
        
        st.markdown(f"**Question:** {q1}")
        st.success(a1.answer)
        st.metric("Confidence", f"{a1.confidence}%")
    
    # Demo 2
    with st.expander("👶 Demo 2: Rookie Mode - Tire Explanation"):
        st.info("**Situation:** A new F1 fan is confused about tire colors and why teams switch them.")
        
        q2 = "I'm new to F1. Why do they use different colored tires and swap them?"
        a2 = st.session_state.racelens.ask_the_race(q2)
        
        st.markdown(f"**Question:** {q2}")
        st.success(a2.answer)
        st.metric("Confidence", f"{a2.confidence}%")
    
    # Demo 3
    with st.expander("📺 Demo 3: Broadcaster Copilot - Historical Research"):
        st.info("**Situation:** A Sky Sports commentator is on air at Spa. Heavy rain hits. They need: 'In wet qualifying, how much does pole position matter?'")
        
        q3 = "Last 5 wet races at Spa - pole sitter vs race winner correlation?"
        a3 = st.session_state.racelens.ask_the_race(q3)
        
        st.markdown(f"**Question:** {q3}")
        st.success(a3.answer)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{a3.confidence}%")
        with col2:
            st.metric("Response Time", "~2 seconds")

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #666;">
<strong>🏁 RaceLens AI</strong> — Explainable F1 Race Data Interpreter<br>
Powered by IBM Granite 3.1 + LangFlow + Docling<br>
<small>Built for the F1 Grand Prix AI Challenge</small>
</p>
""", unsafe_allow_html=True) 