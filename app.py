import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌱",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f5fff7;
}
.big-title {
    font-size: 48px;
    font-weight: 800;
    color: #1b5e20;
    text-align: center;
}
.subtitle {
    font-size: 20px;
    color: #4f4f4f;
    text-align: center;
    margin-bottom: 30px;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.metric-card {
    background: linear-gradient(135deg, #d4fc79, #96e6a1);
    padding: 22px;
    border-radius: 18px;
    text-align: center;
    font-weight: bold;
}
.result-box {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 28px;
    border-radius: 20px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
data = pd.read_csv("Crop_recommendation.csv")

if len(data.columns) == 1:
    data = data.iloc[:, 0].str.split(",", expand=True)
    data.columns = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall", "label"]

data.columns = data.columns.str.strip().str.lower()

for col in ["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna()

X = data.drop("label", axis=1)
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------- HEADER ----------
st.markdown("<div class='big-title'>🌱 AgriVision AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Smart Crop Recommendation System using Machine Learning</div>",
    unsafe_allow_html=True
)

# ---------- TOP METRICS ----------
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class='metric-card'>
        🤖 Model Accuracy<br>
        <span style='font-size:28px'>{round(accuracy*100,2)}%</span>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class='metric-card'>
        🌾 Crop Classes<br>
        <span style='font-size:28px'>22+</span>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class='metric-card'>
        📊 Input Factors<br>
        <span style='font-size:28px'>7</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------- REGION DATA ----------
region_data = {
    "Manual Entry": None,
    "Chennai": [80, 45, 40, 30.5, 75, 6.5, 140],
    "Coimbatore": [70, 40, 38, 26.5, 68, 6.8, 110],
    "Madurai": [75, 42, 35, 31.0, 60, 7.0, 90],
    "Salem": [65, 38, 36, 28.5, 65, 6.7, 100],
    "Tirunelveli": [85, 45, 42, 29.0, 70, 6.4, 130],
    "Trichy": [78, 44, 38, 30.0, 67, 6.9, 95],
    "Thanjavur": [90, 50, 45, 29.0, 78, 6.6, 150],
    "Erode": [72, 41, 36, 28.0, 66, 6.8, 100],
    "Vellore": [68, 39, 35, 31.5, 58, 7.1, 85],
    "Kanyakumari": [88, 48, 44, 27.5, 80, 6.4, 180],
    "Dindigul": [70, 40, 37, 29.0, 62, 6.9, 95],
    "Thoothukudi": [76, 43, 39, 32.0, 65, 7.2, 75],
    "Nagapattinam": [92, 52, 46, 28.5, 82, 6.5, 170],
    "Cuddalore": [86, 47, 43, 29.5, 79, 6.6, 155],
    "Karur": [69, 38, 34, 31.0, 60, 7.0, 80]
}

fertilizer_map = {
    "rice": "Urea + Potash",
    "maize": "NPK Fertilizer",
    "cotton": "Nitrogen-rich Fertilizer",
    "wheat": "DAP + Urea",
    "sugarcane": "Organic Compost + NPK",
    "banana": "Organic Manure + Potash",
    "mango": "Farmyard Manure + NPK",
    "grapes": "Potassium-rich Fertilizer"
}

profit_map = {
    "rice": "High demand crop in water-rich regions.",
    "maize": "Good market demand with moderate investment.",
    "cotton": "High profit but needs pest control.",
    "wheat": "Stable crop with reliable returns.",
    "sugarcane": "Very profitable but needs more water.",
    "banana": "High commercial value with regular demand.",
    "mango": "Long-term profitable fruit crop.",
    "grapes": "Premium market value crop."
}

# ---------- MAIN LAYOUT ----------
left, right = st.columns([1.1, 0.9])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🌍 Select Region / Enter Field Data")

    region = st.selectbox("Choose Region", list(region_data.keys()))

    if region_data[region]:
        n, p, k, temperature, humidity, ph, rainfall = region_data[region]
    else:
        n = p = k = 0
        temperature = humidity = ph = rainfall = 0.0

    c1, c2 = st.columns(2)

    with c1:
        n = st.slider("Nitrogen (N)", 0, 150, int(n))
        p = st.slider("Phosphorus (P)", 0, 150, int(p))
        k = st.slider("Potassium (K)", 0, 210, int(k))
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, float(temperature))

    with c2:
        humidity = st.slider("Humidity (%)", 0.0, 100.0, float(humidity))
        ph = st.slider("Soil pH", 0.0, 14.0, float(ph))
        rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, float(rainfall))

    predict_btn = st.button("🚀 Predict Best Crop", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Field Data Overview")

    chart_data = pd.DataFrame({
        "Parameter": ["N", "P", "K", "Temp", "Humidity", "pH", "Rainfall"],
        "Value": [n, p, k, temperature, humidity, ph, rainfall]
    })

    st.bar_chart(chart_data.set_index("Parameter"))
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- RESULT ----------
if predict_btn:
    input_data = [[n, p, k, temperature, humidity, ph, rainfall]]
    crop = model.predict(input_data)[0]
    suitability = max(model.predict_proba(input_data)[0]) * 100

    fertilizer = fertilizer_map.get(crop, "Balanced NPK Fertilizer")
    profit = profit_map.get(crop, "Profitable based on soil and climate suitability.")

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(f"<h1>🌾 Recommended Crop: {crop.capitalize()}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3>Suitability Score: {round(suitability,2)}%</h3>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    r1, r2 = st.columns(2)

    with r1:
        st.info(f"🌿 **Fertilizer Suggestion:** {fertilizer}")

    with r2:
        st.success(f"💰 **Profit Insight:** {profit}")

    st.write("### 🧠 Why this crop?")
    st.write(
        f"The AI model analyzed soil nutrients, temperature, humidity, pH, and rainfall. "
        f"Based on these conditions, **{crop.capitalize()}** is the most suitable crop."
    )

# ---------- AGRIBOT ----------
# ---------- AGRIBOT ----------
st.markdown("---")
st.subheader("🤖 AgriBot Assistant")

question = st.text_input("Chat with AgriBot")

crop_details = {
    "rice": {
        "info": "Rice grows best in warm, humid areas with high rainfall and good water availability.",
        "fertilizer": "Urea + Potash",
        "profit": "Rice can be profitable in water-rich regions because demand is high."
    },
    "maize": {
        "info": "Maize grows well in warm weather with moderate rainfall and well-drained soil.",
        "fertilizer": "NPK fertilizer",
        "profit": "Maize has good market demand and gives steady returns."
    },
    "cotton": {
        "info": "Cotton prefers warm climate and nutrient-rich soil, but it needs pest control.",
        "fertilizer": "Nitrogen-rich fertilizer",
        "profit": "Cotton can give high profit, but farming cost is also higher."
    },
    "banana": {
        "info": "Banana grows well in warm and humid regions with regular irrigation.",
        "fertilizer": "Organic manure + Potash",
        "profit": "Banana has good commercial value and regular market demand."
    },
    "sugarcane": {
        "info": "Sugarcane needs high water supply, warm climate, and fertile soil.",
        "fertilizer": "Organic compost + NPK",
        "profit": "Sugarcane can be very profitable but needs more water and time."
    }
}

def agribot_answer(q):
    q = q.lower().strip()

    if q == "":
        return ""

    greetings = ["hi", "hello", "hey", "hai", "hii", "hiii", "good morning", "good evening"]
    if any(word == q for word in greetings):
        return "Hello 👋 I’m AgriBot! You can ask me about crops, fertilizer, rainfall, soil health, profit, or region-based farming 🌱"

    if "how are you" in q:
        return "I’m doing great 😄 I’m ready to help you choose better crops and improve farming decisions 🌾"

    if "who are you" in q or "what are you" in q:
        return "I’m AgriBot 🤖, a smart agriculture assistant that helps with crop selection, fertilizer suggestions, soil health, rainfall, and profit tips."

    if "thank" in q:
        return "You’re welcome 😄 Happy farming and smart cultivation 🌱"

    if "bye" in q:
        return "Goodbye 👋 Wishing you healthy crops and high-profit harvests 🌾💰"

    for crop, details in crop_details.items():
        if crop in q:
            if "fertilizer" in q:
                return f"For **{crop.capitalize()}**, the recommended fertilizer is **{details['fertilizer']}** 🌿"
            elif "profit" in q or "profitable" in q:
                return f"**{crop.capitalize()} profit insight:** {details['profit']} 💰"
            elif "rainfall" in q or "water" in q:
                return f"**{crop.capitalize()} water/rainfall guidance:** {details['info']} 🌧️"
            else:
                return f"**{crop.capitalize()} guidance:** {details['info']} 🌱"

    if "best crop" in q or "which crop" in q:
        return "The best crop depends on soil nutrients, temperature, humidity, pH, and rainfall. Use the prediction section above to get an AI-based crop recommendation 🌾"

    if "fertilizer" in q:
        return "Fertilizer depends on the crop and soil condition. Nitrogen supports leaf growth, Phosphorus helps root development, and Potassium improves crop quality 🌿"

    if "soil" in q or "ph" in q:
        return "Good soil should have balanced NPK nutrients, suitable pH, moisture, and organic matter. Soil testing helps choose the right crop 🧪"

    if "rainfall" in q or "water" in q:
        return "Rainfall is important because crops need water for germination, nutrient absorption, and yield. Too little or too much rainfall can reduce crop growth 🌧️"

    if "profit" in q or "market" in q:
        return "To improve profit, choose crops suitable for your region, reduce fertilizer waste, use proper irrigation, and consider market demand before cultivation 💰"

    if "chennai" in q:
        return "For Chennai, crops like rice, banana, maize, and groundnut may be suitable depending on soil nutrients, rainfall, and irrigation availability 🌍"

    return "I can help with farming questions 😊 Try asking: **fertilizer for rice**, **best crop for Chennai**, **how to improve profit**, or **soil health tips**."

if question:
    st.info(agribot_answer(question))
