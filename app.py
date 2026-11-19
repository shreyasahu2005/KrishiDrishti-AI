import streamlit as st
import plotly.express as px
from PIL import Image
from utils import predict_disease
from datetime import datetime
from analytics import calculate_analytics
import pandas as pd
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO
from reportlab.platypus import Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Image as ReportLabImage
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet

HISTORY_FILE = "prediction_history.csv"

# --------------------------------
# STEP 15.5 - CSV ERROR HANDLING
# --------------------------------

def load_prediction_history():

    try:
        if os.path.exists(HISTORY_FILE):
            return pd.read_csv(HISTORY_FILE)

        else:
            return pd.DataFrame()

    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    except pd.errors.ParserError:
        st.error("Prediction history CSV file is corrupted.")
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Unable to load prediction history: {e}")
        return pd.DataFrame()


def save_prediction_history(record):

    if os.path.exists(HISTORY_FILE):
        try:
            old_records = pd.read_csv(HISTORY_FILE)
        except pd.errors.EmptyDataError:
            old_records = pd.DataFrame()
    else:
        old_records = pd.DataFrame()

    new_record = pd.DataFrame([record])

    updated_history = pd.concat(
        [old_records, new_record],
        ignore_index=True
    )

    updated_history.to_csv(
        HISTORY_FILE,
        index=False
    )


    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = load_prediction_history()

def generate_pdf_report(predicted_class, confidence, uploaded_image):

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    story = []

    # Project Title
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#16803C"),
        fontSize=24
    )

    story.append(
        Paragraph("KrishiDrishti AI", title_style)
    )

    story.append(
        Paragraph(
            "Plant Disease Detection Report",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 20))

    # Uploaded Leaf Image
    story.append(
        Paragraph("Uploaded Leaf Image", styles["Heading2"])
    )

    story.append(Spacer(1, 10))

    image_buffer = BytesIO(uploaded_image.getvalue())
    image_buffer.seek(0)

    leaf_image = ReportLabImage(
        image_buffer,
        width=4 * inch,
        height=3 * inch,
        kind="proportional"
    )

    story.append(leaf_image)

    story.append(Spacer(1, 20))

    # Date and Time
    date_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    # Prediction Details
    data = [
        ["Parameter", "Prediction Details"],
        ["Date & Time", date_time],
        ["Detected Disease", predicted_class],
        ["AI Confidence", f"{confidence:.2f}%"]
    ]

    table = Table(data, colWidths=[160, 300])

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#16803C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1),
         colors.HexColor("#F0FDF4")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))

    story.append(table)

    story.append(Spacer(1, 25))

    # Disclaimer
    story.append(
        Paragraph(
            "<b>Disclaimer:</b> This is an AI-generated "
            "prediction. Model confidence does not guarantee "
            "diagnostic accuracy. Consult an agricultural "
            "expert before treatment decisions.",
            styles["Normal"]
        )
    )

    doc.build(story)

    buffer.seek(0)

    return buffer.getvalue()



# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="KrishiDrishti AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame(
        columns=[
            "Date & Time",
            "Crop",
            "Disease",
            "Confidence"
        ]
    )


# --------------------------------
# 2. CUSTOM CSS
# --------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #F4F8F1;
    color: #243B2A;
}

[data-testid="stSidebar"] {
    background-color: #E5F0E2;
}

/* SIDEBAR TEXT VISIBILITY FIX */

[data-testid="stSidebar"] * {
    color: #243B2A !important;
}

/* Sidebar headings */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #176B3A !important;
}

/* Sidebar paragraph and labels */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #243B2A !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #A8BEA5 !important;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #176B3A;
}

.subtitle {
    font-size: 18px;
    color: #52705B;
}

.section-title {
    color: #176B3A;
    font-size: 25px;
    font-weight: 700;
}

div.stButton > button {
    background-color: #238B45;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 25px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #176B3A;
    color: white;
}

[data-testid="stMetric"] {
    background-color: #E5F0E2;
    padding: 18px;
    border-radius: 12px;
}

/* Fix metric label color */
[data-testid="stMetricLabel"] {
    color: #176B3A !important;
}

[data-testid="stMetricLabel"] p {
    color: #176B3A !important;
    font-weight: 600 !important;
}

/* Fix metric value color */
[data-testid="stMetricValue"] {
    color: #176B3A !important;
    font-weight: 800 !important;
}

/* Fix metric delta color */
[data-testid="stMetricDelta"] {
    color: #176B3A !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# 3. SIDEBAR
# --------------------------------

with st.sidebar:

    st.title("🌱 KrishiDrishti AI")

    st.write("Smart Farming | AI Technology")

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "ℹ️ About Project"]
    )

    st.divider()

    st.caption("Supported Crops")

    st.write("🍅 Tomato")
    st.write("🥔 Potato")

    st.divider()

    st.caption("Powered by MobileNetV2 CNN")


# --------------------------------
# 4. DASHBOARD
# --------------------------------

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🌱 KrishiDrishti AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-Powered Plant Disease Detection System</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Identify possible diseases in Tomato and Potato leaves "
        "using Deep Learning."
    )

    st.divider()

    # Dashboard metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Supported Crops", "02")
    col2.metric("Disease Classes", "06")
    col3.metric("Model Accuracy", "97.60%")

    st.divider()

    # Image upload section
    st.markdown(
        '<div class="section-title">📷 Upload Leaf Image</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload a clear photograph of a Tomato or Potato leaf."
    )

    uploaded_file = st.file_uploader(
        "Choose your leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")


            # --------------------------------
            # STEP 14 - IMAGE QUALITY CHECK
            # --------------------------------

        width, height = image.size

        st.subheader("🖼️ Image Quality Check")

        if width < 224 or height < 224:
            st.warning(
                    "Image resolution is low. "
                    "Please upload a clearer image of at least 224 × 224 pixels."
                )
        else:
            st.success(
                    f"Image resolution looks sufficient: {width} × {height}"
                )

            # --------------------------------
# IMAGE UPLOAD ERROR HANDLING
# --------------------------------

if uploaded_file is not None:

    if uploaded_file.type not in ["image/jpeg", "image/png", "image/jpg"]:
        st.error("Invalid image format! Please upload JPG or PNG image.")

    else:
        st.success("Image uploaded successfully!")


        left, right = st.columns(2)

        with left:
            st.subheader("Uploaded Image")
            st.image(
                image,
                caption="Leaf Photograph",
                width="stretch"
            )

        with right:
            st.subheader("Image Information")
            st.write("Format:", uploaded_file.type)
            st.write("Dimensions:", image.size)
            st.write("Color mode:", image.mode)

            st.divider()

    if st.button("🔍 Detect Plant Disease", type="primary"):

        with st.spinner("AI is analyzing your leaf..."):
            predicted_class, confidence = predict_disease(image)

            # Store prediction history
            prediction_record = {
                "Date & Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "Crop": predicted_class.split("___")[0],
                "Disease": predicted_class.replace("___", " - ").replace("_", " "),
                "Confidence": round(confidence, 2)
            }
            save_prediction_history(prediction_record)
            # --------------------------------
# STEP 15.3 - PREDICTION HISTORY ERROR HANDLING
# --------------------------------

        def save_prediction_history(record):

            try:
                    history_df = pd.DataFrame([record])

                    if os.path.exists(HISTORY_FILE):
                        history_df.to_csv(
                            HISTORY_FILE,
                            mode="a",
                            header=False,
                            index=False
            )
                    else:
                        history_df.to_csv(
                            HISTORY_FILE,
                            index=False
            )

            except Exception as e:
                st.error(f"Unable to save prediction history: {e}")

            
            st.session_state.prediction_history = pd.concat(
                [
            st.session_state.prediction_history,
            pd.DataFrame([prediction_record])
            ],
            ignore_index=True
            )

            disease_name = (
                predicted_class
                .replace("___", " - ")
                .replace("_", " ")
            )

            st.divider()

            st.markdown(
                '<div class="section-title">📊 Prediction Results</div>',
                unsafe_allow_html=True
            )

            st.success(
                f"Detected Disease: {disease_name}"
            )

            result1, result2 = st.columns(2)

            with result1:
                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

            with result2:
                st.metric(
                    "Predicted Crop",
                    predicted_class.split("___")[0]
                )

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

            # --------------------------------
            # DISEASE INFORMATION
            # --------------------------------

            disease_information = {

                "Potato___Early_blight": {
                    "description": "Early blight is a common potato disease caused by Alternaria fungi.",
                    "symptoms": [
                        "Brown spots on older leaves.",
                        "Concentric circular rings inside leaf spots.",
                        "Yellowing around infected areas."
                    ],
                    "management": [
                        "Remove severely infected leaves.",
                        "Practice crop rotation.",
                        "Use mulch to reduce soil splash.",
                        "Maintain proper spacing and airflow.",
                        "Avoid overhead irrigation."
                    ]
                },

                "Potato___Late_blight": {
                    "description": "Late blight is a destructive potato disease caused by Phytophthora infestans.",
                    "symptoms": [
                        "Water-soaked patches on leaves.",
                        "Dark brown or black lesions.",
                        "White growth underneath leaves in humid conditions.",
                        "Rapid spreading of infected areas."
                    ],
                    "management": [
                        "Remove and safely dispose of infected plants.",
                        "Use certified disease-free seed potatoes.",
                        "Avoid excessive moisture on leaves.",
                        "Inspect plants regularly.",
                        "Consult an agricultural expert immediately if suspected."
                    ]
                },

                "Potato___healthy": {
                    "description": "The potato leaf is classified as healthy by the AI model.",
                    "symptoms": [
                        "No visible disease symptoms detected by the model.",
                        "Leaf appearance is consistent with a healthy class."
                    ],
                    "management": [
                        "Continue regular crop monitoring.",
                        "Maintain balanced irrigation.",
                        "Use healthy planting material.",
                        "Keep the field free from weeds."
                    ]
                },

                "Tomato___Early_blight": {
                    "description": "Tomato early blight is a fungal disease caused mainly by Alternaria species.",
                    "symptoms": [
                        "Dark brown spots on older leaves.",
                        "Target-like concentric rings.",
                        "Yellowing around infected spots.",
                        "Premature drying and falling of leaves."
                    ],
                    "management": [
                        "Remove infected leaves carefully.",
                        "Practice crop rotation.",
                        "Use mulch around plants.",
                        "Maintain proper plant spacing.",
                        "Use drip irrigation where possible."
                    ]
                },

                "Tomato___Late_blight": {
                    "description": "Tomato late blight is a rapidly spreading disease caused by Phytophthora infestans.",
                    "symptoms": [
                        "Water-soaked leaf patches.",
                        "Dark brown or black lesions.",
                        "White growth underneath leaves in humid weather.",
                        "Brown patches on fruits."
                    ],
                    "management": [
                        "Remove and safely dispose of infected plant material.",
                        "Avoid overhead watering.",
                        "Maintain good airflow between plants.",
                        "Monitor plants frequently.",
                        "Seek agricultural expert guidance quickly."
                    ]
                },

                "Tomato___healthy": {
                    "description": "The tomato leaf is classified as healthy by the AI model.",
                    "symptoms": [
                        "No visible disease symptoms detected by the model.",
                        "Leaf appearance is consistent with a healthy class."
                    ],
                    "management": [
                        "Continue regular plant inspection.",
                        "Maintain proper watering.",
                        "Provide adequate sunlight.",
                        "Maintain good field sanitation."
                    ]
                }
            }

            # Get information for predicted class
            info = disease_information[predicted_class]

            st.divider()

            # Disease description
            st.subheader("🌿 About This Prediction")
            st.write(info["description"])

            # Symptoms
            st.subheader("🔍 Common Symptoms")

            for symptom in info["symptoms"]:
                st.write("•", symptom)

            # Management
            st.subheader("🛡️ Prevention and Management")

            for recommendation in info["management"]:
                st.write("•", recommendation)

            st.warning(
                "These are general educational recommendations. "
                "AI predictions are not laboratory diagnoses. "
                "Confirm suspected disease with a local agricultural "
                "expert before applying pesticides or other treatments."
            )

            # --------------------------------
            # DOWNLOAD DISEASE REPORT
            # --------------------------------

            report = f"""
KRISHIDRISHTI AI
PLANT DISEASE DETECTION REPORT

Date: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}

Crop: {predicted_class.split("___")[0]}

Predicted Disease:
{disease_name}

Model Confidence:
{confidence:.2f}%

ABOUT THE DISEASE:
{info["description"]}

COMMON SYMPTOMS:
"""

            for symptom in info["symptoms"]:
                report += f"\n- {symptom}"

            report += "\n\nPREVENTION AND MANAGEMENT:\n"

            for recommendation in info["management"]:
                report += f"\n- {recommendation}"

            report += """

DISCLAIMER:
This is an AI-generated prediction.
Confirm the result with an agricultural expert
before taking treatment decisions.
"""

            st.markdown("""
              <style>
              div.st-key-pdf_report button {
                  background-color: #16803C;
                  color: white;
                  border: none;
                  border-radius: 8px;
                  font-weight: bold;
              }
  
              div.st-key-pdf_report button:hover {
                  background-color: #11632F;
                  color: white;
              }
              </style>
              """, unsafe_allow_html=True)
  
            with st.container(key="txt_report"):
                st.download_button(
            label="📥 Download Disease Report",
            data=report,
            file_name="KrishiDrishti_Disease_Report.txt",
            mime="text/plain"
            )



            # PDF Disease Report

            pdf_file = generate_pdf_report(
                predicted_class,
                confidence,
                uploaded_file
            )

            st.markdown("""
            <style>
            div.st-key-pdf_report button {
                background-color: #16803C;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
            }

            div.st-key-pdf_report button:hover {
                background-color: #11632F;
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)

            with st.container(key="pdf_report"):
                st.download_button(
                    label="📥 Download Disease Report (PDF)",
                    data=pdf_file,
                    file_name="KrishiDrishti_Disease_Report.pdf",
                    mime="application/pdf"
                )

            st.info(
                "This is an AI-generated prediction. "
                "Please consult an agricultural expert "
                "before making treatment decisions."
            )


    else:
        st.info("Upload a leaf image to start disease detection.")

    # --------------------------------
    # PREDICTION HISTORY
    # --------------------------------

    st.divider()
    st.subheader("📋 Prediction History")

    history_df = load_prediction_history()
    
# Farmer Analytics Dashboard

    st.divider()
    st.header("📊 Farmer Analytics Dashboard")

    analytics = calculate_analytics(history_df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Predictions",
            analytics["total"]
       )

    with col2:
        st.metric(
            "Healthy Leaves",
            analytics["healthy"]
        )

    with col3:
        st.metric(
            "Diseased Leaves",
            analytics["diseased"]
        )

    with col4:
        st.metric(
            "Average Confidence",
            f'{analytics["average_confidence"]}%'
        )

    if not history_df.empty:

        st.subheader("🏆 Most Detected Disease")
        st.info(analytics["most_detected"])

        
        st.subheader("📈 Disease Distribution")

        disease_counts = history_df["Disease"].value_counts().reset_index()
        disease_counts.columns = ["Disease", "Total Predictions"]

        fig = px.bar(
            disease_counts,
            x="Disease",
           y="Total Predictions",
    color="Disease",
    text="Total Predictions",
    title="Crop Disease Distribution",
    template="plotly_white"
)

        fig.update_layout(
        showlegend=False,
        xaxis_title="Disease Type",
        yaxis_title="Total Predictions",
        height=400
)

        fig.update_traces(
            textposition="outside",
            marker_line_width=0
      )

        st.plotly_chart(fig, use_container_width=True)

# Crop-wise Analysis

        st.divider()

        st.subheader("🌱 Crop-wise Prediction Analysis")

        crop_chart_df = history_df["Crop"].value_counts().reset_index()
        crop_chart_df.columns = ["Crop", "Total Predictions"]

        fig = px.bar(
            crop_chart_df,
            x="Crop",
            y="Total Predictions",
            color="Crop",
            text="Total Predictions",
            title="Crop-wise Prediction Analysis",
            template="plotly_white"
        )

        fig.update_layout(
            showlegend=False,
            xaxis_title="Crop",
            yaxis_title="Total Predictions",
            height=350
        )

        fig.update_traces(textposition="outside")

        st.plotly_chart(fig, use_container_width=True)
        crop_counts = history_df["Crop"].value_counts()

        st.subheader("📋 Crop Prediction Summary")

        st.dataframe(
        crop_counts.reset_index().rename(
        columns={
            "Crop": "Crop",
            "count": "Total Predictions"
        }
    ),
    use_container_width=True
)
    
        # --------------------------------
        # PREDICTION TRENDS OVER TIME
        # --------------------------------

        st.divider()

        st.subheader("📈 Prediction Trends Over Time")

        trend_df = history_df.copy()

        trend_df["Date & Time"] = pd.to_datetime(
            trend_df["Date & Time"],
            format="%d-%m-%Y %H:%M:%S",
            errors="coerce"
        )

        trend_df = trend_df.dropna(
            subset=["Date & Time"]
        )

        if not trend_df.empty:

            trend_df["Date"] = trend_df["Date & Time"].dt.date

            daily_predictions = (
                trend_df.groupby("Date")
                .size()
                .reset_index(name="Total Predictions")
            )

            fig = px.line(
    daily_predictions,
    x="Date",
    y="Total Predictions",
    markers=True,
    title="Prediction Trends Over Time",
    template="plotly_white"
)

            fig.update_traces(
    line=dict(width=3),
    marker=dict(size=8)
)

            fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Total Predictions",
    height=400,
    hovermode="x unified"
)

            st.plotly_chart(fig, use_container_width=True)
            

            st.metric(
                "Total Active Prediction Days",
                daily_predictions["Date"].nunique()
            )

        else:
            st.info("No valid date records available.")


    else:
        st.info("No prediction data available yet.")



    if not history_df.empty:
        # --------------------------------
# CROP-WISE CONFIDENCE ANALYSIS
# --------------------------------

        st.divider()
        st.subheader("🌱 Crop-wise Confidence Analysis")

        crop_confidence = (
            history_df.groupby("Crop")["Confidence"]
            .mean()
            .round(2)
    )

        confidence_chart_df = crop_confidence.reset_index()
        confidence_chart_df.columns = ["Crop", "Average Confidence"]

        fig = px.bar(
    confidence_chart_df,
    x="Crop",
    y="Average Confidence",
    color="Crop",
    text="Average Confidence",
    title="Crop-wise Average Confidence",
    template="plotly_white"
)

        fig.update_layout(
    showlegend=False,
    xaxis_title="Crop",
    yaxis_title="Average Confidence",
    height=350
)

        fig.update_traces(texttemplate="%{y:.2f}", textposition="outside")

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
    crop_confidence.reset_index().rename(
        columns={"Confidence": "Average Confidence"}
    ),
    use_container_width=True,
    hide_index=True
)
         
        # --------------------------------
        # AVERAGE CONFIDENCE BY DISEASE
        # --------------------------------

        st.divider()
        st.subheader("📊 Average Confidence by Disease")

        disease_df = history_df.copy()

        disease_df["Confidence"] = pd.to_numeric(
            disease_df["Confidence"],
            errors="coerce"
        )

        disease_avg = (
            disease_df.groupby("Disease")["Confidence"]
            .mean()
            .dropna()
            .round(2)
        )

        if not disease_avg.empty:

            disease_avg_df = disease_avg.reset_index()
            disease_avg_df.columns = [
                "Disease",
                "Average Confidence"
            ]

            fig = px.bar(
                disease_avg_df.sort_values("Average Confidence"),
                x="Average Confidence",
                y="Disease",
                orientation="h",
                text="Average Confidence",
                color="Average Confidence",
                color_continuous_scale=[
                    "#D6B3FF",
                    "#A855F7",
                    "#6D28D9"
                ],
                template="plotly_white",
                title="Disease-wise Model Confidence"
            )

            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside"
            )

            fig.update_layout(
                height=450,
                coloraxis_showscale=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(
                    title="Average Confidence (%)",
                    range=[0, 110],
                    showgrid=True,
                    gridcolor="#EEEEEE"
                ),
                yaxis=dict(
                    title="Disease",
                    showgrid=False
                ),
                margin=dict(l=20, r=60, t=65, b=35)
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key="average_confidence_by_disease"
            )

        else:
            st.info("No valid disease confidence data available.")
        
        # EXISTING CROP-WISE CONFIDENCE GRAPH
        st.plotly_chart(
            fig,
            use_container_width=True,
            key="confidence_distribution_chart"
      )

        # EXISTING CROP-WISE TABLE
        st.dataframe(
            crop_confidence.reset_index().rename(
                columns={"Confidence": "Average Confidence"}
            ),
            use_container_width=True,
            hide_index=True
        )

        # PASTE NEW AVERAGE CONFIDENCE BY DISEASE CODE HERE


        st.session_state.prediction_history = history_df


        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True
        )

        csv_data = history_df.to_csv(index=False)

        st.markdown("""

<style>
div.stDownloadButton > button {
    background-color: #28a745;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}

div.stDownloadButton > button:hover {
    background-color: #218838;
    color: white;
}
</style>
""", unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Prediction History",
            data=csv_data,
            file_name="KrishiDrishti_History.csv",
            mime="text/csv"
        )



        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            pd.DataFrame().to_csv(HISTORY_FILE, index=False)
            st.rerun()

    else:
        st.info("No prediction history available yet.")


    
# --------------------------------
# PREDICTION CONFIDENCE DISTRIBUTION
# --------------------------------

        st.divider()
        st.subheader("🎯 Model Confidence Analysis")

        if not history_df.empty:

            confidence_values = pd.to_numeric(
        history_df["Confidence"],
        errors="coerce"
    ).dropna()

            if not confidence_values.empty:

                avg_confidence = confidence_values.mean()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                "Average Confidence",
                f"{avg_confidence:.2f}%"
            )

                with col2:
                    st.metric(
                "Highest Confidence",
                f"{confidence_values.max():.2f}%"
            )

                st.subheader("📊 Confidence Distribution")

                confidence_df = pd.DataFrame({
            "Model Confidence (%)": confidence_values
        })

                fig = px.histogram(
            confidence_df,
            x="Model Confidence (%)",
            nbins=15,
            title="Prediction Confidence Distribution",
            template="plotly_white",
            color_discrete_sequence=["#A855F7"]
        )

                fig.update_traces(
            marker_line_color="#8B5CF6",
            marker_line_width=1,
            opacity=0.85
        )

                fig.update_layout(
            height=420,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Arial", size=13, color="#263238"),
            title_font=dict(size=20, color="#263238"),
            bargap=0.25,
            xaxis_title="Model Confidence (%)",
            yaxis_title="Number of Predictions",
            margin=dict(l=25, r=25, t=65, b=40),
            xaxis=dict(showgrid=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="#EEEEEE"
            )
        )

                st.plotly_chart(
            fig,
            use_container_width=True,
            key="confidence_distribution_chart"
        )

            else:
                st.info("No valid confidence values available.")

        else:
            st.info("No prediction data available for confidence analysis.")

        
    
# --------------------------------
# PREDICTION TRENDS OVER TIME
# --------------------------------

        st.divider()
        st.subheader("📅 Prediction Trends Over Time")

        if not history_df.empty and "Date & Time" in history_df.columns:

            trend_df = history_df.copy()

            trend_df["Date & Time"] = pd.to_datetime(
        trend_df["Date & Time"],
        format="%d-%m-%Y %H:%M:%S",
        errors="coerce"
    )

            trend_df = trend_df.dropna(subset=["Date & Time"])

            if not trend_df.empty:

                trend_df["Date"] = trend_df["Date & Time"].dt.date

                daily_predictions = (
                    trend_df.groupby("Date")
            .size()
            .reset_index(name="Total Predictions")
        )

                fig = px.line(
            daily_predictions,
            x="Date",
            y="Total Predictions",
            markers=True,
            title="Daily Crop Disease Predictions",
            template="plotly_white"
        )

                fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Predictions",
            hovermode="x unified"
        )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No valid date records available.")

        else:
            st.info("No prediction history available.")



    
# --------------------------------
# CROP-WISE CONFIDENCE ANALYSIS
# --------------------------------

        st.divider()
        st.subheader("🌱 Crop-wise Confidence Analysis")

        crop_confidence = (
        history_df.groupby("Crop")["Confidence"]
    .mean()
    .round(2)
)

        confidence_chart_df = crop_confidence.reset_index()
        confidence_chart_df.columns = ["Crop", "Average Confidence"]

        fig = px.bar(
            confidence_chart_df,
            x="Crop",
            y="Average Confidence",
            color="Crop",
            text="Average Confidence",
            title="Crop-wise Average Confidence",
            template="plotly_white",
            color_discrete_sequence=["#A855F7", "#C4B5FD"]
        )

        fig.update_traces(
    texttemplate="%{y:.2f}%",
    textposition="outside",
    marker_line_width=0
        )

        fig.update_layout(
    height=420,
    showlegend=False,
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(family="Arial", size=13, color="#263238"),
    title_font=dict(size=20, color="#263238"),
    xaxis_title="Crop",
    yaxis_title="Average Confidence (%)",
    bargap=0.35,
    margin=dict(l=25, r=25, t=65, b=40),
    yaxis=dict(
        showgrid=True,
        gridcolor="#EEEEEE",
        zeroline=False
    )
)

        st.plotly_chart(
    fig,
    use_container_width=True,
    key="crop_confidence_analysis"
)

        st.dataframe(
    confidence_chart_df,
    use_container_width=True,
    hide_index=True
)


    # --------------------------------
    # HEALTHY VS DISEASED ANALYSIS
    # --------------------------------

    st.divider()

    st.subheader("🥧 Healthy vs Diseased Analysis")

    if not history_df.empty:

        valid_diseases = [
            "Potato Early Blight",
            "Potato Late Blight",
            "Potato Healthy",
            "Tomato Early Blight",
            "Tomato Late Blight",
            "Tomato Healthy"
        ]

        valid_history = history_df[
            history_df["Disease"].isin(valid_diseases)
        ]

        healthy_count = valid_history["Disease"].str.contains(
            "healthy",
            case=False,
            na=False
        ).sum()

        diseased_count = len(valid_history) - healthy_count

        total_predictions = len(valid_history)

        if total_predictions > 0:

            healthy_percentage = (
                healthy_count / total_predictions
            ) * 100

            diseased_percentage = (
                diseased_count / total_predictions
            ) * 100

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Healthy Leaves",
                    f"{healthy_count} ({healthy_percentage:.2f}%)"
                )

            with col2:
                st.metric(
                    "Diseased Leaves",
                    f"{diseased_count} ({diseased_percentage:.2f}%)"
                )

            comparison_df = pd.DataFrame({
                "Category": ["Healthy", "Diseased"],
                "Percentage": [
                    healthy_percentage,
                    diseased_percentage
                ]
            })

            st.bar_chart(
                comparison_df.set_index("Category")
            )

        else:
            st.info("No valid disease predictions available.")

    else:
        st.info("No prediction data available yet.")


    # --------------------------------
    # DISEASE-WISE CONFIDENCE COMPARISON
    # --------------------------------

    st.divider()

    st.subheader("🔍 Disease-wise Confidence Comparison")

    if not history_df.empty:

        disease_confidence = history_df.copy()

        disease_confidence["Confidence"] = pd.to_numeric(
            disease_confidence["Confidence"],
            errors="coerce"
        )

        disease_confidence = disease_confidence.dropna(
            subset=["Confidence"]
        )

        if not disease_confidence.empty:

            disease_avg = (
                disease_confidence.groupby("Disease")["Confidence"]
                .mean()
                .round(2)
                .sort_values(ascending=False)
            )

            
            
# --------------------------------
# AVERAGE CONFIDENCE BY DISEASE
# --------------------------------


    st.subheader("📊 Average Confidence by Disease")

    disease_df = history_df.copy()

    disease_df["Confidence"] = pd.to_numeric(
        disease_df["Confidence"],
        errors="coerce"
    )

    disease_avg = (
    disease_df.groupby("Disease")["Confidence"]
    .mean()
    .dropna()
    .round(2)
)

    if not disease_avg.empty:

        disease_avg_df = disease_avg.reset_index()
        disease_avg_df.columns = [
        "Disease",
        "Average Confidence"
    ]

        fig = px.bar(
            disease_avg_df,
            x="Disease",
            y="Average Confidence",
            color="Disease",
            text="Average Confidence",
            title="Average Model Confidence for Each Disease",
            template="plotly_white",
            color_discrete_sequence=[
            "#214297",
            "#5333D3",
            "#142794",
            "#2B1DC5",
            "#332089",
            "#0717A5"
        ]
    )

        fig.update_traces(
        texttemplate="%{y:.2f}%",
        textposition="outside",
        marker_line_width=0
    )

        fig.update_layout(
        height=450,
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=13, color="#263238"),
        title_font=dict(size=20, color="#263238"),
        xaxis_title="Disease Type",
        yaxis_title="Average Confidence (%)",
        bargap=0.35,
        margin=dict(l=25, r=25, t=65, b=80),
        xaxis=dict(
            showgrid=False,
            tickangle=-25
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#EEEEEE",
            zeroline=False
        )
    )

        st.plotly_chart(
        fig,
        use_container_width=True,
        key="average_confidence_by_disease_2"
    )

    else:
        st.info("No valid disease confidence data available.")



    # --------------------------------
    #  CONFIDENCE DISTRIBUTION
    # --------------------------------

    st.divider()
    st.subheader("📊 Prediction Confidence Distribution")

    if not history_df.empty:

        confidence_df = history_df.copy()

        confidence_df["Confidence"] = pd.to_numeric(
            confidence_df["Confidence"],
            errors="coerce"
        )

        confidence_df = confidence_df.dropna(
            subset=["Confidence"]
        )

        if not confidence_df.empty:

            confidence_df["Confidence Level"] = confidence_df[
                "Confidence"
            ].apply(
                lambda x: "High" if x >= 80
                else "Medium" if x >= 50
                else "Low"
            )

            level_counts = confidence_df[
                "Confidence Level"
            ].value_counts().reindex(
                ["High", "Medium", "Low"],
                fill_value=0
            )

            st.bar_chart(level_counts)

            st.subheader("📋 Confidence Level Summary")

            st.dataframe(
                level_counts.reset_index().rename(
                    columns={
                        "Confidence Level": "Category",
                        "count": "Total Predictions"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No valid confidence values available.")

    else:
        st.info("No prediction history available.")


   
    # --------------------------------
    # STEP 9 - DISEASE RISK INDICATOR
    # --------------------------------

    st.divider()
    st.subheader("🚨 Disease Risk Indicator")

    if not history_df.empty:

        risk_df = history_df.copy()

        # Calculate Disease Risk
        def calculate_risk(disease):

            disease = str(disease).lower()

            if "healthy" in disease:
                return "Low Risk"

            elif "late blight" in disease or "late_blight" in disease:
                return "High Risk"

            elif "early blight" in disease or "early_blight" in disease:
                return "Medium Risk"

            else:
                return "Unknown"

        risk_df["Risk Level"] = risk_df["Disease"].apply(
            calculate_risk
        )

        # Risk Distribution
        risk_order = [
            "Low Risk",
            "Medium Risk",
            "High Risk",
            "Unknown"
        ]

        risk_counts = (
            risk_df["Risk Level"]
            .value_counts()
            .reindex(risk_order, fill_value=0)
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Total Predictions"
        ]

        # Risk Distribution Graph
        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Total Predictions",
            color="Risk Level",
            text="Total Predictions",
            color_discrete_map={
                "Low Risk": "#00C853",
                "Medium Risk": "#FFA500",
                "High Risk": "#FF4B4B",
                "Unknown": "#9E9E9E"
            },
            title="🚨 Disease Risk Distribution",
            template="plotly_white"
        )

        fig.update_traces(
            textposition="outside",
            marker_line_width=0,
            textfont=dict(
                size=14,
                color="#263238"
            )
        )

        fig.update_layout(
            height=420,
            title_font=dict(
                size=21,
                color="#263238"
            ),
            font=dict(
                family="Arial",
                size=13
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            bargap=0.35,
            margin=dict(
                l=25,
                r=25,
                t=65,
                b=40
            ),
            xaxis_title="Disease Risk Level",
            yaxis_title="Number of Predictions",
            yaxis=dict(
                showgrid=True,
                gridcolor="#EAEAEA",
                zeroline=False
            ),
            xaxis=dict(
                showgrid=False
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Risk Summary Table
        st.subheader("📋 Risk Summary")

        display_risk_df = risk_df[
            ["Crop", "Disease", "Risk Level"]
        ].tail(10).copy()

        st.dataframe(
            display_risk_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No prediction data available yet.")



    # --------------------------------
    # STEP 10 - DISEASE TREATMENT GUIDE
    # --------------------------------

    st.divider()
    st.subheader("💊 Disease Treatment Guide")

    if not history_df.empty:

        disease_guide = {
            "healthy": [
                "Continue regular crop monitoring.",
                "Maintain balanced irrigation.",
                "Use healthy planting material."
            ],

            "early blight": [
                "Remove infected leaves carefully.",
                "Practice crop rotation.",
                "Maintain proper plant spacing.",
                "Avoid overhead irrigation."
            ],

            "late blight": [
                "Remove and safely dispose of infected plants.",
                "Avoid excessive moisture on leaves.",
                "Inspect crops regularly.",
                "Consult an agricultural expert."
            ]
        }

        selected_disease = st.selectbox(
            "Select Disease",
            sorted(history_df["Disease"].dropna().unique()),
            key="treatment_disease"
        )

        disease_key = selected_disease.lower()

        if "healthy" in disease_key:
            recommendations = disease_guide["healthy"]

        elif "late blight" in disease_key:
            recommendations = disease_guide["late blight"]

        elif "early blight" in disease_key:
            recommendations = disease_guide["early blight"]

        else:
            recommendations = [
                "Consult an agricultural expert for guidance."
            ]

        st.write("### 🌿 Recommended Management")

        for recommendation in recommendations:
            st.write("✅", recommendation)

        st.warning(
            "These are general recommendations. "
            "Confirm the disease with an agricultural expert "
            "before applying any pesticide or treatment."
        )

    else:
        st.info("No disease history available.")


    # --------------------------------
    # PREVENTION RECOMMENDATIONS
    # --------------------------------

    st.divider()
    st.subheader("🛡️ Crop Disease Prevention")

    if not history_df.empty:

        prevention_guide = {
            "healthy": [
                "Inspect crops regularly.",
                "Maintain balanced irrigation.",
                "Use disease-free seeds.",
                "Keep the field clean."
            ],

            "early blight": [
                "Practice crop rotation.",
                "Remove infected plant debris.",
                "Avoid overhead irrigation.",
                "Maintain proper plant spacing."
            ],

            "late blight": [
                "Use certified disease-free seeds.",
                "Avoid prolonged leaf wetness.",
                "Monitor crops during humid weather.",
                "Remove infected plant material safely."
            ]
        }

        selected_crop_disease = st.selectbox(
            "Choose Disease for Prevention Tips",
            sorted(history_df["Disease"].dropna().unique()),
            key="prevention_disease"
        )

        disease_name = selected_crop_disease.lower()

        if "healthy" in disease_name:
            tips = prevention_guide["healthy"]

        elif "late blight" in disease_name:
            tips = prevention_guide["late blight"]

        elif "early blight" in disease_name:
            tips = prevention_guide["early blight"]

        else:
            tips = [
                "Consult an agricultural expert for disease-specific prevention."
            ]

        st.write("### 🌿 Recommended Preventive Measures")

        for tip in tips:
            st.write("✅", tip)

    else:
        st.info("No prediction history available yet.")


    # --------------------------------
    #  DOWNLOAD ANALYTICS PDF
    # --------------------------------

    st.divider()
    st.subheader("📄 Download Farmer Analytics Report")

    if not history_df.empty:

        pdf_buffer = BytesIO()

        pdf = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4
        )

        styles = getSampleStyleSheet()

        elements = []

        elements.append(
            Paragraph(
                "KrishiDrishti AI - Farmer Analytics Report",
                styles["Title"]
            )
        )

        elements.append(Spacer(1, 20))

        total = len(history_df)

        healthy = history_df["Disease"].str.contains(
            "healthy",
            case=False,
            na=False
        ).sum()

        diseased = total - healthy

        avg_confidence = pd.to_numeric(
            history_df["Confidence"],
            errors="coerce"
        ).mean()

        summary_data = [
            ["Metric", "Value"],
            ["Total Predictions", str(total)],
            ["Healthy Leaves", str(healthy)],
            ["Diseased Leaves", str(diseased)],
            ["Average Confidence", f"{avg_confidence:.2f}%"]
        ]

        elements.append(
            Paragraph("Prediction Summary", styles["Heading2"])
        )

        elements.append(
            Table(summary_data, hAlign="LEFT")
        )

        elements.append(Spacer(1, 20))

        crop_counts = history_df["Crop"].value_counts()

        crop_data = [["Crop", "Total Predictions"]]

        for crop, count in crop_counts.items():
            crop_data.append([str(crop), str(count)])

        elements.append(
            Paragraph("Crop-wise Analysis", styles["Heading2"])
        )

        elements.append(Table(crop_data, hAlign="LEFT"))

        elements.append(Spacer(1, 20))

        disease_counts = history_df["Disease"].value_counts()

        disease_data = [["Disease", "Total Predictions"]]

        for disease, count in disease_counts.items():
            disease_data.append([str(disease), str(count)])

        elements.append(
            Paragraph("Disease-wise Analysis", styles["Heading2"])
        )

        elements.append(Table(disease_data, hAlign="LEFT"))

        elements.append(Spacer(1, 20))

        elements.append(
            Paragraph("Recent Prediction History", styles["Heading2"])
        )

        recent_data = [
            ["Crop", "Disease", "Confidence"]
        ]

        for _, row in history_df.tail(10).iterrows():

            recent_data.append([
                str(row["Crop"]),
                str(row["Disease"]),
                f'{float(row["Confidence"]):.2f}%'
            ])

        elements.append(Table(recent_data, hAlign="LEFT"))

        pdf.build(elements)

        st.download_button(
            label="📥 Download Analytics PDF",
            data=pdf_buffer.getvalue(),
            file_name="KrishiDrishti_Analytics_Report.pdf",
            mime="application/pdf"
        )

    else:
        st.info("No prediction history available.")


    # --------------------------------
    # STEP 13 - PREDICTION HISTORY SEARCH
    # --------------------------------

    st.divider()
    st.subheader("🔍 Search Prediction History")

    if not history_df.empty:

        search_query = st.text_input(
            "Search Crop or Disease",
            placeholder="Enter crop or disease name..."
        )

        filtered_history = history_df.copy()

        if search_query:

            filtered_history = filtered_history[
                filtered_history["Crop"].str.contains(
                    search_query,
                    case=False,
                    na=False
                )
                |
                filtered_history["Disease"].str.contains(
                    search_query,
                    case=False,
                    na=False
                )
            ]

        st.write(
            f"Total Matching Predictions: {len(filtered_history)}"
        )

        st.dataframe(
            filtered_history,
            use_container_width=True,
            hide_index=True
        )

        if filtered_history.empty:
            st.warning("No matching predictions found.")

    else:
        st.info("No prediction history available yet.")



    # --------------------------------
    # RECENT PREDICTION ACTIVITY
    # --------------------------------

    st.divider()

    st.subheader("🕒 Recent Prediction Activity")

    if not history_df.empty:

        recent_df = history_df.copy()

        recent_df["Date & Time"] = pd.to_datetime(
            recent_df["Date & Time"],
            format="%d-%m-%Y %H:%M:%S",
            errors="coerce"
        )

        recent_df = recent_df.dropna(
            subset=["Date & Time"]
        )

        recent_df = recent_df.sort_values(
            by="Date & Time",
            ascending=False
        )

        st.dataframe(
            recent_df.head(5),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No recent predictions available.")

# --------------------------------
# FARMER ANALYTICS DASHBOARD
# --------------------------------

    st.divider()
    st.header("📊 Farmer Analytics Dashboard")

    if not history_df.empty:

        total_predictions = len(history_df)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
            "Total Predictions",
            total_predictions
            )

        with col2:
            st.metric(
            "Total Crops",
            history_df["Crop"].nunique()
        )

        with col3:
            st.metric(
            "Different Diseases",
            history_df["Disease"].nunique()
        )
            
        
    # --------------------------------
    # HEALTHY VS DISEASED SUMMARY
    # --------------------------------

    st.divider()
    st.subheader("🌿 Healthy vs Diseased Summary")

    if not history_df.empty:

        healthy_count = history_df["Disease"].apply(
            lambda x: "healthy" in str(x).lower()
        ).sum()

        diseased_count = len(history_df) - healthy_count

        summary_df = pd.DataFrame({
            "Crop Condition": ["Healthy", "Diseased"],
            "Total Predictions": [
                healthy_count,
                diseased_count
            ]
        })

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🟢 Healthy Predictions",
                healthy_count
            )

        with col2:
            st.metric(
                "🔴 Diseased Predictions",
                diseased_count
            )

        fig = px.pie(
            summary_df,
            names="Crop Condition",
            values="Total Predictions",
            hole=0.55,
            color="Crop Condition",
            color_discrete_map={
                "Healthy": "#00C853",
                "Diseased": "#FF4B4B"
            },
            title="Healthy vs Diseased Crop Distribution",
            template="plotly_white"
        )

        fig.update_traces(
            textinfo="percent+value",
            textposition="inside",
            marker=dict(
                line=dict(
                    color="white",
                    width=3
                )
            ),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Predictions: %{value}<br>"
                "Percentage: %{percent}"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            height=430,
            title_font=dict(
                size=21,
                color="#263238"
            ),
            font=dict(
                family="Arial",
                size=14
            ),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(
                l=25,
                r=25,
                t=65,
                b=30
            ),
            legend_title_text="Crop Condition"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info("No prediction data available yet.")





# --------------------------------
# 5. ABOUT PROJECT
# --------------------------------

elif page == "ℹ️ About Project":

    st.title("About KrishiDrishti AI")

    st.write(
        "KrishiDrishti AI is an AI-powered plant disease "
        "detection project designed to help identify "
        "diseases in Tomato and Potato leaves."
    )

    st.subheader("Technology Used")

    st.write("• Python")
    st.write("• TensorFlow and Keras")
    st.write("• MobileNetV2")
    st.write("• Convolutional Neural Network (CNN)")
    st.write("• Streamlit")

    st.subheader("Supported Diseases")

    for name in [
        "Potato Early Blight",
        "Potato Late Blight",
        "Potato Healthy",
        "Tomato Early Blight",
        "Tomato Late Blight",
        "Tomato Healthy"
    ]:
        st.write("🌿", name)


# --------------------------------
# 6. FOOTER
# --------------------------------

st.divider()

st.caption(
    "© 2026 KrishiDrishti AI | AI-Based Plant Disease Detection"
)
