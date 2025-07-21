import pandas as pd
import numpy as np
import pymysql
import pickle
import boto3
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, accuracy_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import json
import streamlit as st
import io
import seaborn as sns
import plotly.express as px
import base64

EXPLAINABILITY_LOG = "delay_reason_tags.csv"
SOURCE_TABLE = "time_analytics_model"
SECONDARY_TABLE = "time_model"

# Only fetch latest 6 months
SIX_MONTHS_AGO = (datetime.today() - timedelta(days=180)).date()

# -------------------------#
# LOAD DB CONFIG
# -------------------------#
def load_db_config():
    with open('db_config.json', 'r') as f:
        return json.load(f)

# -------------------------#
# LOAD DATA FROM RDS
# -------------------------#
def load_data_from_rds(table_name, date_col_filter=None):
    config = load_db_config()
    connection = pymysql.connect(**config, cursorclass=pymysql.cursors.DictCursor)
    try:
        if table_name == SECONDARY_TABLE and date_col_filter:
            query = f"SELECT * FROM {table_name} WHERE `{date_col_filter}` >= '{SIX_MONTHS_AGO}'"
        else:
            query = f"SELECT * FROM {table_name}"

        with connection.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
        return pd.DataFrame(result)
    finally:
        connection.close()

# -------------------------#
# FEATURE ENGINEERING
# -------------------------#
def prepare_features(df):
    df = df.copy()
    cat_cols = ['task_bucket', 'status_name', 'client_type', 'industry_type']
    df = df.dropna(subset=cat_cols)

    encoder = ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ], remainder='passthrough')

    df['task_complexity'] = df[['task_level_billable_time', 'task_hours_variance']].sum(axis=1)
    df['user_overload_score'] = df['total_time_spent']
    df['dependency_delay_score'] = df['delay_days'].fillna(0)

    features = [
        'task_complexity', 'user_overload_score', 'dependency_delay_score',
        'task_bucket', 'status_name', 'client_type', 'industry_type',
        'realization_rate', 'production_rate'
    ]

    df = df.dropna(subset=features + ['is_delayed'])
    X_raw = df[features]
    y = df['is_delayed']

    # pipeline = Pipeline(steps=[
    #     ('preprocessor', encoder),
    #     ('model', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    # ])
    with open('delay_model.pkl', 'rb') as file:
        pipeline = pickle.load(file)

    return pipeline, X_raw, y, df

# -------------------------#
# TRAIN & EVALUATE MODEL
# -------------------------#
def train_model(pipeline, X, y):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    # pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print(f"Model Precision: {precision_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    return pipeline, X_test, y_test

# -------------------------#
# TAG DELAY REASONS
# -------------------------#
def tag_delay_reason(row):
    reasons = []
    if row['task_complexity'] > 15:
        reasons.append("Complex Task")
    if row['user_overload_score'] > 20:
        reasons.append("User Overload")
    if row['dependency_delay_score'] > 2:
        reasons.append("Dependency Delays")
    return reasons if reasons else ["Unclassified"]

def add_reason_tags(df):
    df['delay_reason_tags'] = df.apply(tag_delay_reason, axis=1)
    df[['task_id', 'user_involved', 'is_delayed', 'delay_reason_tags']].to_csv(EXPLAINABILITY_LOG, index=False)
    print(f"Delay reason tags saved to {EXPLAINABILITY_LOG}")

# -------------------------#
# STREAMLIT DASHBOARD
# -------------------------#
def launch_dashboard(df, model=None, X_test=None, y_test=None):
    st.set_page_config(page_title="Time-Based Insights", layout="wide")
    st.title("🕒 Time-Based Insights Dashboard")
    st.markdown("---")

    # Sidebar filters
    st.sidebar.header("Filters")
    df['completed_at'] = pd.to_datetime(df['completed_at'], errors='coerce')
    min_date = df['completed_at'].min()
    max_date = df['completed_at'].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['completed_at'] >= pd.Timestamp(start_date)) & (df['completed_at'] <= pd.Timestamp(end_date))]

    if 'status_name' in df.columns:
        selected_status = st.sidebar.multiselect("Task Status", options=df['status_name'].unique(), default=list(df['status_name'].unique()))
        df = df[df['status_name'].isin(selected_status)]

    if 'delay_reason_tags' in df.columns:
        all_tags = sorted(set(sum(df['delay_reason_tags'].tolist(), [])))
        selected_reasons = st.sidebar.multiselect("Delay Reasons", options=all_tags, default=all_tags)
        df = df[df['delay_reason_tags'].apply(lambda tags: any(tag in selected_reasons for tag in tags))]

    # Model Evaluation
    if model and X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        st.subheader("📊 Model Evaluation Metrics")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [
                accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred),
                recall_score(y_test, y_pred),
                f1_score(y_test, y_pred)
            ]
        })
        st.dataframe(metrics_df.set_index("Metric").style.format("{:.2%}"))
        st.download_button("Download Metrics CSV", metrics_df.to_csv(index=False), "model_metrics.csv", "text/csv")

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(pd.DataFrame(cm, index=["Not Delayed", "Delayed"], columns=["Not Delayed", "Delayed"]),
                           text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
        st.plotly_chart(fig_cm)

    # Pie charts for categories
    st.subheader("📊 Category Distributions")
    for col in ['client_type', 'task_bucket']:
        if col in df.columns:
            fig = px.pie(df, names=col, title=f"{col.title()} Distribution")
            st.plotly_chart(fig)

    # Delay reason frequency
    if 'delay_reason_tags' in df.columns:
        reason_counts = pd.Series(sum(df['delay_reason_tags'].tolist(), [])).value_counts()
        fig_bar = px.bar(reason_counts, x=reason_counts.index, y=reason_counts.values,
                         labels={'x': 'Delay Reason', 'y': 'Count'},
                         title="🗂 Delay Reason Frequency")
        st.plotly_chart(fig_bar)

    # Paginated Task Table
    st.subheader("📄 Delay Detail Table")
    page_size = 1000
    total_rows = len(df)
    total_pages = (total_rows // page_size) + 1
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    df_safe = df.iloc[start_idx:end_idx].copy()
    for col in df_safe.columns:
        if df_safe[col].dtype == 'object':
            df_safe[col] = df_safe[col].astype(str)
    st.dataframe(df_safe)
    st.download_button("Download Filtered Data CSV", df.to_csv(index=False), "filtered_tasks.csv", "text/csv")

    # Time Trend from Secondary Table
    st.markdown("---")
    st.subheader("📈 Time-Trend Analysis")

    time_df = load_data_from_rds(SECONDARY_TABLE, date_col_filter='date')
    time_df['date'] = pd.to_datetime(time_df['date'], errors='coerce')
    time_df = time_df.dropna(subset=['date'])  # Drop rows with invalid dates
    time_df = time_df[time_df['date'] >= pd.Timestamp(SIX_MONTHS_AGO)]

    if not time_df.empty:
        min_d = time_df['date'].min().date()
        max_d = time_df['date'].max().date()
        trend_date_range = st.slider("Select Trend Date Range", min_value=min_d, max_value=max_d, value=(min_d, max_d))

        # Filter by selected date range
        start_ts = pd.Timestamp(trend_date_range[0])
        end_ts = pd.Timestamp(trend_date_range[1])
        time_df = time_df[(time_df['date'] >= start_ts) & (time_df['date'] <= end_ts)]

        # Identify numeric metrics
        numeric_cols = time_df.select_dtypes(include='number').columns.tolist()
        metrics = [col for col in numeric_cols if col not in ['id']]
        if not metrics:
            st.warning("No numeric metrics available for trend analysis.")
            return

        selected_metric = st.selectbox("Select metric to trend:", metrics)

        # Ensure selected metric is numeric (again)
        time_df[selected_metric] = pd.to_numeric(time_df[selected_metric], errors='coerce')

        # Group by options
        group_options = ['None']
        if 'client_type' in time_df.columns:
            group_options.append('client_type')
        if 'workflow' in time_df.columns:
            group_options.append('workflow')

        group_by = st.selectbox("Group trend by", group_options)

        if group_by == 'None':
            trend_df = time_df.groupby('date', as_index=False)[selected_metric].mean()
            fig_trend = px.line(trend_df, x='date', y=selected_metric, title=f"{selected_metric} Trend")
        else:
            trend_df = time_df.groupby(['date', group_by], as_index=False)[selected_metric].mean()
            fig_trend = px.line(trend_df, x='date', y=selected_metric, color=group_by,
                                title=f"{selected_metric} Trend by {group_by}")

        st.plotly_chart(fig_trend)
    else:
        st.warning("No time trend data available for the selected date range.")


# -------------------------#
# MAIN EXECUTION
# -------------------------#
if __name__ == "__main__":
    df = load_data_from_rds(SOURCE_TABLE)
    pipeline, X, y, enriched_df = prepare_features(df)
    model, X_test, y_test = train_model(pipeline, X, y)
    add_reason_tags(enriched_df)
    launch_dashboard(enriched_df, model=model, X_test=X_test, y_test=y_test)