import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats.mstats import winsorize

# Set page configuration
st.set_page_config(layout="wide", page_title="Home Credit Default Risk Dashboard")

# --- Global Data Loading and Preprocessing ---
@st.cache_data
def load_data(path):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(path)

    # 1. Feature Engineering
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365.25
    df['EMPLOYMENT_YEARS'] = df['EMPLOYMENT_YEARS'].replace(df['EMPLOYMENT_YEARS'].max(), np.nan)
    df['DTI'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['LOAN_TO_INCOME'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_TO_CREDIT'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # 2. Missing Value Handling
    missing_percentages = df.isnull().mean() * 100
    columns_to_drop = missing_percentages[missing_percentages > 60].index
    df = df.drop(columns=columns_to_drop)
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Categorical Data Handling (Merge rare labels)
    for col in df.select_dtypes(include='object').columns:
        value_counts = df[col].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < 0.01].index
        if not rare_categories.empty:
            df[col] = df[col].replace(rare_categories, 'Other')

    # 4. Outlier Handling (Winsorizing)
    for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY']:
        if col in df.columns:
            df[col] = winsorize(df[col], limits=[0.01, 0.01])

    # 5. Binning
    df['INCOME_BRACKET'] = pd.qcut(df['AMT_INCOME_TOTAL'], q=[0, 0.25, 0.75, 1], labels=['Low', 'Mid', 'High'], duplicates='drop')

    return df

try:
    df = load_data('application_train.csv')
except FileNotFoundError:
    st.error("Error: The 'application_train.csv' file was not found. Please ensure it is in the same directory as the script.")
    st.stop()

# --- Sidebar Filters ---
st.sidebar.title("Global Filters")
st.sidebar.markdown("---")

# Use a global state to manage filters
if 'filters' not in st.session_state:
    st.session_state.filters = {}

# Filter creation functions
def create_multiselect_filter(col_name, label, options):
    selected = st.sidebar.multiselect(label, options, default=options)
    st.session_state.filters[col_name] = selected

def create_slider_filter(col_name, label, min_val, max_val):
    selected = st.sidebar.slider(label, min_val, max_val, (min_val, max_val))
    st.session_state.filters[col_name] = selected

# Define filter options based on the preprocessed data
age_min, age_max = float(df['AGE_YEARS'].min()), float(df['AGE_YEARS'].max())
employment_min, employment_max = float(df['EMPLOYMENT_YEARS'].min()), float(df['EMPLOYMENT_YEARS'].max()) if not df['EMPLOYMENT_YEARS'].isnull().all() else 0.0

create_multiselect_filter('CODE_GENDER', 'Gender', sorted(df['CODE_GENDER'].unique()))
create_multiselect_filter('NAME_EDUCATION_TYPE', 'Education Level', sorted(df['NAME_EDUCATION_TYPE'].unique()))
create_multiselect_filter('NAME_FAMILY_STATUS', 'Family Status', sorted(df['NAME_FAMILY_STATUS'].unique()))
create_multiselect_filter('NAME_HOUSING_TYPE', 'Housing Type', sorted(df['NAME_HOUSING_TYPE'].unique()))
create_slider_filter('AGE_YEARS', 'Age Range', int(age_min), int(age_max))
create_multiselect_filter('INCOME_BRACKET', 'Income Bracket', sorted(df['INCOME_BRACKET'].unique()))
create_slider_filter('EMPLOYMENT_YEARS', 'Employment Tenure (Years)', int(employment_min), int(employment_max))

# Apply filters
def apply_filters(data):
    filtered_df = data.copy()
    for col, value in st.session_state.filters.items():
        if isinstance(value, tuple):
            min_val, max_val = value
            filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
        else:
            filtered_df = filtered_df[filtered_df[col].isin(value)]
    return filtered_df

filtered_df = apply_filters(df)
if filtered_df.empty:
    st.warning("No data matches the selected filters. Please adjust the filter settings.")
    st.stop()

# --- Navigation ---
st.sidebar.header("Dashboard Pages")
page = st.sidebar.radio("Go to", ["1. Project Overview & Data Quality",
                                  "2. Target & Risk Segmentation",
                                  "3. Demographics & Household Profile",
                                  "4. Financial Health & Affordability",
                                  "5. Correlations, Drivers & Interactive Slice-and-Dice"])

# --- Page 1: Overview & Data Quality ---
if page == "1. Project Overview & Data Quality":
    st.title("Page 1: Project Overview & Data Quality 📊")
    st.markdown("---")
    st.header("Key Performance Indicators (KPIs)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 1. Total Applicants
    with col1:
        total_applicants = len(filtered_df)
        st.metric("Total Applicants", f"{total_applicants:,}")
    
    # 2. Default Rate (%)
    with col2:
        default_rate = filtered_df['TARGET'].mean() * 100
        st.metric("Default Rate (%)", f"{default_rate:.2f}%")
        
    # 3. Repaid Rate (%)
    with col3:
        repaid_rate = (1 - filtered_df['TARGET'].mean()) * 100
        st.metric("Repaid Rate (%)", f"{repaid_rate:.2f}%")
        
    # 4. Total Features
    with col4:
        total_features = df.shape[1]
        st.metric("Total Features", f"{total_features}")
        
    # 5. Avg Missing per Feature (%) - Calculated on the original df
    with col5:
        original_df = pd.read_csv('application_train.csv')
        avg_missing = original_df.isnull().mean().mean() * 100
        st.metric("Avg Missing per Feature (%)", f"{avg_missing:.2f}%")
        
    col6, col7, col8, col9, col10 = st.columns(5)
    
    # 6. # Numerical Features
    with col6:
        num_features = len(df.select_dtypes(include=np.number).columns)
        st.metric("Numerical Features", f"{num_features}")
    
    # 7. # Categorical Features
    with col7:
        cat_features = len(df.select_dtypes(include='object').columns)
        st.metric("Categorical Features", f"{cat_features}")
        
    # 8. Median Age (Years)
    with col8:
        median_age = filtered_df['AGE_YEARS'].median()
        st.metric("Median Age (Years)", f"{median_age:.2f}")
    
    # 9. Median Annual Income
    with col9:
        median_income = filtered_df['AMT_INCOME_TOTAL'].median()
        st.metric("Median Annual Income", f"${median_income:,.0f}")
        
    # 10. Average Credit Amount
    with col10:
        avg_credit = filtered_df['AMT_CREDIT'].mean()
        st.metric("Average Credit Amount", f"${avg_credit:,.0f}")

    st.markdown("---")
    st.header("Graphs")
    
    col1, col2 = st.columns(2)
    
    # 1. Pie/Donut Chart: Target distribution
    with col1:
        st.subheader("Target Distribution")
        target_counts = filtered_df['TARGET'].map({0: 'Repaid', 1: 'Defaulted'}).value_counts()
        fig = px.pie(values=target_counts.values, names=target_counts.index, hole=0.5,
                     title="Target Distribution (Repaid vs. Defaulted)")
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Bar Chart: Top 20 features by missing %
    with col2:
        st.subheader("Top 20 Features by Missingness")
        original_df = pd.read_csv('application_train.csv')
        missing_data = original_df.isnull().mean() * 100
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False).head(20)
        fig = px.bar(x=missing_data.index, y=missing_data.values,
                     labels={'x': 'Feature', 'y': 'Missing Percentage (%)'},
                     title="Top 20 Features with Missing Values")
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4, col5 = st.columns(3)
    
    # 3. Histogram: AGE_YEARS
    with col3:
        st.subheader("Age Distribution")
        fig = px.histogram(filtered_df, x='AGE_YEARS', nbins=50, title="Distribution of Applicant Ages")
        st.plotly_chart(fig, use_container_width=True)
        
    # 4. Histogram: AMT_INCOME_TOTAL
    with col4:
        st.subheader("Annual Income Distribution")
        fig = px.histogram(filtered_df, x='AMT_INCOME_TOTAL', nbins=50, title="Distribution of Annual Income")
        st.plotly_chart(fig, use_container_width=True)
        
    # 5. Histogram: AMT_CREDIT
    with col5:
        st.subheader("Credit Amount Distribution")
        fig = px.histogram(filtered_df, x='AMT_CREDIT', nbins=50, title="Distribution of Credit Amount")
        st.plotly_chart(fig, use_container_width=True)
    
    col6, col7 = st.columns(2)
    
    # 6. Boxplot: AMT_INCOME_TOTAL
    with col6:
        st.subheader("Income Distribution (Boxplot)")
        fig = px.box(filtered_df, y='AMT_INCOME_TOTAL', title="Boxplot of Annual Income")
        st.plotly_chart(fig, use_container_width=True)
    
    # 7. Boxplot: AMT_CREDIT
    with col7:
        st.subheader("Credit Amount Distribution (Boxplot)")
        fig = px.box(filtered_df, y='AMT_CREDIT', title="Boxplot of Credit Amount")
        st.plotly_chart(fig, use_container_width=True)
    
    col8, col9, col10 = st.columns(3)
    
    # 8. Countplot: CODE_GENDER
    with col8:
        st.subheader("Gender Distribution")
        gender_counts = filtered_df['CODE_GENDER'].value_counts()
        fig = px.bar(x=gender_counts.index, y=gender_counts.values,
                     labels={'x': 'Gender', 'y': 'Count'},
                     title="Applicant Gender Counts")
        st.plotly_chart(fig, use_container_width=True)
        
    # 9. Countplot: NAME_FAMILY_STATUS
    with col9:
        st.subheader("Family Status Distribution")
        family_status_counts = filtered_df['NAME_FAMILY_STATUS'].value_counts()
        fig = px.bar(x=family_status_counts.index, y=family_status_counts.values,
                     labels={'x': 'Family Status', 'y': 'Count'},
                     title="Applicant Family Status Counts")
        st.plotly_chart(fig, use_container_width=True)
        
    # 10. Countplot: NAME_EDUCATION_TYPE
    with col10:
        st.subheader("Education Distribution")
        education_counts = filtered_df['NAME_EDUCATION_TYPE'].value_counts()
        fig = px.bar(x=education_counts.index, y=education_counts.values,
                     labels={'x': 'Education Level', 'y': 'Count'},
                     title="Applicant Education Counts")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.header("Narrative Insights")
    st.markdown("""
    * The majority of the applicants have successfully repaid their loans, indicated by the *low default rate*. This suggests a generally low-risk portfolio.
    * The *age and income distributions* are right-skewed, with most applicants being in their 30s and 40s and having incomes below $200,000.
    * Key variables like EXT_SOURCE_1 and EXT_SOURCE_2 still have a significant percentage of missing values, which can be a red flag for model building and requires careful handling.
    """)

# --- Page 2: Target & Risk Segmentation ---
elif page == "2. Target & Risk Segmentation":
    st.title("Page 2: Target & Risk Segmentation 🎯")
    st.markdown("---")
    st.header("Key Performance Indicators (KPIs)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Filter for defaulters
    defaulters_df = filtered_df[filtered_df['TARGET'] == 1]
    
    # 1. Total Defaults
    with col1:
        total_defaults = defaulters_df.shape[0]
        st.metric("Total Defaults", f"{total_defaults:,}")
    
    # 2. Default Rate (%)
    with col2:
        default_rate = filtered_df['TARGET'].mean() * 100
        st.metric("Default Rate (%)", f"{default_rate:.2f}%")
    
    # 3. Default Rate by Gender (%)
    with col3:
        def_rate_gender = filtered_df.groupby('CODE_GENDER')['TARGET'].mean() * 100
        if not def_rate_gender.empty:
            for gender, rate in def_rate_gender.items():
                st.metric(f"Def. Rate ({gender})", f"{rate:.2f}%")
            
    # 4. Default Rate by Education (%)
    with col4:
        def_rate_edu = filtered_df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean() * 100
        if not def_rate_edu.empty:
            st.metric("Def. Rate (Highest)", f"{def_rate_edu.max():.2f}%", help=f"Highest default rate is for '{def_rate_edu.idxmax()}'")
        
    # 5. Default Rate by Family Status (%)
    with col5:
        def_rate_fam = filtered_df.groupby('NAME_FAMILY_STATUS')['TARGET'].mean() * 100
        if not def_rate_fam.empty:
            st.metric("Def. Rate (Highest)", f"{def_rate_fam.max():.2f}%", help=f"Highest default rate is for '{def_rate_fam.idxmax()}'")

    col6, col7, col8, col9, col10 = st.columns(5)
    
    # 6. Avg Income — Defaulters
    with col6:
        avg_income_defaulters = defaulters_df['AMT_INCOME_TOTAL'].mean()
        st.metric("Avg Income (Defaulters)", f"${avg_income_defaulters:,.0f}")
        
    # 7. Avg Credit — Defaulters
    with col7:
        avg_credit_defaulters = defaulters_df['AMT_CREDIT'].mean()
        st.metric("Avg Credit (Defaulters)", f"${avg_credit_defaulters:,.0f}")
    
    # 8. Avg Annuity — Defaulters
    with col8:
        avg_annuity_defaulters = defaulters_df['AMT_ANNUITY'].mean()
        st.metric("Avg Annuity (Defaulters)", f"${avg_annuity_defaulters:,.0f}")
        
    # 9. Avg Employment (Years) — Defaulters
    with col9:
        avg_emp_defaulters = defaulters_df['EMPLOYMENT_YEARS'].mean()
        st.metric("Avg Emp. Years (Def.)", f"{avg_emp_defaulters:.2f}")
    
    # 10. Default Rate by Housing Type (%)
    with col10:
        def_rate_housing = filtered_df.groupby('NAME_HOUSING_TYPE')['TARGET'].mean() * 100
        if not def_rate_housing.empty:
            st.metric("Def. Rate (Highest)", f"{def_rate_housing.max():.2f}%", help=f"Highest default rate is for '{def_rate_housing.idxmax()}'")
        
    st.markdown("---")
    st.header("Graphs")
    
    col1, col2 = st.columns(2)
    
    # 1. Bar Chart: Counts - Default vs Repaid
    with col1:
        st.subheader("Default vs. Repaid Counts")
        target_counts = filtered_df['TARGET'].value_counts().reset_index()
        target_counts.columns = ['TARGET', 'count']
        target_counts['TARGET'] = target_counts['TARGET'].map({0: 'Repaid', 1: 'Defaulted'})
        fig = px.bar(target_counts, x='TARGET', y='count', color='TARGET', title="Count of Defaulted vs. Repaid Loans")
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Bar Chart: Default % by Gender
    with col2:
        st.subheader("Default Rate by Gender")
        default_rate_gender = filtered_df.groupby('CODE_GENDER')['TARGET'].mean().reset_index()
        fig = px.bar(default_rate_gender, x='CODE_GENDER', y='TARGET',
                     labels={'CODE_GENDER': 'Gender', 'TARGET': 'Default Rate'},
                     title="Default Rate by Gender")
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    # 3. Bar Chart: Default % by Education
    with col3:
        st.subheader("Default Rate by Education")
        default_rate_edu = filtered_df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().sort_values().reset_index()
        fig = px.bar(default_rate_edu, x='NAME_EDUCATION_TYPE', y='TARGET',
                     labels={'NAME_EDUCATION_TYPE': 'Education Level', 'TARGET': 'Default Rate'},
                     title="Default Rate by Education Level")
        st.plotly_chart(fig, use_container_width=True)
    
    # 4. Bar Chart: Default % by Family Status
    with col4:
        st.subheader("Default Rate by Family Status")
        default_rate_fam = filtered_df.groupby('NAME_FAMILY_STATUS')['TARGET'].mean().sort_values().reset_index()
        fig = px.bar(default_rate_fam, x='NAME_FAMILY_STATUS', y='TARGET',
                     labels={'NAME_FAMILY_STATUS': 'Family Status', 'TARGET': 'Default Rate'},
                     title="Default Rate by Family Status")
        st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)
    
    # 5. Bar Chart: Default % by Housing Type
    with col5:
        st.subheader("Default Rate by Housing Type")
        default_rate_housing = filtered_df.groupby('NAME_HOUSING_TYPE')['TARGET'].mean().sort_values().reset_index()
        fig = px.bar(default_rate_housing, x='NAME_HOUSING_TYPE', y='TARGET',
                     labels={'NAME_HOUSING_TYPE': 'Housing Type', 'TARGET': 'Default Rate'},
                     title="Default Rate by Housing Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # 6. Boxplot: Income by Target
    with col6:
        st.subheader("Income by Target")
        fig = px.box(filtered_df, x='TARGET', y='AMT_INCOME_TOTAL', color='TARGET',
                     title="Income Distribution by Target (0=Repaid, 1=Defaulted)")
        st.plotly_chart(fig, use_container_width=True)
    
    col7, col8 = st.columns(2)
    
    # 7. Boxplot: Credit by Target
    with col7:
        st.subheader("Credit by Target")
        fig = px.box(filtered_df, x='TARGET', y='AMT_CREDIT', color='TARGET',
                     title="Credit Amount Distribution by Target (0=Repaid, 1=Defaulted)")
        st.plotly_chart(fig, use_container_width=True)
        
    # 8. Violin Plot: Age vs Target
    with col8:
        st.subheader("Age Distribution by Target")
        fig = px.violin(filtered_df, y='AGE_YEARS', x='TARGET', color='TARGET', box=True,
                        title="Age Distribution by Target (0=Repaid, 1=Defaulted)")
        st.plotly_chart(fig, use_container_width=True)
        
    col9, col10 = st.columns(2)
    
    # 9. Histogram (stacked): EMPLOYMENT_YEARS by Target
    with col9:
        st.subheader("Employment Years by Target")
        df_plot = filtered_df.dropna(subset=['EMPLOYMENT_YEARS'])
        fig = px.histogram(df_plot, x='EMPLOYMENT_YEARS', color='TARGET',
                           labels={'EMPLOYMENT_YEARS': 'Employment Years', 'TARGET': 'Default'},
                           title="Employment Years Distribution by Target",
                           barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
    
    # 10. Stacked Bar: NAME_CONTRACT_TYPE vs Target
    with col10:
        st.subheader("Contract Type vs. Target")
        contract_target_df = filtered_df.groupby(['NAME_CONTRACT_TYPE', 'TARGET']).size().reset_index(name='count')
        fig = px.bar(contract_target_df, x='NAME_CONTRACT_TYPE', y='count', color='TARGET',
                     labels={'NAME_CONTRACT_TYPE': 'Contract Type', 'count': 'Count', 'TARGET': 'Default'},
                     title="Contract Type Distribution by Target")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.header("Narrative Insights")
    st.markdown("""
    * Applicants with *lower levels of education* and those in a *'Single / not married'* family status appear to have a higher default rate.
    * The median income and credit amount are slightly lower for defaulters, though the distributions show significant overlap. This suggests that financial metrics alone are not sufficient to predict default.
    * There is a clear pattern in employment years; a large number of defaults come from applicants with a few years of employment, while those with long-term employment seem to be a lower risk.
    """)

# --- Page 3: Demographics & Household Profile ---
elif page == "3. Demographics & Household Profile":
    st.title("Page 3: Demographics & Household Profile 👨‍👩‍👧‍👦")
    st.markdown("---")
    st.header("Key Performance Indicators (KPIs)")

    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 1. % Male vs Female
    with col1:
        gender_pct = filtered_df['CODE_GENDER'].value_counts(normalize=True) * 100
        st.metric("% Male", f"{gender_pct.get('M', 0):.2f}%")
        st.metric("% Female", f"{gender_pct.get('F', 0):.2f}%")
        
    # 2. Avg Age - Defaulters
    with col2:
        avg_age_def = filtered_df[filtered_df['TARGET'] == 1]['AGE_YEARS'].mean()
        st.metric("Avg Age (Defaulters)", f"{avg_age_def:.2f}")
        
    # 3. Avg Age - Non-Defaulters
    with col3:
        avg_age_non_def = filtered_df[filtered_df['TARGET'] == 0]['AGE_YEARS'].mean()
        st.metric("Avg Age (Non-Defaulters)", f"{avg_age_non_def:.2f}")

    # 4. % With Children
    with col4:
        with_children_pct = (filtered_df['CNT_CHILDREN'] > 0).mean() * 100
        st.metric("% With Children", f"{with_children_pct:.2f}%")
        
    # 5. Avg Family Size
    with col5:
        avg_fam_size = filtered_df['CNT_FAM_MEMBERS'].mean()
        st.metric("Avg Family Size", f"{avg_fam_size:.2f}")

    col6, col7, col8, col9, col10 = st.columns(5)
    
    # 6. % Married vs Single
    with col6:
        married_pct = (filtered_df['NAME_FAMILY_STATUS'] == 'Married').mean() * 100
        single_pct = (filtered_df['NAME_FAMILY_STATUS'] == 'Single / not married').mean() * 100
        st.metric("% Married", f"{married_pct:.2f}%")
        st.metric("% Single", f"{single_pct:.2f}%")
        
    # 7. % Higher Education
    with col7:
        higher_edu_pct = (filtered_df['NAME_EDUCATION_TYPE'].isin(['Higher education'])).mean() * 100
        st.metric("% Higher Education", f"{higher_edu_pct:.2f}%")

    # 8. % Living With Parents
    with col8:
        parents_pct = (filtered_df['NAME_HOUSING_TYPE'] == 'With parents').mean() * 100
        st.metric("% Living With Parents", f"{parents_pct:.2f}%")
        
    # 9. % Currently Working (using DAYS_EMPLOYED)
    with col9:
        working_pct = (filtered_df['DAYS_EMPLOYED'] != 365243).mean() * 100
        st.metric("% Currently Working", f"{working_pct:.2f}%")
        
    # 10. Avg Employment Years
    with col10:
        avg_emp_years = filtered_df['EMPLOYMENT_YEARS'].mean()
        st.metric("Avg Employment Years", f"{avg_emp_years:.2f}")

    st.markdown("---")
    st.header("Graphs")
    
    col1, col2 = st.columns(2)
    
    # 1. Histogram: Age distribution (all)
    with col1:
        st.subheader("Age Distribution")
        fig = px.histogram(filtered_df, x='AGE_YEARS', nbins=50, title="Distribution of Applicant Ages")
        st.plotly_chart(fig, use_container_width=True)
        
    # 2. Histogram: Age by Target (overlay)
    with col2:
        st.subheader("Age Distribution by Target")
        fig = px.histogram(filtered_df, x='AGE_YEARS', color='TARGET', nbins=50,
                           title="Age Distribution by Target", barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4, col5 = st.columns(3)
    
    # 3. Bar: Gender distribution
    with col3:
        st.subheader("Gender Distribution")
        gender_counts = filtered_df['CODE_GENDER'].value_counts()
        fig = px.bar(x=gender_counts.index, y=gender_counts.values, title="Gender Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
    # 4. Bar: Family Status distribution
    with col4:
        st.subheader("Family Status Distribution")
        family_counts = filtered_df['NAME_FAMILY_STATUS'].value_counts()
        fig = px.bar(x=family_counts.index, y=family_counts.values, title="Family Status Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
    # 5. Bar: Education distribution
    with col5:
        st.subheader("Education Distribution")
        edu_counts = filtered_df['NAME_EDUCATION_TYPE'].value_counts()
        fig = px.bar(x=edu_counts.index, y=edu_counts.values, title="Education Distribution")
        st.plotly_chart(fig, use_container_width=True)

    col6, col7 = st.columns(2)
    
    # 6. Bar: Occupation distribution (top 10)
    with col6:
        st.subheader("Top 10 Occupations")
        top_occupations = filtered_df['OCCUPATION_TYPE'].value_counts().head(10).index
        df_top_occ = filtered_df[filtered_df['OCCUPATION_TYPE'].isin(top_occupations)]
        fig = px.bar(df_top_occ, x='OCCUPATION_TYPE', title="Top 10 Applicant Occupations")
        st.plotly_chart(fig, use_container_width=True)
    
    # 7. Pie: Housing Type distribution
    with col7:
        st.subheader("Housing Type Distribution")
        housing_counts = filtered_df['NAME_HOUSING_TYPE'].value_counts()
        fig = px.pie(values=housing_counts.values, names=housing_counts.index,
                     title="Applicant Housing Type")
        st.plotly_chart(fig, use_container_width=True)

    col8, col9, col10 = st.columns(3)
    
    # 8. Countplot: CNT_CHILDREN
    with col8:
        st.subheader("Number of Children")
        children_counts = filtered_df['CNT_CHILDREN'].value_counts()
        fig = px.bar(x=children_counts.index, y=children_counts.values, title="Distribution of Children")
        st.plotly_chart(fig, use_container_width=True)
        
    # 9. Boxplot: Age vs Target
    with col9:
        st.subheader("Age vs. Target")
        fig = px.box(filtered_df, x='TARGET', y='AGE_YEARS', color='TARGET', title="Age Distribution by Default Status")
        st.plotly_chart(fig, use_container_width=True)
    
    # 10. Heatmap: Corr(Age, Children, Family Size, TARGET)
    with col10:
        st.subheader("Demographic Correlations")
        demo_cols = ['AGE_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'TARGET']
        corr_matrix = filtered_df[demo_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.columns,
                                        colorscale='Viridis'))
        fig.update_layout(title="Correlation Matrix: Demographics")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.header("Narrative Insights")
    st.markdown("""
    * Applicants in their late 20s and early 30s form a significant portion of the applicant pool and appear to have higher risk.
    * There is a small positive correlation between having more children and the TARGET variable, suggesting that larger families may be associated with a slightly higher default risk.
    * The distribution of applicants across family and education status highlights the need for targeted risk models for different life stages and socioeconomic backgrounds.
    """)

# --- Page 4: Financial Health & Affordability ---
elif page == "4. Financial Health & Affordability":
    st.title("Page 4: Financial Health & Affordability 💰")
    st.markdown("---")
    st.header("Key Performance Indicators (KPIs)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 1. Avg Annual Income
    with col1:
        avg_income = filtered_df['AMT_INCOME_TOTAL'].mean()
        st.metric("Avg Annual Income", f"${avg_income:,.0f}")
        
    # 2. Median Annual Income
    with col2:
        med_income = filtered_df['AMT_INCOME_TOTAL'].median()
        st.metric("Median Annual Income", f"${med_income:,.0f}")
    
    # 3. Avg Credit Amount
    with col3:
        avg_credit = filtered_df['AMT_CREDIT'].mean()
        st.metric("Avg Credit Amount", f"${avg_credit:,.0f}")
        
    # 4. Avg Annuity
    with col4:
        avg_annuity = filtered_df['AMT_ANNUITY'].mean()
        st.metric("Avg Annuity", f"${avg_annuity:,.0f}")
    
    # 5. Avg Goods Price
    with col5:
        avg_goods = filtered_df['AMT_GOODS_PRICE'].mean()
        st.metric("Avg Goods Price", f"${avg_goods:,.0f}")
        
    col6, col7, col8, col9, col10 = st.columns(5)
    
    # 6. Avg DTI
    with col6:
        avg_dti = filtered_df['DTI'].mean()
        st.metric("Avg DTI", f"{avg_dti:.2f}")
        
    # 7. Avg Loan-to-Income (LTI)
    with col7:
        avg_lti = filtered_df['LOAN_TO_INCOME'].mean()
        st.metric("Avg Loan-to-Income", f"{avg_lti:.2f}")
    
    # 8. Income Gap (Non-def - Def)
    with col8:
        income_gap = filtered_df[filtered_df['TARGET'] == 0]['AMT_INCOME_TOTAL'].mean() - filtered_df[filtered_df['TARGET'] == 1]['AMT_INCOME_TOTAL'].mean()
        st.metric("Income Gap (Non-def - Def)", f"${income_gap:,.0f}")
        
    # 9. Credit Gap (Non-def - Def)
    with col9:
        credit_gap = filtered_df[filtered_df['TARGET'] == 0]['AMT_CREDIT'].mean() - filtered_df[filtered_df['TARGET'] == 1]['AMT_CREDIT'].mean()
        st.metric("Credit Gap (Non-def - Def)", f"${credit_gap:,.0f}")
        
    # 10. % High Credit (> 1M)
    with col10:
        high_credit_pct = (filtered_df['AMT_CREDIT'] > 1_000_000).mean() * 100
        st.metric("% High Credit (>1M)", f"{high_credit_pct:.2f}%")

    st.markdown("---")
    st.header("Graphs")
    
    col1, col2, col3 = st.columns(3)
    
    # 1. Histogram: Income distribution
    with col1:
        st.subheader("Income Distribution")
        fig = px.histogram(filtered_df, x='AMT_INCOME_TOTAL', title="Annual Income Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
    # 2. Histogram: Credit distribution
    with col2:
        st.subheader("Credit Distribution")
        fig = px.histogram(filtered_df, x='AMT_CREDIT', title="Credit Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)
        
    # 3. Histogram: Annuity distribution
    with col3:
        st.subheader("Annuity Distribution")
        fig = px.histogram(filtered_df, x='AMT_ANNUITY', title="Annuity Amount Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    col4, col5 = st.columns(2)
    
    # 4. Scatter: Income vs Credit
    with col4:
        st.subheader("Income vs. Credit Amount")
        fig = px.scatter(filtered_df, x='AMT_INCOME_TOTAL', y='AMT_CREDIT',
                         title="Income vs. Credit (with alpha blending)", opacity=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    # 5. Scatter: Income vs Annuity
    with col5:
        st.subheader("Income vs. Annuity")
        fig = px.scatter(filtered_df, x='AMT_INCOME_TOTAL', y='AMT_ANNUITY',
                         title="Income vs. Annuity (with alpha blending)", opacity=0.3)
        st.plotly_chart(fig, use_container_width=True)
        
    col6, col7, col8 = st.columns(3)
    
    # 6. Boxplot: Credit by Target
    with col6:
        st.subheader("Credit by Target")
        fig = px.box(filtered_df, x='TARGET', y='AMT_CREDIT', color='TARGET', title="Credit by Default Status")
        st.plotly_chart(fig, use_container_width=True)
        
    # 7. Boxplot: Income by Target
    with col7:
        st.subheader("Income by Target")
        fig = px.box(filtered_df, x='TARGET', y='AMT_INCOME_TOTAL', color='TARGET', title="Income by Default Status")
        st.plotly_chart(fig, use_container_width=True)
    
    # 8. KDE / Density: Joint Income-Credit
    with col8:
        st.subheader("Joint Density: Income & Credit")
        fig = go.Figure(data=go.Contour(z=filtered_df['AMT_CREDIT'], x=filtered_df['AMT_INCOME_TOTAL'],
                                        colorscale='Blues'))
        fig.update_layout(title="Joint Density of Income and Credit", xaxis_title="Income", yaxis_title="Credit")
        st.plotly_chart(fig, use_container_width=True)
        
    col9, col10 = st.columns(2)
    
    # 9. Bar: Income Brackets vs Default Rate
    with col9:
        st.subheader("Default Rate by Income Bracket")
        income_bracket_default = filtered_df.groupby('INCOME_BRACKET')['TARGET'].mean().reset_index()
        fig = px.bar(income_bracket_default, x='INCOME_BRACKET', y='TARGET',
                     labels={'TARGET': 'Default Rate', 'INCOME_BRACKET': 'Income Bracket'},
                     title="Default Rate by Income Bracket")
        st.plotly_chart(fig, use_container_width=True)

    # 10. Heatmap: Financial variable correlations
    with col10:
        st.subheader("Financial Variable Correlations")
        fin_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DTI', 'LOAN_TO_INCOME', 'TARGET']
        corr_matrix = filtered_df[fin_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.columns,
                                        colorscale='Viridis'))
        fig.update_layout(title="Correlation Matrix: Financials")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Narrative Insights")
    st.markdown("""
    * The scatter plots show a strong positive relationship between income, credit, and annuity, which is expected.
    * The default rate is highest for applicants in the *'Low' income bracket*, which intuitively makes sense as they have less financial flexibility.
    * The DTI and LTI ratios are important affordability indicators. As seen in the correlation matrix, they have a positive correlation with TARGET, suggesting that higher debt and loan burdens are associated with higher default risk.
    """)

# --- Page 5: Correlations, Drivers & Interactive Slice-and-Dice ---
elif page == "5. Correlations, Drivers & Interactive Slice-and-Dice":
    st.title("Page 5: Correlations, Drivers & Interactive Slice-and-Dice 🔍")
    st.markdown("---")
    st.header("Key Performance Indicators (KPIs)")
    
    # Pre-calculate correlations for KPIs
    numeric_df = filtered_df.select_dtypes(include=np.number)
    correlations = numeric_df.corr()['TARGET'].sort_values(key=abs, ascending=False)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # 1. Top 5 +Corr with TARGET
    with col1:
        top_pos_corr = correlations[correlations > 0].head(5)
        st.subheader("Top 5 +Corr with TARGET")
        for feature, corr_val in top_pos_corr.items():
            st.write(f"- {feature}: {corr_val:.2f}")

    # 2. Top 5 -Corr with TARGET
    with col2:
        top_neg_corr = correlations[correlations < 0].head(5)
        st.subheader("Top 5 -Corr with TARGET")
        for feature, corr_val in top_neg_corr.items():
            st.write(f"- {feature}: {corr_val:.2f}")

    # 3. Most correlated with Income
    with col3:
        income_corr = numeric_df.corr()['AMT_INCOME_TOTAL'].sort_values(key=abs, ascending=False).drop('AMT_INCOME_TOTAL').head(1)
        st.metric("Most Correlated with Income", f"{income_corr.index[0]} ({income_corr.values[0]:.2f})")
    
    # 4. Most correlated with Credit
    with col4:
        credit_corr = numeric_df.corr()['AMT_CREDIT'].sort_values(key=abs, ascending=False).drop('AMT_CREDIT').head(1)
        st.metric("Most Correlated with Credit", f"{credit_corr.index[0]} ({credit_corr.values[0]:.2f})")
        
    # 5. Corr(Income, Credit)
    with col5:
        inc_credit_corr = filtered_df[['AMT_INCOME_TOTAL', 'AMT_CREDIT']].corr().iloc[0, 1]
        st.metric("Corr(Income, Credit)", f"{inc_credit_corr:.2f}")

    col6, col7, col8, col9, col10 = st.columns(5)
    
    # 6. Corr(Age, TARGET)
    with col6:
        age_target_corr = filtered_df[['AGE_YEARS', 'TARGET']].corr().iloc[0, 1]
        st.metric("Corr(Age, TARGET)", f"{age_target_corr:.2f}")
        
    # 7. Corr(Employment Years, TARGET)
    with col7:
        emp_target_corr = filtered_df[['EMPLOYMENT_YEARS', 'TARGET']].corr().iloc[0, 1]
        st.metric("Corr(Emp. Years, TARGET)", f"{emp_target_corr:.2f}")
        
    # 8. Corr(Family Size, TARGET)
    with col8:
        fam_target_corr = filtered_df[['CNT_FAM_MEMBERS', 'TARGET']].corr().iloc[0, 1]
        st.metric("Corr(Family Size, TARGET)", f"{fam_target_corr:.2f}")
        
    # 9. Variance Explained by Top 5 Features (proxy via |corr|)
    with col9:
        top5_corr_abs_sum = correlations.head(5).abs().sum()
        st.metric("Sum |Corr| of Top 5", f"{top5_corr_abs_sum:.2f}")
    
    # 10. # Features with |corr| > 0.5
    with col10:
        high_corr_features = (correlations.abs() > 0.5).sum()
        st.metric("# Features with |Corr| > 0.5", f"{high_corr_features}")

    st.markdown("---")
    st.header("Graphs")
    
    col1, col2 = st.columns(2)
    
    # 1. Heatmap: Correlation (selected numerics)
    with col1:
        st.subheader("Correlation Heatmap")
        selected_cols = ['TARGET', 'AGE_YEARS', 'EMPLOYMENT_YEARS', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'DTI', 'LOAN_TO_INCOME', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS']
        corr_matrix = filtered_df[selected_cols].corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                        x=corr_matrix.columns,
                                        y=corr_matrix.columns,
                                        colorscale='Viridis'))
        fig.update_layout(title="Correlation Matrix of Key Features")
        st.plotly_chart(fig, use_container_width=True)

    # 2. Bar: |Correlation| of features vs TARGET (top N)
    with col2:
        st.subheader("Feature Correlation with TARGET")
        corrs_abs = filtered_df.corr(numeric_only=True)['TARGET'].abs().sort_values(ascending=False).drop('TARGET').head(10)
        fig = px.bar(x=corrs_abs.index, y=corrs_abs.values,
                     labels={'x': 'Feature', 'y': '|Correlation|'},
                     title="Top 10 Features by Absolute Correlation with TARGET")
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    # 3. Scatter: Age vs Credit (hue=TARGET)
    with col3:
        st.subheader("Age vs. Credit (by Target)")
        fig = px.scatter(filtered_df, x='AGE_YEARS', y='AMT_CREDIT', color='TARGET',
                         title="Age vs. Credit Amount", opacity=0.3)
        st.plotly_chart(fig, use_container_width=True)
        
    # 4. Scatter: Age vs Income (hue=TARGET)
    with col4:
        st.subheader("Age vs. Income (by Target)")
        fig = px.scatter(filtered_df, x='AGE_YEARS', y='AMT_INCOME_TOTAL', color='TARGET',
                         title="Age vs. Annual Income", opacity=0.3)
        st.plotly_chart(fig, use_container_width=True)
    
    col5, col6, col7 = st.columns(3)
    
    # 5. Scatter: Employment Years vs TARGET
    with col5:
        st.subheader("Employment Years vs. Target")
        fig = px.box(filtered_df.dropna(subset=['EMPLOYMENT_YEARS']),
                     x='TARGET', y='EMPLOYMENT_YEARS', color='TARGET',
                     title="Employment Years by Default Status")
        st.plotly_chart(fig, use_container_width=True)
    
    # 6. Boxplot: Credit by Education
    with col6:
        st.subheader("Credit Amount by Education")
        fig = px.box(filtered_df, x='NAME_EDUCATION_TYPE', y='AMT_CREDIT',
                     title="Credit Amount by Education Level")
        st.plotly_chart(fig, use_container_width=True)
        
    # 7. Boxplot: Income by Family Status
    with col7:
        st.subheader("Income by Family Status")
        fig = px.box(filtered_df, x='NAME_FAMILY_STATUS', y='AMT_INCOME_TOTAL',
                     title="Income by Family Status")
        st.plotly_chart(fig, use_container_width=True)

    col8, col9, col10 = st.columns(3)
    
    # 8. Pair Plot: Income, Credit, Annuity, TARGET
    with col8:
        st.subheader("Pair Plot of Financials")
        sample_df = filtered_df.sample(n=min(5000, len(filtered_df)), random_state=42)
        fig = px.scatter_matrix(sample_df, dimensions=['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY'], color='TARGET',
                                title="Pair Plot of Key Financials")
        st.plotly_chart(fig, use_container_width=True)
    
    # 9. Filtered Bar: Default Rate by Gender
    with col9:
        st.subheader("Default Rate by Gender (Filtered)")
        default_rate_gender = filtered_df.groupby('CODE_GENDER')['TARGET'].mean().reset_index()
        fig = px.bar(default_rate_gender, x='CODE_GENDER', y='TARGET',
                     labels={'CODE_GENDER': 'Gender', 'TARGET': 'Default Rate'},
                     title="Responsive Default Rate by Gender")
        st.plotly_chart(fig, use_container_width=True)
        
    # 10. Filtered Bar: Default Rate by Education
    with col10:
        st.subheader("Default Rate by Education (Filtered)")
        default_rate_edu = filtered_df.groupby('NAME_EDUCATION_TYPE')['TARGET'].mean().reset_index()
        fig = px.bar(default_rate_edu, x='NAME_EDUCATION_TYPE', y='TARGET',
                     labels={'NAME_EDUCATION_TYPE': 'Education Level', 'TARGET': 'Default Rate'},
                     title="Responsive Default Rate by Education")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("Narrative Insights")
    st.markdown("""
    * **EXT_SOURCE** variables are the strongest predictors of default, with a significant negative correlation. This suggests that these external scores are crucial for risk assessment.
    * There is a weak but notable correlation between AGE_YEARS, EMPLOYMENT_YEARS, and TARGET. Younger, less-experienced applicants may be at a higher risk of default.
    * The affordability ratios like DTI and LTI are positively correlated with default, indicating that as these ratios increase, so does the risk. This provides a strong basis for setting policy-based credit caps.
    """)