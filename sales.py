import streamlit as st

# Set page config
st.set_page_config(layout="wide")

# Centered title
st.markdown("<h1 style='text-align: center;'>Region Level Dashboard</h1>", unsafe_allow_html=True)

# Dropdowns
col1, col2, col3 = st.columns([1, 2, 3])
with col1:
    region = st.selectbox("Region", ["East", "West", "North", "South"])
with col2:
    category = st.selectbox("Category", ["Electronics", "Furniture", "Office Supplies"])
with col3:
    st.write(" ")

# Centered headings without values
col4, col5, col6, col7, col8 = st.columns([1, 1, 1, 1, 1])
with col4:
    st.markdown("**Total Sales**")
with col5:
    st.markdown("**Average Discount**")
with col6:
    st.markdown("**Total Quantity**")
with col7:
    st.markdown("**Total Profit**")
with col8:
    st.write("")

# Divider
st.markdown("---")

# Bottom section
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Sales by order date")
    st.subheader("Discounts by order date")
with chart_col2:
    st.subheader("Units by order date")
    st.subheader("Profits by order date")
