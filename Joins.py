

import streamlit as st
import seaborn as sns
st.title("Seaborn Graphs")

tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

# 1. Lineplot
fig = sns.lineplot(data=tips, x="total_bill", y="tip")
st.pyplot(fig.figure)

# 2. Barplot
fig = sns.barplot(data=tips, x="day", y="total_bill")
st.pyplot(fig.figure)

# 3. Boxplot
fig = sns.boxplot(data=tips, x="day", y="tip")
st.pyplot(fig.figure)

# 4. Violin Plot
fig = sns.violinplot(data=tips, x="day", y="tip")
st.pyplot(fig.figure)

# 5. Swarmplot
fig = sns.swarmplot(data=tips, x="day", y="tip")
st.pyplot(fig.figure)

# 6. Stripplot
fig = sns.stripplot(data=tips, x="day", y="tip", jitter=True)
st.pyplot(fig.figure)

# 7. Countplot
fig = sns.countplot(data=tips, x="day")
st.pyplot(fig.figure)

# 8. Histogram
fig = sns.histplot(data=tips, x="total_bill", kde=True)
st.pyplot(fig.figure)

# 9. KDE Plot
fig = sns.kdeplot(data=tips["total_bill"])
st.pyplot(fig.figure)

# 10. Pairplot
fig = sns.pairplot(iris)
st.pyplot(fig)

# 11. Heatmap
fig = sns.heatmap(tips.select_dtypes(include='number').corr(), annot=True)
st.pyplot(fig.figure)

# 12. Jointplot
import seaborn as sns; sns.set_theme(style="white")
g = sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex")
st.pyplot(g.fig)

# 13. Regression Plot
fig = sns.regplot(data=tips, x="total_bill", y="tip")
st.pyplot(fig.figure)

# 14. Catplot
g = sns.catplot(data=tips, x="day", y="tip", kind="box")
st.pyplot(g.fig)

# 15. FacetGrid
g = sns.FacetGrid(tips, col="time")
g.map(sns.histplot, "tip")
st.pyplot(g.fig)

# 16. LMplot
g = sns.lmplot(data=tips, x="total_bill", y="tip", col="sex")
st.pyplot(g.fig)

# 17. Scatterplot
fig = sns.scatterplot(data=iris, x="sepal_length", y="petal_length", hue="species")
st.pyplot(fig.figure)
