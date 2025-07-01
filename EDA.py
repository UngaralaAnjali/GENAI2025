import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas  as pd
import seaborn as sns
# df=pd.read_csv("path of the csv file")
# x = df['Sales']
# y = df['Units']
# plt.plot(x, y)
# st.pyplot(plt)




st.title("Matplotlib Graphs")

x = np.linspace(0, 10, 100)
y = np.sin(x)

# 1. Line
fig, ax = plt.subplots()
ax.plot(x, y)
st.pyplot(fig)

# 2. Bar
fig, ax = plt.subplots()
ax.bar(["A", "B", "C"], [10, 20, 15])
st.pyplot(fig)

# 3. Histogram
data = np.random.randn(1000)
fig, ax = plt.subplots()
ax.hist(data, bins=30)
st.pyplot(fig)

# 4. Scatter
fig, ax = plt.subplots()
ax.scatter(x, y)
st.pyplot(fig)

# 5. Pie Chart
fig, ax = plt.subplots()
ax.pie([30, 40, 30], labels=["A", "B", "C"], autopct="%1.1f%%")
st.pyplot(fig)

# 6. Boxplot
fig, ax = plt.subplots()
ax.boxplot(np.random.randn(100))
st.pyplot(fig)

# 7. Area Chart
fig, ax = plt.subplots()
ax.fill_between(x, y)
st.pyplot(fig)

# 8. Stem Plot
fig, ax = plt.subplots()
ax.stem(x[:10], y[:10])
st.pyplot(fig)

# 9. Step Plot
fig, ax = plt.subplots()
ax.step(x, y)
st.pyplot(fig)

# 10. Error Bars
fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=0.1)
st.pyplot(fig)

# 11. Heatmap (Matplotlib)
fig, ax = plt.subplots()
c = ax.imshow(np.random.rand(10, 10))
fig.colorbar(c)
st.pyplot(fig)

# 12. Polar Plot
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)
ax.plot(x, np.abs(np.sin(x)))
st.pyplot(fig)

# 13. Subplots
fig, axs = plt.subplots(1, 2)
axs[0].plot(x, np.cos(x))
axs[1].plot(x, np.sin(x))
st.pyplot(fig)

# 14. Logarithmic Plot
fig, ax = plt.subplots()
ax.semilogy(x, np.exp(x/3))
st.pyplot(fig)

# 15. Barh (horizontal bar)
fig, ax = plt.subplots()
ax.barh(["X", "Y", "Z"], [5, 10, 3])
st.pyplot(fig)

