import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.collections import PatchCollection
from matplotlib import cm
from scipy.cluster import hierarchy
import squarify
import random
from matplotlib_venn import venn2
from mpl_toolkits.mplot3d import Axes3D
from wordcloud import WordCloud
import geopandas
import plotly.graph_objects as go
import plotly.express as px

# Define a dictionary with chart names and their corresponding functions
chart_functions = {
    1: "Scatter Plot",
    2: "Bubble Plot",
    3: "Connected Scatter Plot",
    4: "Correlogram",
    5: "Heatmap",
    6: "Network Graph",
    7: "Chord Diagram",
    8: "Arc Diagram",
    9: "Edge Bundling",
    10: "Connection Map",
    11: "3D Surface Plot",
    12: "Beeswarm Plot",
    13: "Candlestick Chart",
    14: "Map",
    15: "Draw Arrow",
    16: "Histogram",
    17: "Density Plot",
    18: "Box Plot",
    19: "Violin Plot",
    20: "Ridgeline Plot",
    21: "2D Density Plot",
    22: "Hexbin Map",
    23: "Statistics Chart",
    24: "Pie Chart",
    25: "Donut Chart",
    26: "Treemap",
    27: "Venn Diagram",
    28: "Circular Packing",
    29: "Sankey Diagram",
    30: "Waffle Chart",
    31: "Wordcloud",
    32: "Choropleth Map",
    33: "Bubble Map",
    34: "Cartogram",
    35: "Vertical Bar Plot",
    36: "Horizontal Bar Plot",
    37: "Lollipop Plot",
    38: "Parallel Plot",
    39: "Spider/Radar Chart",
    40: "Line Chart",
    41: "Area Plot",
    42: "Stacked Area Plot",
    43: "Stream Graph",
    44: "Time Series Chart",
    45: "Circular Bar Plot",
    46: "Dendrogram",
    47: "Animation (Placeholder)",
    48: "Hierarchical Edge Bundling (Placeholder)",
    49: "Tree Diagram",
    50: "Stacked Bar Chart",
    51: "Line Plot Example"
}

def plot_chart(chart_number):
    plt.figure(figsize=(8, 5))
    if chart_number == 1:  # Scatter Plot
      x = np.random.rand(50)
      y = np.random.rand(50)
      plt.scatter(x, y)
      plt.title("Scatter Plot")
      plt.show()

    elif chart_number == 2:  # Bubble Plot
      x = np.random.rand(30)
      y = np.random.rand(30)
      sizes = np.random.rand(30) * 500
      plt.scatter(x, y, s=sizes, alpha=0.5)
      plt.title("Bubble Plot")
      plt.show()

    elif chart_number == 3:  # Connected Scatter Plot
      x = np.sort(np.random.rand(20) * 10)
      y = np.random.rand(20) * 10
      plt.plot(x, y, marker='o', linestyle='-')
      plt.title("Connected Scatter Plot")
      plt.show()

    elif chart_number == 4:  # Correlogram
      df = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])
      sns.pairplot(df)
      plt.title("Correlogram")
      plt.show()

    elif chart_number == 5:  # Heatmap
      data = np.random.rand(10, 10)
      plt.imshow(data, cmap='viridis')
      plt.colorbar()
      plt.title("Heatmap")
      plt.show()

    elif chart_number == 6:  # Network Graph
      points = np.random.rand(20, 2)
      for i in range(20):
          for j in range(i + 1, 20):
              if np.random.rand() < 0.3:
                  plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], color='gray', linewidth=0.5)
      plt.scatter(points[:, 0], points[:, 1], s=50)
      plt.title("Network Graph")
      plt.show()

    elif chart_number == 7:  # Chord Diagram
      try:
          data_matrix = np.random.rand(4,4)
          fig = go.Figure(data=[go.Chord(
                  matrix = data_matrix,
                  labels = ['A', 'B', 'C', 'D']
          )])
          fig.show()
          plt.title("Chord Diagram")
      except:
        print("Chord Diagram could not be rendered")

    elif chart_number == 8:  # Arc Diagram
      categories = ['A', 'B', 'C', 'D', 'E']
      links = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 4)]
      positions = np.linspace(0, 1, len(categories))
      for i, j in links:
          start = positions[i]
          end = positions[j]
          x = np.linspace(start, end, 100)
          y = 0.1*np.sin(np.pi * (x - start) / (end - start))
          plt.plot(x, y, color='gray')
      plt.scatter(positions, np.zeros(len(categories)), s=100)
      plt.xticks(positions, categories)
      plt.title("Arc Diagram")
      plt.show()

    elif chart_number == 9:  # Edge Bundling
      plt.plot([0, 1], [0, 1], marker='o')
      plt.plot([0, 0.8], [0.1,0.9], marker="o", linestyle="--")
      plt.plot([0, 0.9], [0.1,0.5], marker="o", linestyle="--")
      plt.title("Edge Bundling (Placeholder)")
      plt.show()

    elif chart_number == 10:  # Connection Map
      cities = ['City A', 'City B', 'City C', 'City D']
      coords = np.random.rand(4, 2)
      for i in range(4):
          for j in range(i + 1, 4):
              if np.random.rand() < 0.5:
                  plt.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color='gray', linewidth=0.5)
      plt.scatter(coords[:, 0], coords[:, 1], s=100)
      for i, city in enumerate(cities):
          plt.text(coords[i, 0], coords[i, 1], city, ha='left', va='bottom')
      plt.title("Connection Map")
      plt.show()

    elif chart_number == 11:  # 3D Surface Plot
      fig = plt.figure(figsize=(8, 5))
      ax = fig.add_subplot(111, projection='3d')
      x = np.linspace(-5, 5, 100)
      y = np.linspace(-5, 5, 100)
      X, Y = np.meshgrid(x, y)
      Z = np.sin(np.sqrt(X**2 + Y**2))
      ax.plot_surface(X, Y, Z, cmap='viridis')
      ax.set_title("3D Surface Plot")
      plt.show()

    elif chart_number == 12: # Beeswarm Plot
      data = np.random.rand(100)
      sns.swarmplot(x=data)
      plt.title("Beeswarm Plot")
      plt.show()

    elif chart_number == 13: # Candlestick Chart
      dates = pd.to_datetime(pd.Series(np.arange(10)))
      opens = np.random.rand(10) * 10
      highs = opens + np.random.rand(10) * 2
      lows = opens - np.random.rand(10) * 2
      closes = opens + np.random.randn(10)
      for i in range(10):
          plt.vlines(dates[i], lows[i], highs[i], color='black', linewidth=1)
          if closes[i] > opens[i]:
              plt.bar(dates[i], closes[i] - opens[i], bottom=opens[i], width=0.4, color='green')
          else:
              plt.bar(dates[i], opens[i] - closes[i], bottom=closes[i], width=0.4, color='red')
      plt.title("Candlestick Chart")
      plt.show()

    elif chart_number == 14: # Map
      try:
         world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
         fig, ax = plt.subplots(figsize=(8, 5))
         world.plot(ax=ax)
         plt.title("Map")
         plt.show()
      except:
         print("Map Chart could not be rendered")

    elif chart_number == 15:  # Draw Arrow
        plt.plot([0, 1], [0, 1], marker='o')
        plt.annotate('', xy=(0.8, 0.8), xytext=(0.2, 0.2),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        plt.title("Draw Arrow")
        plt.show()

    elif chart_number == 16: # Histogram
        data = np.random.randn(1000)
        plt.hist(data, bins=30)
        plt.title("Histogram")
        plt.show()

    elif chart_number == 17:  # Density Plot
        data = np.random.randn(1000)
        sns.kdeplot(data)
        plt.title("Density Plot")
        plt.show()

    elif chart_number == 18: # Box Plot
        data = np.random.rand(100, 4)
        plt.boxplot(data)
        plt.title("Box Plot")
        plt.show()

    elif chart_number == 19: # Violin Plot
        data = np.random.rand(100, 4)
        sns.violinplot(data=data)
        plt.title("Violin Plot")
        plt.show()

    elif chart_number == 20: # Ridgeline Plot
        for i in range(5):
          data = np.random.normal(0,1,100)
          sns.kdeplot(data + i, fill=True, alpha=0.5)
        plt.title("Ridgeline Plot")
        plt.show()

    elif chart_number == 21: # 2D Density Plot
        x = np.random.randn(500)
        y = np.random.randn(500)
        sns.kdeplot(x=x, y=y)
        plt.title("2D Density Plot")
        plt.show()

    elif chart_number == 22: # Hexbin Map
        x = np.random.rand(500)
        y = np.random.rand(500)
        plt.hexbin(x, y, gridsize=15)
        plt.colorbar()
        plt.title("Hexbin Map")
        plt.show()

    elif chart_number == 23:  # Statistics Chart
        data = np.random.rand(10, 2)
        plt.errorbar(data[:,0],data[:,1], yerr=0.2, fmt='o')
        plt.title("Statistics Chart")
        plt.show()

    elif chart_number == 24: # Pie Chart
        labels = ['A', 'B', 'C', 'D']
        sizes = np.random.rand(4)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%')
        plt.title("Pie Chart")
        plt.show()

    elif chart_number == 25: # Donut Chart
        labels = ['A', 'B', 'C', 'D']
        sizes = np.random.rand(4)
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', wedgeprops=dict(width=0.5), startangle=90)
        plt.title("Donut Chart")
        plt.show()

    elif chart_number == 26: # Treemap
        data = np.random.rand(10) * 100
        labels = [f'Label {i}' for i in range(10)]
        squarify.plot(sizes=data, label=labels, alpha=.8 )
        plt.axis('off')
        plt.title("Treemap")
        plt.show()

    elif chart_number == 27: # Venn Diagram
      venn2(subsets=(3, 2, 1))
      plt.title("Venn Diagram")
      plt.show()

    elif chart_number == 28: # Circular Packing
        data = np.random.rand(20)
        for i, size in enumerate(data):
            circle = plt.Circle(xy=(np.cos(i/len(data)*2*np.pi)*np.sqrt(i), np.sin(i/len(data)*2*np.pi)*np.sqrt(i)), radius=size*0.2, facecolor=cm.viridis(size), edgecolor='black')
            plt.gca().add_patch(circle)
        plt.axis('equal')
        plt.axis('off')
        plt.title("Circular Packing")
        plt.show()

    elif chart_number == 29: # Sankey Diagram
        try:
            fig = go.Figure(data=[go.Sankey(
            node = dict(
              pad = 15,
              thickness = 20,
              line = dict(color = "black", width = 0.5),
              label = ["A1", "A2", "B1", "B2", "C1","C2"]
            ),
            link = dict(
              source = [0, 1, 0, 2, 3, 3],
              target = [2, 3, 3, 4, 4, 5],
              value = [8, 4, 2, 8, 4, 2]
            ))])
            fig.show()
            plt.title("Sankey Diagram")
        except:
            print("Sankey Diagram could not be rendered")

    elif chart_number == 30:  # Waffle Chart
        data = np.random.randint(0, 5, 10)
        labels = [f'Label {i}' for i in range(10)]
        total = sum(data)
        grid = np.zeros((10, 10))
        c = 0
        for i, num in enumerate(data):
            for j in range(num):
                x = c % 10
                y = c // 10
                grid[y][x] = i+1
                c += 1
        for row in range(10):
            for col in range(10):
                if grid[row][col] != 0:
                    plt.gca().add_patch(plt.Rectangle((col,row), width=1, height=1, facecolor=cm.viridis(grid[row][col]/10)))
        plt.xticks([])
        plt.yticks([])
        plt.title("Waffle Chart")
        plt.show()

    elif chart_number == 31: # Wordcloud
        try:
            text = " ".join([f"word{i}" for i in range(1000)])
            wordcloud = WordCloud().generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title("Wordcloud")
            plt.show()
        except:
            print("Wordcloud Chart could not be rendered")

    elif chart_number == 32: # Choropleth Map
        try:
            world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
            world['random_values'] = np.random.rand(len(world))
            fig, ax = plt.subplots(figsize=(8, 5))
            world.plot(column='random_values', ax=ax, legend=True)
            plt.title("Choropleth Map")
            plt.show()
        except:
            print("Choropleth Map Chart could not be rendered")

    elif chart_number == 33: # Bubble Map
        try:
            world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
            fig, ax = plt.subplots(figsize=(8, 5))
            world.plot(ax=ax)
            x = np.random.rand(10)
            y = np.random.rand(10)
            sizes = np.random.rand(10)*500
            ax.scatter(x,y,s=sizes)
            plt.title("Bubble Map")
            plt.show()
        except:
            print("Bubble Map Chart could not be rendered")

    elif chart_number == 34: # Cartogram
        plt.scatter(np.random.rand(50), np.random.rand(50), s=np.random.rand(50)*500, alpha=0.5)
        plt.title("Cartogram (Placeholder)")
        plt.show()

    elif chart_number == 35: # Vertical Bar Plot
        categories = ['A', 'B', 'C', 'D']
        values = np.random.rand(4)
        plt.bar(categories, values)
        plt.title("Vertical Bar Plot")
        plt.show()

    elif chart_number == 36: # Horizontal Bar Plot
      categories = ['A', 'B', 'C', 'D']
      values = np.random.rand(4)
      plt.barh(categories, values)
      plt.title("Horizontal Bar Plot")
      plt.show()

    elif chart_number == 37:  # Lollipop Plot
        categories = ['A', 'B', 'C', 'D']
        values = np.random.rand(4)
        plt.stem(categories, values)
        plt.title("Lollipop Plot")
        plt.show()

    elif chart_number == 38:  # Parallel Plot
        data = np.random.rand(10, 4)
        for i in range(10):
            plt.plot(range(4), data[i])
        plt.xticks(range(4), ['Var1', 'Var2', 'Var3', 'Var4'])
        plt.title("Parallel Plot")
        plt.show()

    elif chart_number == 39:  # Spider/Radar Chart
        categories = ['A', 'B', 'C', 'D', 'E']
        values = np.random.rand(5)
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        values = list(values)
        values += values[:1]
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories)
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.3)
        plt.title("Spider/Radar Chart")
        plt.show()

    elif chart_number == 40:  # Line Chart
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        plt.plot(x, y1, label="Sine", color="blue")
        plt.plot(x, y2, label="Cosine", color="red", linestyle="--")
        plt.title("Line Chart")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.grid(True)
        plt.show()

    elif chart_number == 41: # Area Plot
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)
        plt.fill_between(x, y1, alpha=0.4, label="Sine Area")
        plt.fill_between(x, y2, alpha=0.4, label="Cosine Area")
        plt.title("Area Plot")
        plt.legend()
        plt.show()

    elif chart_number == 42: # Stacked Area Plot
      x = np.linspace(0, 10, 100)
      y1 = np.random.rand(100)
      y2 = np.random.rand(100)
      y3 = np.random.rand(100)
      plt.stackplot(x, y1, y2, y3, labels=["Data1", "Data2", "Data3"])
      plt.title("Stacked Area Plot")
      plt.legend()
      plt.show()

    elif chart_number == 43: # Stream Graph
      x = np.linspace(0, 10, 100)
      y1 = np.random.rand(100)
      y2 = np.random.rand(100)
      y3 = np.random.rand(100)
      plt.fill_between(x, 0, y1, alpha=0.6)
      plt.fill_between(x, y1, y1+y2, alpha=0.6)
      plt.fill_between(x, y1+y2, y1+y2+y3, alpha=0.6)
      plt.title("Stream Graph")
      plt.show()

    elif chart_number == 44: # Time Series Chart
        dates = pd.date_range('2023-01-01', periods=100)
        values = np.random.randn(100).cumsum()
        plt.plot(dates, values)
        plt.title("Time Series Chart")
        plt.show()

    elif chart_number == 45: # Circular Bar Plot
        categories = ['A', 'B', 'C', 'D']
        values = np.random.rand(4)
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        values = list(values)
        values += values[:1]
        ax = plt.subplot(111, polar=True)
        bars = ax.bar(angles[:-1], values[:-1], width=0.5)
        plt.xticks(angles[:-1], categories)
        plt.title("Circular Bar Plot")
        plt.show()

    elif chart_number == 46:  # Dendrogram
      data = np.random.rand(10, 2)
      linked = hierarchy.linkage(data, 'single')
      hierarchy.dendrogram(linked, orientation='top', labels=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
      plt.title("Dendrogram")
      plt.show()

    elif chart_number == 47:  # Animation Placeholder
        plt.plot([0, 1], [0, 1])
        plt.title("Animation (Placeholder)")
        plt.show()

    elif chart_number == 48:  # Hierarchical Edge Bundling Placeholder
        plt.plot([0, 1], [0, 1], marker='o')
        plt.plot([0, 0.8], [0.1,0.9], marker="o", linestyle="--")
        plt.plot([0, 0.9], [0.1,0.5], marker="o", linestyle="--")
        plt.title("Hierarchical Edge Bundling (Placeholder)")
        plt.show()

    elif chart_number == 49:  # Tree Diagram
        plt.plot([0,1], [0,1], marker="o")
        plt.plot([1,2], [1,0], marker="o")
        plt.plot([1,2], [1,2], marker="o")
        plt.title("Tree Diagram")
        plt.show()

    elif chart_number == 50: # Stacked Bar Chart
        categories = ['A', 'B', 'C', 'D']
        data1 = np.random.rand(4)
        data2 = np.random.rand(4)
        data3 = np.random.rand(4)
        plt.bar(categories, data1, label="Data1")
        plt.bar(categories, data2, bottom=data1, label="Data2")
        plt.bar(categories, data3, bottom=data1+data2, label="Data3")
        plt.title("Stacked Bar Chart")
        plt.legend()
        plt.show()

    elif chart_number == 51: # Line Plot Example
      x = np.linspace(0, 10, 100)
      y1 = np.sin(x)
      y2 = np.cos(x)
      plt.plot(x, y1, label="Sine", color="blue")
      plt.plot(x, y2, label="Cosine", color="red", linestyle="--")
      plt.title("Line Plot Example")
      plt.xlabel("X-axis")
      plt.ylabel("Y-axis")
      plt.legend()
      plt.grid(True)
      plt.show()
    else:
        print("Invalid chart number.")

if __name__ == "__main__":
    print("Available Charts:")
    for key, value in chart_functions.items():
        print(f"{key}. {value}")

    while True:
      try:
          choice = int(input("Enter the number of the chart you want to view (or 0 to exit): "))
          if choice == 0:
            break
          plot_chart(choice)
      except ValueError:
          print("Invalid input, Please enter a number.")