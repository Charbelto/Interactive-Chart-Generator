# Chart Visualization Library

This repository contains a Python script that generates 51 different types of charts using `matplotlib`, `seaborn`, `plotly`, `geopandas`, `wordcloud`, and other visualization libraries. It's designed to be an interactive way to explore various data visualization techniques.

## Features

*   **51 Chart Types:** Generate a wide variety of visualizations including scatter plots, histograms, pie charts, network graphs, maps, and more.
*   **Interactive Selection:** Users can choose a chart to display from a numbered list via a simple command-line interface.
*   **Library Integration:** Utilizes popular Python visualization libraries for diverse plotting capabilities.
*   **Error Handling:** Gracefully handles missing library dependencies and rendering issues.
*   **Modular Structure:** Code is organized for easy understanding and modification.

## Chart Types Included

The following 51 chart types are available:

1.  Scatter Plot
2.  Bubble Plot
3.  Connected Scatter Plot
4.  Correlogram
5.  Heatmap
6.  Network Graph
7.  Chord Diagram
8.  Arc Diagram
9.  Edge Bundling
10. Connection Map
11. 3D Surface Plot
12. Beeswarm Plot
13. Candlestick Chart
14. Map
15. Draw Arrow
16. Histogram
17. Density Plot
18. Box Plot
19. Violin Plot
20. Ridgeline Plot
21. 2D Density Plot
22. Hexbin Map
23. Statistics Chart
24. Pie Chart
25. Donut Chart
26. Treemap
27. Venn Diagram
28. Circular Packing
29. Sankey Diagram
30. Waffle Chart
31. Wordcloud
32. Choropleth Map
33. Bubble Map
34. Cartogram
35. Vertical Bar Plot
36. Horizontal Bar Plot
37. Lollipop Plot
38. Parallel Plot
39. Spider/Radar Chart
40. Line Chart
41. Area Plot
42. Stacked Area Plot
43. Stream Graph
44. Time Series Chart
45. Circular Bar Plot
46. Dendrogram
47. Animation (Placeholder)
48. Hierarchical Edge Bundling (Placeholder)
49. Tree Diagram
50. Stacked Bar Chart
51. Line Plot Example

## Setup

### Prerequisites

*   **Python 3.6+:** Make sure you have Python 3.6 or higher installed. You can download it from the [official Python website](https://www.python.org/downloads/).
*   **Pip:** Python package installer should be installed with your python installation.
*   **Git:** (Optional) If you want to clone the repo.

### Installation

1.  **Clone the Repository** (Optional):
    ```bash
    git clone https://github.com/Charbelto/Interactive-Chart-Generator.git
    cd Interactive-Chart-Generator
    ```
2.  **Install Dependencies:**
    ```bash
    pip install matplotlib numpy pandas seaborn matplotlib-venn scipy squarify wordcloud geopandas plotly
    ```

## Usage

1.  **Navigate to the directory with the `main.py` file in your terminal**
2.  **Run the script:**
    ```bash
    python main.py
    ```
3.  **Choose a Chart:** The script will display a numbered list of available chart types. Enter the corresponding number for the chart you wish to see.
4.  **View Chart:** The chosen chart will be generated. Close the window to choose another.
5.  **Repeat or Exit:** Enter 0 to stop running the script.

## Example

After running the script, you will be prompted to enter a chart number:

```
Enter the number of the chart you want to view (or 0 to exit): 1
```
Entering `1` will generate a Scatter Plot.

## Notes

*   Some charts (like Chord Diagram, Sankey Diagram, Maps) require specific data formats and might render placeholders due to the lack of real-world datasets and are simplified for demonstration purposes.
*   Certain chart types, notably those utilizing `plotly` library, may generate separate browser windows to display charts.
*   For more advanced customization and accurate plots, refer to the individual documentation of the libraries used.

## Contributing

Feel free to submit pull requests for bug fixes or new features.  
If you have any ideas for new charts, please raise an issue, and we can discuss it.
