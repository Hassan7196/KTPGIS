# Image Processing and Geospatial Data Mapping Web Application

This project is a web-based application built with Flask that uses machine learning for image processing and prediction.
Additionally, it provides a feature to visualize geospatial data on a map. T
he application allows users to upload images and geospatial data files, processes them, and displays the results using various Python libraries, including TensorFlow and GeoPandas.

## Features

- **Image Upload and Prediction**: Users can upload images via the web interface. The application processes the images and generates predictions using a pre-trained TensorFlow model.
- **Geospatial Data Visualization**: Users can upload geospatial data files (e.g., shapefiles), and the application will render maps using GeoPandas and Matplotlib.
- **Visualization Tools**: Visualize results using Matplotlib for image data and GeoPandas for geospatial data.

## Installation

To set up and run the project locally, follow these steps:

### Prerequisites

Ensure you have Python 3.x installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Clone the Repository

Clone the repository to your local machine in pycharm:

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

upon completion you will view the following
### Install Dependencies

Install the required Python packages using pip:
```
pip install -r requirements.txt
```

Once everything is download and completed. 
Go to Server.py and 148 line: click on green arrow:
![1](https://github.com/user-attachments/assets/26fd9922-d4e1-4ead-b124-7754dec4932b)





or u can just go to the terminal write the following
```
python server.py
```
![1](https://github.com/user-attachments/assets/1df65ae7-8710-4ff1-9b37-ca664bc977c0)

## Prediction Dataset

Click on the following : https://drive.google.com/drive/folders/1GOzjBxyaY15di9SthQLp06suoHjvK_2z?usp=sharing

go to data->train_image, and upload on the image for prediction
### output
The colours represents the following:
- Cyan - Urban Areas
- Blue - Water
- Green - Forest
- Yellow - Agriculture
- Magneta - Road
![1](https://github.com/user-attachments/assets/a0c007f7-4f02-4a89-b9a4-d7d025786226)

### Visualise the geospatial data

click on the following link: https://drive.google.com/drive/folders/1vgxhZBXednIG_v8rJFZT1txlNVXLdAe7?usp=sharing
download any file , and upload on the geojson file
### Output
![1](https://github.com/user-attachments/assets/1401af95-7751-41e3-9f98-b6badcb0bdb2)
