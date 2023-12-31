#import required libraries
from flask import Flask, render_template, request, jsonify  # Importing Flask library for creating web application routes and rendering templates
import warnings  # Importing warnings library to suppress warning messages
warnings.filterwarnings('ignore')

import datacube  # Importing datacube library for loading and processing satellite imagery datasets efficiently
import numpy as np  # Importing NumPy library for numerical operations
import io  # Importing io library for handling input/output operations
import odc.algo  # Importing odc.algo library for performing specific algorithms on satellite imagery
import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting and visualization
from datacube.utils.cog import write_cog  # Importing write_cog module from datacube.utils.cog for writing Cloud-Optimized GeoTIFFs
import base64  # Importing base64 library for encoding and decoding data in base64 format
import pandas as pd  # Importing Pandas library for data manipulation and analysis
from sklearn.ensemble import RandomForestRegressor  # Importing RandomForestRegressor from scikit-learn for machine learning modeling
from sklearn.model_selection import train_test_split  # Importing train_test_split from scikit-learn for splitting data into training and testing sets
import matplotlib.pyplot as plt  # Importing Matplotlib library for plotting and visualization
from deafrica_tools.plotting import display_map, rgb  # Importing display_map and rgb functions from deafrica_tools.plotting for map visualization

dc = datacube.Datacube(app="04_Plotting") #creates an instance of the Datacube class from the datacube module.
def mang_ml_analysis(ds, lat_range, lon_range):
    ndvi = (ds.nir - ds.red) / (ds.nir + ds.red)
    mvi = ds.nir - ds.green / ds.swir_1 - ds.green
    ndvi_threshold = 0.4

    # Create forest mask based on NDVI
    mangrove_mask_ndvi = np.where(ndvi > ndvi_threshold, 1, 0)

    mvi_threshold = 3.5

    # Create forest mask based on MVI within the threshold range
    mangrove_mask_mvi = np.where(mvi > mvi_threshold, 1, 0)

    regular_mask = np.where(ndvi <= 0.6, True, False)
    closed_mask = np.where(ndvi > 0.6, True, False)

    mangrove = np.logical_and(mangrove_mask_ndvi, mangrove_mask_mvi)
    regular = np.logical_and(mangrove, regular_mask)
    closed = np.logical_and(mangrove, closed_mask)

    # Calculate the area of each pixel
    pixel_area = abs(ds.geobox.affine[0] * ds.geobox.affine[4])

    data = [['day', 'month', 'year', 'mangrove', 'regular', 'closed', 'total']]
    regular_values = []
    closed_values = []

    for i in range(mangrove.shape[0]):
        data_time = str(ndvi.time[i].values).split("T")[0]
        new_data_time = data_time.split("-")

        # Calculate the total mangrove cover area
        mangrove_cover_area = np.sum(mangrove[i]) * pixel_area
        regular_cover_area = np.sum(regular[i]) * pixel_area
        closed_cover_area = np.sum(closed[i]) * pixel_area

        original_array = np.where(ndvi > -10, 1, 0)
        original = np.sum(original_array[i]) * pixel_area
        regular_values.append(regular_cover_area / 1000000)
        closed_values.append(closed_cover_area / 1000000)

        data.append([new_data_time[2], new_data_time[1], new_data_time[0], mangrove_cover_area / 1000000,
                     regular_cover_area / 1000000, closed_cover_area / 1000000, original / 1000000])

    df = pd.DataFrame(data[1:], columns=data[0])
    df["year-month"] = df["year"].astype('str') + "-" + df["month"].astype('str')

    grouped_df = df.groupby(['year', 'month'])
    mean_forest_field = grouped_df['mangrove'].mean()
    mean_forest_field = mean_forest_field.reset_index()

    df = mean_forest_field

    years = df["year"].tolist()

    X = df[["year", "month"]]
    y = df["mangrove"]

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=101)
    rf_regressor.fit(X, y)
    y_pred = rf_regressor.predict(X)

    labels = df["year"].tolist()
    actual_values = df['mangrove'].tolist()
   
    predicted_values = y_pred.tolist()

    return {
        "labels": labels,
        "actual_values": actual_values,
        "predicted_values": predicted_values,
        "regular":regular_values,
        "closed":closed_values
    }

app = Flask(__name__)
# @app.route("/")
# def hi():
#     return render_template("home.html")

@app.route("/")
def hello_world():
        # Read the merged CSV file
        
    
    return render_template("index.html")

@app.route('/my_flask_route', methods=['GET', 'POST'])
def my_flask_function():
    if request.method == "POST":
        lmin = request.json['lat_min']
        lmax = request.json['lat_max']
        lnmin = request.json['lng_min']
        lnmax = request.json['lng_max']
        ty = request.json['t']
        fd1=request.json['t1']
        td1=request.json['t2']
        lat_range = (lmin, lmax)
        lon_range = (lnmin, lnmax)
        print(lat_range, lon_range)
        time_range = (fd1, td1)
        # display_map(x=lon_range, y=lat_range)
        try:
            ds = dc.load(product="s2a_sen2cor_granule",
                            measurements=["red","green","blue", "nir", "swir_1"],
                        x=lon_range,
                        y=lat_range,
                        time=time_range,
                        output_crs='EPSG:3857',
                        resolution=(-30, 30))
            dataset = ds
            dataset =  odc.algo.to_f32(dataset)
            band_diff = dataset.nir - dataset.red
            band_sum = dataset.nir + dataset.red
        except Exception as e:
            return jsonify({'error': "No Data Found"})


        ndvi = band_diff / band_sum
        mvi = (dataset.nir - dataset.green) / (dataset.swir_1 - dataset.green)
        # Calculate NDVI and store it as a measurement in the original dataset
        if (ty=='ndvi'):
            
            plt.figure(figsize=(8, 8))
            ndvi_subplot=ndvi.isel(time=[0,-1])
            ndvi_subplot.plot(col='time', cmap='YlGn', vmin=-1, vmax=1, col_wrap=2)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            # plt.savefig('./static/my_plot.png')
            # Serve the image file in the Flask app
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        elif (ty=='mvi'):
            
            plt.figure(figsize=(8, 8))
            mvi_subplot=mvi.isel(time=[0,-1])
            mvi_subplot.plot(col='time', cmap='cividis', vmin=1, vmax=20, col_wrap=2)
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            # plt.savefig('./static/my_plot.png')
            # Serve the image file in the Flask app
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        elif (ty=='change'):
            time_range1 = ('2022-01-15', '2022-12-15')
            time_range2 = ('2023-01-15', '2023-02-15')

            query = {
                'lat': lat_range,
                'lon': lon_range,
                'time': time_range1,
                'measurements': ["red","green","blue", "nir", "swir_1"],
                'product': 's2a_sen2cor_granule',
                'output_crs': 'EPSG:3875',
                'resolution': (-10, 10)
            }

            # Load the data for the first time period
            ds1 = dc.load(**query)


            # Compute the MVI for the first time period
            mangrove1 = ((ds1.nir - ds1.green) / (ds1.swir_1 - ds1.green))
            # Set threshold for mangrove detection
            mangrove_thresh = 10

            # Create a mangrove mask
            mangrove_mask1 = np.where(mangrove1 > mangrove_thresh, 1, 0)

            # Load the data for the second time period
            query['time'] = time_range2
            ds2 = dc.load(**query)

            # Compute the MVI for the second time period
            mangrove2 = ((ds2.nir - ds2.green) / (ds2.swir_1 - ds2.green))
            # Create a mangrove mask
            mangrove_mask2 = np.where(mangrove2 > mangrove_thresh, 1, 0)

            # Compute the change in mangrove extent
            mangrove_change = mangrove_mask2 - mangrove_mask1

            # Create a colormap
            cmap = plt.get_cmap('PiYG')

            # Plot the change in mangrove extent
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(mangrove_change[-1], cmap=cmap, vmin=-1, vmax=1)
            ax.set_title('Change in Mangrove Extent')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            cbar = fig.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Change in Mangrove Extent')
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            # plt.savefig('./static/my_plot.png')
            # Serve the image file in the Flask app
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        ndvi_threshold = 0.4
        # Create forest mask based on NDVI
        mangrove_mask_ndvi = np.where(ndvi > ndvi_threshold, 1, 0)
        mvi_threshold = 4
        # Create forest mask based on MVI within the threshold range
        mangrove_mask_mvi = np.where(mvi > mvi_threshold, 1, 0)

        mangrove = np.logical_and(mangrove_mask_ndvi, mangrove_mask_mvi)

        # Calculate the area of each pixel
        pixel_area = abs(ds.geobox.affine[0] * ds.geobox.affine[4])
        year=[]
        mangrove1=[]

        for i in range(mangrove.shape[0]):
            data_time = str(ndvi.time[i].values).split("T")[0]
            print(data_time)
            new_data_time = data_time.split("-")           
            year.append(new_data_time[0])

        for i in range(mangrove.shape[0]):
            mangrove_cover_area = np.sum(mangrove[i]) * pixel_area
            mangrove1.append(mangrove_cover_area/1000000)

        
        a = mang_ml_analysis(ds, lat_range, lon_range)
        print(a)
    # Return the base64 encoded PNG image as JSON
        return jsonify({'image': img_base64,'mangrove_data':mangrove1,'year_data':year,'a': a})

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)